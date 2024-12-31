import asyncio
import json
import traceback
from starlette.websockets import WebSocket
from websockets import ConnectionClosed
from gallama.data_classes.realtime_client_proto import (
    SessionConfig,
    ConversationItem,
    ConversationItemMessage,
    ContentType,
    MessageContent,
    ResponseCreate,
    ConversationItemInputAudioTranscriptionComplete,
    ConversationItemCreate,
    ConversationItemTruncate
)
from gallama.data_classes.realtime_server_proto import MessageContentServer, ConversationItemMessageServer, \
    ConversationItemServer
from gallama.data_classes.internal_ws import WSInterSTTResponse, WSInterSTT
from gallama.realtime.response import Response
from gallama.realtime.session_manager import SessionManager
from gallama.realtime.websocket_handler import WebSocketMessageHandler
from gallama.realtime.websocket_session import WebSocketSession
from ..dependencies_server import get_server_logger

logger = get_server_logger()

class WebSocketManager:
    """Orchestrates the overall WebSocket operations"""
    def __init__(self, session_manager: SessionManager, message_handler: WebSocketMessageHandler):
        self.session_manager = session_manager
        self.message_handler = message_handler

    async def initialize_session(self, websocket: WebSocket, model: str, api_key: str = None) -> WebSocketSession:
        protocols = websocket.headers.get("sec-websocket-protocol", "").split(", ")
        if "openai-beta.realtime-v1" in protocols:
            await websocket.accept(subprotocol="openai-beta.realtime-v1")
        else:
            await websocket.accept()

        session_id = str(id(websocket))

        # Create session config with the model parameter
        config = SessionConfig(
            model=model,
            modalities=["text"],  # or ["text", "audio"] if you're supporting audio
            instructions="",
            streaming_transcription=True
        )

        session = self.session_manager.create_session(session_id, config)

        # Format the session data according to OpenAI's expected structure
        session_data = {
            "event_id": await session.queues.next_event(),
            "type": "session.created",
            "session": {
                **config.model_dump(exclude_none=True),  # This ensures null values aren't included
                "id": session_id  # Add the session ID explicitly
            }
        }

        await websocket.send_json(session_data)
        return session

    async def start_background_tasks(self, session: WebSocketSession, websocket: WebSocket):
        """Initialize and start background processing tasks"""

        # start connection with individual ws
        if not self.message_handler.initialize_ws:
            await self.message_handler.initialize()

        session.tasks = [
            asyncio.create_task(self.process_unprocessed_queue(session, websocket)),
            asyncio.create_task(self.process_unprocess_audio_buffer(session, websocket)),
            # asyncio.create_task(self.process_audio_to_client(session, websocket))
        ]

    async def transcribe_audio(self, audio_data: str) -> str:
        """convert audio to text using STT service"""
        """audio_data is a base64 encoded string, convert it to bytes"""
        pass
        #   TODO


    async def send_message_to_client(self, websocket: WebSocket, message: str):
        await websocket.send_json(message)

    async def handle_message_item(self, item: ConversationItem) -> ConversationItemServer:
        # process item
        try:
            processed_item = None
            if isinstance(item, ConversationItemMessage):
                # Create new content list for the updated items
                new_content = []

                for msg in item.content:
                    if msg.type == ContentType.INPUT_AUDIO:
                        # Create new message content with transcribed text
                        new_msg = MessageContentServer(
                            type=msg.type,
                            text=new_msg.text,
                            audio=msg.audio
                        )
                        new_content.append(new_msg)
                    else:
                        new_content.append(MessageContentServer(**msg.model_dump()))

                # Create new item with updated content
                # also convert it to server type
                processed_item = ConversationItemMessageServer(
                    id=item.id,
                    type=item.type,
                    role=item.role,
                    content=new_content,
                    status=item.status if item.status else "completed",
                )

            return processed_item
        except Exception as e:
            logger.error(f"Error in handle_message_item: {str(e)}\n{traceback.format_exc()}")

    async def process_unprocess_audio_buffer(self, session: WebSocketSession, websocket: WebSocket):

        try:
            ws_stt = self.message_handler.ws_stt

            while True:

                if not await ws_stt.ensure_connection():
                    raise Exception("Could not establish STT connection")

                # Receive message from STT
                message = await ws_stt.connection.recv()

                # Parse the message
                try:
                    stt_response = WSInterSTTResponse(**json.loads(message))
                    if stt_response.type == "stt.add_transcription":
                        logger.info(f"Adding transcription: {stt_response.transcription}")
                        await session.queues.append_transcription(stt_response.transcription)

                    elif stt_response.type == "stt.transcription_complete":
                        logger.info(f"Transcription complete")
                        await session.queues.mark_transcription_done()
                        # break     # not break as this is meant to run forever
                    elif stt_response.type == "stt.buffer_cleared":
                        # clear transcription
                        await session.queues.clear_transcription()
                    else:
                        logger.error(f"Unknown stt message type: {stt_response.type}")
                except asyncio.TimeoutError:
                    continue
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse STT message: {message}")
                    continue
                except ConnectionClosed:
                    logger.error("Stt websocket connection closed unexpectedly")
                    await ws_stt.ensure_connection()
                    continue
        except Exception as e:
            logger.error(f"Unexpected error occurred in process_unprocess_audio_buffer: {str(e)}\n{traceback.format_exc()}")
            raise e


    async def process_unprocessed_queue(self, session: WebSocketSession, ws_client: WebSocket):
        """Process items in unprocessed queue one at a time"""
        while True:
            item: ConversationItemCreate | ResponseCreate = await session.queues.unprocessed.get()

            try:
                if isinstance(item, ConversationItemCreate):
                    item_to_create = await self.handle_message_item(item.item)
                    await session.queues.update_conversation_item_ordered_dict(
                        ws_client=ws_client,
                        ws_llm=self.message_handler.ws_llm,
                        item=item_to_create
                    )
                elif isinstance(item, ConversationItemTruncate):
                    await session.queues.truncate_conversation_item(
                        ws_client=ws_client,
                        ws_llm=self.message_handler.ws_llm,
                        event=item,
                        user_interrupt_token=session.config.user_interrupt_token,
                    )
                elif isinstance(item, ResponseCreate):
                    # if there is audio commited, create an item for it

                    modalities = "audio" if await session.queues.audio_exist() and "audio" in session.config.modalities else "text"

                    # get response counter
                    response_id = await session.queues.next_resp()

                    # create new response object that will manage this response
                    response = Response(
                        session=session,
                        ws_client=ws_client,
                        ws_stt=self.message_handler.ws_stt,
                        ws_llm=self.message_handler.ws_llm,
                        ws_tts=self.message_handler.ws_tts,
                        response_id=response_id,
                        event_id_generator=session.queues.next_event
                    )

                    # set the current response for cancellation
                    async with session.current_response_lock:
                        logger.info(f"-------------------Set Current response")
                        session.current_response = response

                    # at this point stt already completed, and text send to the llm
                    await response.response_initialize()
                    await response.response_start(response_create_request=item, session_config=session.config)

                    # update conversation list with assistant answer
                    await session.queues.update_conversation_item_with_assistant_response(
                        response_id=response.item_id,
                        _type=modalities,
                        text=response.text or response.transcription,   # for audio item, fill text with transcription
                        transcript=response.transcription,
                        audio=response.audio,
                        ws_llm=self.message_handler.ws_llm,
                    )

                    await session.queues.reset_after_response()

                    logger.info(f"---------------------Response created: {response_id}")

                    # remove current response
                    async with session.current_response_lock:
                        logger.info(f"-------------------Reset current response")
                        session.current_response = None


            except Exception as e:
                logger.error(f"Error processing queue item: {str(e)}")
            finally:
                session.queues.unprocessed.task_done()
