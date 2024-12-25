import asyncio
import json
import traceback
from starlette.websockets import WebSocket
from sympy.codegen.ast import continue_
from websockets import ConnectionClosed, WebSocketException
from gallama.logger import logger
from gallama.data_classes.realtime_data_classes import (
    ConversationItem,
    ConversationItemMessage,
    ContentType,
    MessageContent,
    ResponseCreate,
    ConversationItemMessageServer,
    MessageContentServer,
    ConversationItemServer,
    ConversationItemInputAudioTranscriptionComplete
)
from gallama.data_classes.internal_ws import WSInterSTTResponse, WSInterSTT
from gallama.realtime.response import Response
from gallama.realtime.session_manager import SessionManager
from gallama.realtime.websocket_handler import WebSocketMessageHandler
from gallama.realtime.websocket_session import WebSocketSession


class WebSocketManager:
    """Orchestrates the overall WebSocket operations"""
    def __init__(self, session_manager: SessionManager, message_handler: WebSocketMessageHandler):
        self.session_manager = session_manager
        self.message_handler = message_handler

    async def initialize_session(self, websocket: WebSocket, model:str, api_key:str = None) -> WebSocketSession:
        protocols = websocket.headers.get("sec-websocket-protocol", "").split(", ")
        if "openai-beta.realtime-v1" in protocols:
            await websocket.accept(subprotocol="openai-beta.realtime-v1")
        else:
            await websocket.accept()

        session_id = str(id(websocket))
        session = self.session_manager.create_session(session_id)

        await websocket.send_json({
            "event_id": await session.queues.next_event(),
            "type": "session.created",
            "session": session.config.model_dump()
        })

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
                    status=item.status,
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

                # Receive message from LLM
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
                    elif stt_response.type == "stt.clear_buffer_done":
                        # clear transcription
                        await session.queues.clear_transcription()
                    else:
                        logger.error(f"Unknown stt message type: {stt_response.type}")
                except asyncio.TimeoutError:
                    continue
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM message: {message}")
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
            item: ConversationItem | ResponseCreate = await session.queues.unprocessed.get()

            try:
                if isinstance(item, ConversationItem):
                    item = await self.handle_message_item(item)
                    await session.queues.update_conversation_item_ordered_dict(
                        ws_client=ws_client,
                        ws_llm=self.message_handler.ws_llm,
                        item=item
                    )
                elif isinstance(item, ResponseCreate):
                    # if there is audio commited, create an item for it

                    modalities = "text"

                    if await session.queues.audio_exist() and "audio" in session.config.modalities:
                        modalities = "audio"
                        if not session.queues.audio_commited:
                            raise Exception("Audio buffer is not commited before create response")
                        # # send signal to ws_stt to complete the transcription
                        # await self.message_handler.ws_stt.send_pydantic_message(
                        #     WSInterSTT(type="stt.sound_done")
                        # )

                        # wait for audio transcription to finish
                        transcription_done = await session.queues.wait_for_transcription_done()

                        if transcription_done and session.queues.transcript_buffer:
                            # create a new item
                            item_id_to_use = await session.queues.next_item()
                            user_audio_item = ConversationItemMessageServer(
                                id=item_id_to_use,
                                type="message",
                                role="user",
                                content=[
                                    MessageContentServer(
                                        type=ContentType.INPUT_AUDIO,
                                        # text=session.queues.transcript_buffer,
                                        audio=session.queues.audio_buffer,
                                        transcript=session.queues.transcript_buffer
                                    )
                                ]
                            )

                            # add this item to the queue
                            await session.queues.update_conversation_item_ordered_dict(
                                ws_client=ws_client,
                                ws_llm=self.message_handler.ws_llm,
                                item=user_audio_item
                            )

                            # send user update that the transcription is done
                            await ws_client.send_json(ConversationItemInputAudioTranscriptionComplete(
                                event_id=await session.queues.next_event(),
                                type="conversation.item.input_audio_transcription.completed",
                                item_id=item_id_to_use,
                                content_index=0,
                                transcript=session.queues.transcript_buffer
                            ).model_dump())


                    # get response counter
                    #event_id = await session.queues.next_event()
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


            except Exception as e:
                logger.error(f"Error processing queue item: {str(e)}")
            finally:
                session.queues.unprocessed.task_done()
