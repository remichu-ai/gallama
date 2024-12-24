import asyncio

from starlette.websockets import WebSocket

from gallama import logger
from gallama.data_classes.realtime_data_classes import ConversationItem, ConversationItemMessage, ContentType, \
    MessageContent, ResponseCreate
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
            # asyncio.create_task(self.process_llm_queue(session, websocket)),
            # asyncio.create_task(self.process_audio_to_client(session, websocket))
        ]

    async def transcribe_audio(self, audio_data: str) -> str:
        """convert audio to text using STT service"""
        """audio_data is a base64 encoded string, convert it to bytes"""
        pass
        #   TODO


    async def send_message_to_client(self, websocket: WebSocket, message: str):
        await websocket.send_json(message)

    async def handle_message_item(self, item: ConversationItem):
        # process item
        processed_item = None
        if isinstance(item, ConversationItemMessage):
            # Create new content list for the updated items
            new_content = []

            for msg in item.content:
                if msg.type == ContentType.INPUT_AUDIO:
                    # Create new message content with transcribed text
                    new_msg = MessageContent(
                        type=msg.type,
                        text=await self.transcribe_audio(msg.audio),
                        audio=msg.audio
                    )
                    new_content.append(new_msg)
                else:
                    new_content.append(msg)

            # Create new item with updated content
            processed_item = ConversationItemMessage(
                id=item.id,
                type=item.type,
                role=item.role,
                content=new_content,
                status=item.status,
            )

        return processed_item

    async def process_unprocessed_queue(self, session: WebSocketSession, websocket: WebSocket):
        """Process items in unprocessed queue one at a time"""
        while True:
            item: ConversationItem | ResponseCreate = await session.queues.unprocessed.get()

            try:
                if isinstance(item, ConversationItem):
                    item = await self.handle_message_item(item)
                    await session.queues.update_conversation_item_ordered_dict(
                        ws_client=websocket,
                        ws_llm=self.message_handler.ws_llm,
                        item=item
                    )
                if isinstance(item, ResponseCreate):
                    # send message to LLM to start generate an answer

                    # get response counter
                    event_id = await session.queues.next_event()
                    response_id = await session.queues.next_resp()

                    # create new response object that will manage this response
                    response = Response(
                        session=session,
                        ws_client=websocket,
                        ws_llm=self.message_handler.ws_llm,
                        ws_tts=self.message_handler.ws_tts,
                        response_id=response_id,
                        event_id_generator=session.queues.next_event
                    )

                    # send ws_stt ResponseCreate event so that it know to complete all the transcription in the queue
                    await self.ws_stt.send_pydantic_message(item)
                    # this function will ensure that user audio transcription finish & sync with llm state
                    await self.stt_complete()

                    # at this point stt already completed, and text send to the llm
                    await response.response_initialize()    # TODO, this can be run in parallel
                    await response.response_start(response_create_request=item, session_config=session.config)

                    await clean_steate()    # TODO


            except Exception as e:
                logger.error(f"Error processing queue item: {str(e)}")
            finally:
                session.queues.unprocessed.task_done()
