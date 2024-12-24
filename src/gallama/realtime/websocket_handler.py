from gallama.logger.logger import logger
from gallama.realtime.websocket_client import WebSocketClient
from gallama.realtime.websocket_session import WebSocketSession
from fastapi import WebSocket
# from gallama.data_classes.realtime_data_classes import (
#     SessionConfig,
#     ConversationItem,
#     ResponseCreate,
#     ConversationItemCreate,
#     ContentType,
#     MessageContent,
#     ConversationItemMessage,
#     ConversationItemFunctionCall,
#     ConversationItemFunctionCallOutput,
#     ResponseCreateServer,
#     ContentPart,
#     ResponseContentPartAddedEvent,
#     ResponseTextDelta,
#     ResponseTextDone,
#     ConversationItemCreated,
#     ConversationItemMessageServer,
#     ConversationItemServer,
#     ResponseCreated,
#     ResponseOutput_ItemAdded
# )

from gallama.data_classes.realtime_data_classes import *


class WebSocketMessageHandler:
    """
    Handles different message types
    This class is the entry point for all client messages.
    This class should refrain from actually doing any processing, but pass on the job to the appropriate handler.
    """

    def __init__(self, stt_url: str, llm_url: str, tts_url: str):
        self.stt_url = stt_url
        self.llm_url = llm_url
        self.tts_url = tts_url

        self.ws_stt = WebSocketClient(stt_url)
        self.ws_llm = WebSocketClient(llm_url)
        self.ws_tts = WebSocketClient(tts_url)

        self.initialize_ws = False

        self.handlers = {
            # session
            "session.update": self._session_update,

            # audio
            # "input_audio_buffer.append": self._input_audio_buffer_append,
            # "input_audio_buffer.commit": self._input_audio_buffer_commit,
            # "input_audio_buffer.clear": self._input_audio_buffer_clear,

            # converssation
            "conversation.item.create": self._conversation_item_create,
            # "conversation.item.truncate": self._conversation_item_truncate,
            # "conversation.item.delete": self._conversation_item_delete,

            # response
            "response.create": self._response_create,
            # "response.cancel": self._response_cancel
        }

    async def initialize(self):
        await self.ws_stt.connect()

        await self.ws_llm.connect()
        await self.ws_tts.connect()
        self.initialize_ws = True

    async def handle_message(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        """This is the entrypoint for client messages. It dispatches the message to the appropriate handler."""
        handler = self.handlers.get(message["type"])
        if handler:
            await handler(websocket, session, message)
        else:
            logger.warning(f"Unknown message type: {message['type']}")


    async def _session_update(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        update_config = SessionConfig(**message["session"])

        # Don't update voice if it's been used
        # This is because voice will stick for the whole session following OpenAI spec
        # can consider change this down the road

        # TODO support changing voice on the fly
        if 'voice' in message["session"] and session.voice_used:
            del message["session"]["voice"]

        session.config = SessionConfig(**{**session.config.model_dump(), **message["session"]})

        await websocket.send_json({
            "event_id": await session.queues.next_event(),
            "type": "session.updated",
            "session": {
                "id": session.id,
                "object": "realtime.session",
                "model": "default model",
                **session.config.model_dump()
            }
        })

    # TODO
    async def _conversation_item_create(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        # for conversation.item.created, it is a completed item, hence no need to do streaming
        event = ConversationItemCreate(**message)
        await session.queues.unprocessed.put(event.item)

    async def _response_create(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        # add a Conversation Item into unprocess queue
        # once the processing reach this model, it will trigger llm -> tts
        item = ResponseCreate(**message)
        await session.queues.unprocessed.put(item)






