from ..data_classes.realtime_server_proto import (
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    InputAudioBufferCommitted,
    InputAudioBufferCleared,
    MessageContentServer,
    ConversationItemMessageServer,
    ContentTypeServer
)
# from ..logger.logger import logger
from gallama.realtime.websocket_client import WebSocketClient
from gallama.realtime.websocket_session import WebSocketSession
from fastapi import WebSocket
from .vad import VADProcessor
import base64
from gallama.data_classes.realtime_client_proto import *
from gallama.data_classes.internal_ws import *
from ..dependencies_server import get_server_logger
import numpy as np
import time
import asyncio

logger = get_server_logger()

class WebSocketMessageHandler:
    """
    Handles different message types
    This class is the entry point for all client messages.
    This class should refrain from actually doing any processing, but pass on the job to the appropriate handler.
    """
    def __init__(self):
        pass

    def initialize(self, stt_url: str, llm_url: str, tts_url: str):
        self.stt_url = stt_url
        self.llm_url = llm_url
        self.tts_url = tts_url

        self.ws_stt = WebSocketClient(stt_url)
        self.ws_llm = WebSocketClient(llm_url)
        self.ws_tts = WebSocketClient(tts_url)

        self.initialize_ws = False
        self.buffer_lock = asyncio.Lock()

        self.handlers = {
            # session
            "session.update": self._session_update,

            # audio
            "input_audio_buffer.append": self._input_audio_buffer_append,
            "input_audio_buffer.commit": self._input_audio_buffer_commit,
            "input_audio_buffer.clear": self._input_audio_buffer_clear,

            # converssation
            "conversation.item.create": self._conversation_item_create,
            "conversation.item.truncate": self._conversation_item_truncate,
            "conversation.item.delete": self._conversation_item_delete,

            # response
            "response.create": self._response_create,
            "response.cancel": self._response_cancel,

        }

    async def initialize_connection(self):
        await self.ws_stt.connect()

        await self.ws_llm.connect()
        await self.ws_tts.connect()
        self.initialize_ws = True

    async def cleanup(self):
        # send signal to respective web socket to clean data
        await self.ws_llm.send_pydantic_message(WSInterCleanup())
        await self.ws_tts.send_pydantic_message(WSInterCleanup())
        await self.ws_stt.send_pydantic_message(WSInterCleanup())



    async def handle_message(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        """This is the entrypoint for client messages. It dispatches the message to the appropriate handler."""
        handler = self.handlers.get(message["type"])
        if handler:
            await handler(websocket, session, message)
        else:
            logger.warning(f"Unknown message type: {message['type']}")

    async def _session_update(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        update_config = SessionConfig(**message["session"])

        if 'voice' in message["session"] and session.voice_used:
            del message["session"]["voice"]

        session.config = SessionConfig(**{**session.config.model_dump(), **message["session"]})

        # send respective ws update
        await self.ws_llm.send_pydantic_message(WSInterConfigUpdate(config=session.config))
        await self.ws_tts.send_pydantic_message(WSInterConfigUpdate(config=session.config))
        await self.ws_stt.send_pydantic_message(WSInterConfigUpdate(config=session.config))

        logger.info(f"Session updated: {session.config.model_dump()}")


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

    async def _input_audio_buffer_append(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        try:
            # Decode and convert audio once
            audio_bytes = base64.b64decode(message["audio"])
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            await session.queues.append_unprocessed_audio(
                message["audio"],
                ws_stt=self.ws_stt,
                audio_float=audio_float
            )

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise

    async def _input_audio_buffer_commit(self, websocket: WebSocket, session: WebSocketSession, message: dict, item_id: str = None):
        # if is none, get a new item_id
        if item_id is None:
            item_id = await session.queues.next_item()

        # handling internal audio commit
        await session.queues.commit_unprocessed_audio(ws_stt=self.ws_stt, item_id=item_id)

        #   send user committed msg
        await websocket.send_json(InputAudioBufferCommitted(**{
            "event_id": await session.queues.next_event(),
            "type": "input_audio_buffer.committed",
            "previous_item_id": await session.queues.current_item_id(),
            "item_id": item_id
        }).model_dump())


    async def _input_audio_buffer_clear(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        # handling internal audio buffer clear
        await session.queues.clear_unprocessed_audio(ws_stt=self.ws_stt)

        #   send user committed msg
        await websocket.send_json(InputAudioBufferCleared(**{
            "event_id": await session.queues.next_event(),
            "type": "input_audio_buffer.cleared",
        }).model_dump())


    async def _conversation_item_create(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        # for conversation.item.created, it is a completed item, hence no need to do streaming
        event = ConversationItemCreate(**message)
        await session.queues.unprocessed.put(event)


    async def _conversation_item_truncate(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        event = ConversationItemTruncate(**message)
        await session.queues.unprocessed.put(event)

    async def _conversation_item_delete(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        event = ConversationItemDelete(**message)
        await session.queues.unprocessed.put(event.item)


    async def _response_create(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        # add a Conversation Item into unprocess queue
        # once the processing reach this model, it will trigger llm -> tts
        item = ResponseCreate(**message)
        await session.queues.unprocessed.put(item)

    async def _response_cancel(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        """Handle response.cancel event"""
        try:
            async with session.current_response_lock:
                if not session.current_response:
                    logger.warning("No current response to cancel")
                else:
                    # Since response is handled synchronously in process_unprocessed_queue,
                    # we just need to add a cancel event to the queue
                    cancel_event = ResponseCancel(**message)
                    await session.current_response.cancel(cancel_event)

        except Exception as e:
            logger.error(f"Error in response cancellation: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "error": {
                    "code": "internal_error",
                    "message": "Failed to cancel response"
                }
            })

