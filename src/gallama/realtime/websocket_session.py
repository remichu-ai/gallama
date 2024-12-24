import asyncio
from typing import List

from gallama.data_classes.realtime_data_classes import SessionConfig
from gallama.realtime.message_queue import MessageQueues
from fastapi import WebSocket

from gallama.realtime.websocket_client import WebSocketClient


class WebSocketSession:
    """Encapsulates all session-related state and operations"""

    def __init__(self, session_id: str, config: SessionConfig = None):
        self.id = session_id
        self.config = config if config else SessionConfig()
        self.voice_used = False
        self.queues = MessageQueues()
        self.tasks: List[asyncio.Task] = []

    def mark_voice_used(self):
        self.voice_used = True

    async def cleanup(self):
        for task in self.tasks:
            task.cancel()
        # Additional cleanup as needed


    async def session_update(
        self,
        ws_client: WebSocket,
        message: dict,
        ws_stt: WebSocketClient,
        ws_llm: WebSocketClient,
        ws_tts: WebSocketClient,
    ):
        # for fields that not updated, will retain existing setting
        self.config = self.config.merge(message.get("session",{}))

        # send client acknowledgement
        await ws_client.send_json({
            "event_id": await self.queues.next_event(),
            "type": "session.updated",
            "session": {
                "id": self.id,
                "object": "realtime.session",
                "model": "default model",
                **self.config.model_dump()
            }
        })


        # send all internal ws update config
        ws_stt.send_message(self.config.stt)
        ws_llm.send_message(self.config.llm)
        ws_tts.send_message(self.config.tts)