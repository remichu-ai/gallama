import asyncio
from typing import List
from gallama.data_classes.realtime_client_proto import SessionConfig
from gallama.realtime.message_queue import MessageQueues
from..data_classes.internal_ws import SpeechState
from .vad import VADProcessor
from fastapi import WebSocket
import numpy as np
from gallama.realtime.websocket_client import WebSocketClient


class WebSocketSession:
    """Encapsulates all session-related state and operations"""

    def __init__(self, session_id: str, config: SessionConfig = None):
        self.id = session_id
        self.config = config if config else SessionConfig()
        self.voice_used = False
        self.queues = MessageQueues()

        # self.vad_status: Literal["idle", "speech_start", "speech_end"] = "idle"

        self.current_response_lock = asyncio.Lock()
        self.current_response: "Response" = None  # this is to track the current response

        # this list keep track of the 2 main concurrent task to receive send response to user
        self.tasks: List[asyncio.Task] = []

    def mark_voice_used(self):
        self.voice_used = True

    async def cleanup(self):
        # reset internal queue
        self.queues.reset()
        self.current_response = None



    # async def session_update(
    #     self,
    #     ws_client: WebSocket,
    #     message: dict,
    #     ws_stt: WebSocketClient,
    #     ws_llm: WebSocketClient,
    #     ws_tts: WebSocketClient,
    # ):
    #     # for fields that not updated, will retain existing setting
    #     self.config = self.config.merge(message.get("session",{}))
    #
    #     # send client acknowledgement
    #     await ws_client.send_json({
    #         "event_id": await self.queues.next_event(),
    #         "type": "session.updated",
    #         "session": {
    #             "id": self.id,
    #             "object": "realtime.session",
    #             "model": "default model",
    #             **self.config.model_dump()
    #         }
    #     })
    #
    #
    #     # send all internal ws update config
    #     await ws_stt.send_message(self.config.stt)
    #     await ws_llm.send_message(self.config.llm)
    #     await ws_tts.send_message(self.config.tts)