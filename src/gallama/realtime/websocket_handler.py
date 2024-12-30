from ..logger.logger import logger
from gallama.realtime.websocket_client import WebSocketClient
from gallama.realtime.websocket_session import WebSocketSession
from fastapi import WebSocket
from .vad import VADProcessor
import base64
from gallama.data_classes.realtime_data_classes import *
from gallama.data_classes.internal_ws import *


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
            "input_audio_buffer.append": self._input_audio_buffer_append,
            "input_audio_buffer.commit": self._input_audio_buffer_commit,
            "input_audio_buffer.clear": self._input_audio_buffer_clear,

            # converssation
            "conversation.item.create": self._conversation_item_create,
            "conversation.item.truncate": self._conversation_item_truncate,
            "conversation.item.delete": self._conversation_item_delete,

            # response
            "response.create": self._response_create,
            "response.cancel": self._response_cancel
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

        if 'voice' in message["session"] and session.voice_used:
            del message["session"]["voice"]

        session.config = SessionConfig(**{**session.config.model_dump(), **message["session"]})

        # Reset VAD state when configuration changes
        if 'turn_detection' in message["session"]:
            if session.config.turn_detection:
                session.vad_processor = VADProcessor(session.config.turn_detection)
                session.vad_item_id = None
            else:
                session.vad_processor = None
                session.vad_item_id = None

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
        """
        Process audio with or without VAD based on session configuration
        """

        try:
            if session.config.turn_detection and session.vad_processor:
                # Process with VAD
                audio_bytes = base64.b64decode(message["audio"])
                should_buffer, speech_event = session.vad_processor.process_audio_chunk(audio_bytes)
                # logger.info(f"VAD speech event: {speech_event}")
                # logger.info(f"VAD should buffer: {should_buffer}")


                if speech_event is not None:
                    if not session.vad_item_id:
                        session.vad_item_id = await session.queues.next_item()

                    if 'start' in speech_event:
                        logger.info(f"VAD speech started: {speech_event}")
                        # Speech started event
                        speech_started_event = InputAudioBufferSpeechStarted(
                            event_id=await session.queues.next_event(),
                            audio_start_ms=int(speech_event['start'] * 1000),
                            item_id=session.vad_item_id
                        )
                        logger.info(f"VAD speech start: {speech_started_event.model_dump()}")
                        await websocket.send_json(speech_started_event.model_dump())

                        # tell stt to clear any buffer it currently has
                        await self.ws_stt.send_pydantic_message(WSInterSTT(
                            type="stt.buffer_clear"
                        ))

                    elif 'end' in speech_event:
                        # Speech ended event
                        logger.info(f"VAD speech ended: {speech_event}")

                        speech_stopped_event = InputAudioBufferSpeechStopped(
                            event_id=await session.queues.next_event(),
                            audio_end_ms=int(speech_event['end'] * 1000),
                            item_id=session.vad_item_id,
                            type="input_audio_buffer.speech_stopped"
                        )

                        logger.info(f"speech_stopped_event: {speech_stopped_event.model_dump()}")
                        await websocket.send_json(speech_stopped_event.model_dump())

                        # Commit buffer and create response if configured
                        if session.vad_processor.create_response:
                            await self._input_audio_buffer_commit(websocket, session, message, item_id=session.vad_item_id)

                            response_event = ResponseCreate(
                                id=await session.queues.next_resp(),
                            )
                            await session.queues.unprocessed.put(response_event)

                        # reset vad id:
                        session.vad_item_id = None

                if should_buffer:
                    await session.queues.append_unprocessed_audio(message["audio"], ws_stt=self.ws_stt)

            else:
                # Process without VAD
                await session.queues.append_unprocessed_audio(message["audio"], ws_stt=self.ws_stt)

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise

    async def _input_audio_buffer_commit(self, websocket: WebSocket, session: WebSocketSession, message: dict, item_id: str = None):
        await session.queues.commit_unprocessed_audio(ws_stt=self.ws_stt, item_id=item_id)

    async def _input_audio_buffer_clear(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        await session.queues.clear_unprocessed_audio(ws_stt=self.ws_stt)

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

