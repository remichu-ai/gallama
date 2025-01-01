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

    def __init__(self, stt_url: str, llm_url: str, tts_url: str):
        self.stt_url = stt_url
        self.llm_url = llm_url
        self.tts_url = tts_url

        self.ws_stt = WebSocketClient(stt_url)
        self.ws_llm = WebSocketClient(llm_url)
        self.ws_tts = WebSocketClient(tts_url)

        self.initialize_ws = False

        self.audio_buffer = []  # List of (timestamp, audio_data) tuples
        self.buffer_duration = 5.0  # Buffer duration in seconds
        self.sample_rate = 24000  # Assuming 24kHz sample rate
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
                session.vad_processor = VADProcessor(
                    turn_detection_config=session.config.turn_detection,
                    input_sample_rate=session.config.input_sample_rate
                )
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

    async def _update_audio_buffer(self, audio_float: np.ndarray, current_time: float):
        """
        Update the audio buffer with new audio data and remove old chunks

        Args:
            audio_float: numpy array of float32 audio data
            current_time: current timestamp in seconds
        """
        # Add new audio chunk to buffer
        async with self.buffer_lock:
            self.audio_buffer.append((current_time, audio_float))

            # Remove chunks older than buffer_duration
            cutoff_time = current_time - self.buffer_duration
            self.audio_buffer = [(t, a) for t, a in self.audio_buffer if t > cutoff_time]

    def _get_prefix_padding(self, prefix_duration_ms: float, current_time: float) -> np.ndarray:
        """
        Get prefix padding audio from the buffer

        Args:
            prefix_duration_ms: desired prefix duration in milliseconds
            current_time: current timestamp in seconds

        Returns:
            numpy array of prefix audio data or None if no suitable prefix found
        """
        if not self.audio_buffer:
            return None

        prefix_duration_sec = prefix_duration_ms / 1000
        start_time = current_time - prefix_duration_sec

        # Collect relevant audio chunks
        prefix_chunks = []
        for timestamp, audio_data in self.audio_buffer:
            if timestamp >= start_time and timestamp < current_time:
                prefix_chunks.append(audio_data)

        if not prefix_chunks:
            return None

        # Concatenate chunks and trim to desired duration
        prefix_audio = np.concatenate(prefix_chunks)
        desired_samples = int(prefix_duration_sec * self.sample_rate)

        if len(prefix_audio) > desired_samples:
            prefix_audio = prefix_audio[-desired_samples:]

        return prefix_audio

    async def _input_audio_buffer_append(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        try:
            # Decode and convert audio once
            audio_bytes = base64.b64decode(message["audio"])
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Get current time
            current_time = time.time()

            # Update the audio buffer
            self._update_audio_buffer(audio_float, current_time)

            if session.config.turn_detection and session.vad_processor:
                should_buffer, speech_event = session.vad_processor.process_audio_chunk(audio_float)

                if speech_event is not None:
                    if not session.vad_item_id:
                        session.vad_item_id = await session.queues.next_item()

                    if 'start' in speech_event:
                        logger.info(f"VAD speech started: {speech_event}")

                        # First clear the STT buffer before sending any new audio
                        await self.ws_stt.send_pydantic_message(WSInterSTT(
                            type="stt.buffer_clear"
                        ))

                        speech_started_event = InputAudioBufferSpeechStarted(
                            event_id=await session.queues.next_event(),
                            audio_start_ms=int(speech_event['start'] * 1000),
                            item_id=session.vad_item_id
                        )
                        logger.info(f"VAD speech start: {speech_started_event.model_dump()}")
                        await websocket.send_json(speech_started_event.model_dump())

                        # Get prefix padding using internal buffer
                        prefix_padding_ms = session.config.turn_detection.prefix_padding_ms
                        prefix_audio = self._get_prefix_padding(prefix_padding_ms, current_time)

                        # If we have prefix audio, send it to STT
                        if prefix_audio is not None:
                            # Convert back to int16 and then to base64
                            prefix_int16 = (prefix_audio * 32768).astype(np.int16)
                            prefix_bytes = prefix_int16.tobytes()
                            prefix_base64 = base64.b64encode(prefix_bytes).decode()

                            # Send prefix audio to STT
                            await session.queues.append_unprocessed_audio(
                                prefix_base64,
                                ws_stt=self.ws_stt,
                                audio_float=prefix_audio
                            )



                    elif 'end' in speech_event:
                        # Rest of the speech end handling remains the same
                        logger.info(f"VAD speech ended: {speech_event}")
                        speech_stopped_event = InputAudioBufferSpeechStopped(
                            event_id=await session.queues.next_event(),
                            audio_end_ms=int(speech_event['end'] * 1000),
                            item_id=session.vad_item_id,
                            type="input_audio_buffer.speech_stopped"
                        )
                        logger.info(f"speech_stopped_event: {speech_stopped_event.model_dump()}")
                        await websocket.send_json(speech_stopped_event.model_dump())

                        if session.config.turn_detection.create_response:
                            await self._input_audio_buffer_commit(websocket, session, message,
                                                                  item_id=session.vad_item_id)

                            # Process transcription and response
                            transcription_done = await session.queues.wait_for_transcription_done()

                            if transcription_done and session.queues.transcript_buffer:
                                user_audio_item = ConversationItemMessageServer(
                                    id=session.vad_item_id,
                                    type="message",
                                    role="user",
                                    status="completed",
                                    content=[
                                        MessageContentServer(
                                            type=ContentTypeServer.INPUT_AUDIO,
                                            audio=session.queues.audio_buffer,
                                            transcript=session.queues.transcript_buffer
                                        )
                                    ]
                                )

                                await session.queues.update_conversation_item_ordered_dict(
                                    ws_client=websocket,
                                    ws_llm=self.ws_llm,
                                    item=user_audio_item
                                )

                                await websocket.send_json(ConversationItemInputAudioTranscriptionComplete(
                                    event_id=await session.queues.next_event(),
                                    type="conversation.item.input_audio_transcription.completed",
                                    item_id=session.vad_item_id,
                                    content_index=0,
                                    transcript=session.queues.transcript_buffer
                                ).model_dump())

                            response_event = ResponseCreate(id=await session.queues.next_resp())
                            await session.queues.unprocessed.put(response_event)

                        session.vad_item_id = None

                if should_buffer:
                    await session.queues.append_unprocessed_audio(
                        message["audio"],
                        ws_stt=self.ws_stt,
                        audio_float=audio_float
                    )

            else:
                # Process without VAD
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

