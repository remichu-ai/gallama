from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
import asyncio
import uuid
from typing import Dict, List, Optional, AsyncGenerator
import json
from datetime import datetime
import websockets
from dataclasses import dataclass
from queue import Queue
import logging
import torch
import numpy as np
from typing import List, Tuple
import wave
import io
from .realtime_data_classes import (
    SessionConfig,
    ConversationItem,
    ConversationItemCreate,
    ConversationItemTruncate,
    ConversationItemDelete,
    MessageContent,
    ContentType,
    ResponseCreate,
    ResponseCancel,
    SessionCreated,
    SessionUpdate
)
from silero_vad import load_silero_vad, VADIterator


# Create router
router = APIRouter(prefix="", tags=["realtime"])

# helper function
def generate_event_id_uuid():
    """Generate event ID using UUID"""
    return f"event_{uuid.uuid4().hex[:20]}"  # Using first 20 chars of UUID



class VADProcessor:
    def __init__(self):
        # Load Silero VAD model using the package's loader
        self.model = load_silero_vad()
        self.model.eval()

        # Initialize VAD iterator
        self.vad_iterator = VADIterator(
            model=self.model,
            threshold=0.5,  # sensitivity threshold, higher = more sensitive
            sampling_rate=16000,  # expected sample rate
            min_speech_duration_ms=250,  # minimum speech chunk duration
            max_speech_duration_s=float('inf'),  # maximum speech chunk duration
            min_silence_duration_ms=500  # minimum silence duration to split speech
        )

        # Buffer for accumulating audio
        self.audio_buffer = []
        self.silence_duration = 0
        self.SILENCE_THRESHOLD = 1000  # ms of silence to consider end of speech

    def process_audio_chunk(self, audio_chunk: bytes) -> Tuple[bool, bool]:
        """
        Process an audio chunk and determine if it contains speech and if it's end of speech
        Returns: (contains_speech, is_end_of_speech)
        """
        # Convert bytes to numpy array (assuming 16-bit PCM audio)
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)

        # Convert to float32 and normalize
        audio_float = audio_np.astype(np.float32) / 32768.0

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_float)

        # Process with VAD
        speech_prob = self.vad_iterator(audio_tensor, return_seconds=True)

        # Update silence duration
        if speech_prob < 0.5:  # No speech detected
            self.silence_duration += len(audio_np) / 16000 * 1000  # Convert to ms
        else:
            self.silence_duration = 0

        # Check if we've hit silence threshold
        is_end_of_speech = self.silence_duration >= self.SILENCE_THRESHOLD

        if is_end_of_speech:
            self.silence_duration = 0  # Reset silence counter

        return speech_prob >= 0.5, is_end_of_speech


@dataclass
class ServiceConnections:
    stt: websockets.WebSocketClientProtocol
    llm: websockets.WebSocketClientProtocol
    tts: websockets.WebSocketClientProtocol


class WebSocketManager:
    def __init__(self):
        self.STT_URL = "ws://stt-service/ws"
        self.LLM_URL = "ws://llm-service/ws"
        self.TTS_URL = "ws://tts-service/ws"

    async def connect_to_service(self, url: str) -> websockets.WebSocketClientProtocol:
        try:
            return await websockets.connect(url)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to service at {url}: {str(e)}")


class AudioBuffer:
    def __init__(self):
        self.buffer = bytearray()
        self.max_size = 15 * 1024 * 1024  # 15 MiB

    def append(self, audio_data: bytes) -> bool:
        if len(self.buffer) + len(audio_data) > self.max_size:
            return False
        self.buffer.extend(audio_data)
        return True

    def clear(self):
        self.buffer = bytearray()

    def get_buffer(self) -> bytes:
        return bytes(self.buffer)

    def is_empty(self) -> bool:
        return len(self.buffer) == 0


class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[ConversationItem]] = {}
        self.active_responses: Dict[str, bool] = {}

    def get_or_create_conversation(self, session_id: str) -> List[ConversationItem]:
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return self.conversations[session_id]

    def add_item(self, session_id: str, item: ConversationItem, previous_item_id: Optional[str] = None) -> Tuple[bool, str]:
        conversation = self.get_or_create_conversation(session_id)

        # Generate new item ID if not provided
        if not item.id:
            item.id = f"msg_{len(conversation)}_{int(datetime.now().timestamp())}"

        # If previous_item_id is provided, insert after that item
        if previous_item_id:
            for i, conv_item in enumerate(conversation):
                if conv_item.id == previous_item_id:
                    conversation.insert(i + 1, item)
                    return True, item.id
            # If previous_item_id not found, return error
            return False, "Previous item ID not found"
        else:
            # If no previous_item_id, append to end
            conversation.append(item)
            return True, item.id

    def delete_item(self, session_id: str, item_id: str) -> bool:
        conversation = self.get_or_create_conversation(session_id)
        for i, item in enumerate(conversation):
            if item.id == item_id:
                conversation.pop(i)
                return True
        return False

# variable:
    # message queue of unprocessed
    # message queue of global history
    # latest_item (the latest item being processing), once processed will go into history

    # response queue
    # audio_to_client queue, any audio put here will be send to client

# function:
    # get snapshot:
        # get a history + latest item snapshot -> ChatQueryML

    # ask llm no audio -> get snapshot -> ask for 1 token
    # ask llm with audio -> get snapshot -> put result into a queue that tts will process

# concurrent task:
    # process unprocessed queue:
        # look out for item and process one at a time
        # if text, straight away add to global history
        # if audio, process, transcribe and stream the result latest_item. Once transcribing finished, move it to history and set latest_item to none

    #  LLM processing:
        # monitor response queue:
            # if a response.create event receive, wait until queue of unprocessed is empty
            # then get snapshot and send LLM

    # audio to client:
        # monitor queue of audio to client and send to client



class MessageQueue:
    def __init__(self):
        self.pending_messages: Dict[str, asyncio.Event] = {}  # Track messages being processed
        self.message_order: List[str] = []  # Maintain order of messages
        self.processed_messages: Dict[str, ConversationItem] = {}  # Store processed messages
        self.transcription_queue: asyncio.Queue = asyncio.Queue()  # Queue for streaming transcriptions
        self.lock = asyncio.Lock()

    async def add_message(self, item_id: str):
        """Add a new message to the queue"""
        async with self.lock:
            self.pending_messages[item_id] = asyncio.Event()
            self.message_order.append(item_id)

    async def add_transcription(self, item_id: str, text: str, is_final: bool):
        """Add transcribed text to the queue"""
        await self.transcription_queue.put({
            "item_id": item_id,
            "text": text,
            "is_final": is_final
        })
        if is_final:
            await self.mark_complete(item_id, text)

    async def mark_complete(self, item_id: str, final_text: str):
        """Mark a message as complete with its final transcription"""
        async with self.lock:
            if item_id in self.processed_messages:
                item = self.processed_messages[item_id]
                if item.content is None:
                    item.content = []
                # Update or add the transcribed text
                for content in item.content:
                    if content.type == ContentType.INPUT_TEXT:
                        content.text = final_text
                        break
                else:
                    item.content.append(MessageContent(
                        type=ContentType.INPUT_TEXT,
                        text=final_text
                    ))
            if item_id in self.pending_messages:
                self.pending_messages[item_id].set()

    async def get_next_transcription(self):
        """Get the next transcription from the queue"""
        return await self.transcription_queue.get()


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, SessionConfig] = {}
        self.conversation_manager = ConversationManager()
        self.message_queues: Dict[str, MessageQueue] = {}
        self.llm_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, AudioBuffer] = {}
        self.ws_manager = WebSocketManager()
        self.active_streams: Dict[str, List[asyncio.Task]] = {}
        self.vad_processors: Dict[str, VADProcessor] = {}

    async def handle_audio_message(self, websocket: WebSocket, session_id: str,
                                   item: ConversationItem, audio_data: bytes):
        """Handle streaming audio input with real-time transcription"""
        message_queue = self.message_queues[session_id]

        try:
            # Start STT streaming
            async for transcript, is_final in self.stream_stt(session_id, audio_data):
                # Add transcription to queue
                await message_queue.add_transcription(item.id, transcript, is_final)

                # Update the conversation item with interim transcription
                if not is_final:
                    interim_item = item.copy()
                    if interim_item.content is None:
                        interim_item.content = []
                    interim_item.content.append(MessageContent(
                        type=ContentType.INPUT_TEXT,
                        text=transcript
                    ))
                    interim_item.status = ItemStatus.INCOMPLETE

                    # Send interim update to client
                    await websocket.send_json({
                        "type": "conversation.item.updated",
                        "item": interim_item.dict()
                    })

        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")
            raise

    async def process_llm_stream(self, session_id: str):
        """Process transcribed text through LLM while maintaining order"""
        message_queue = self.message_queues[session_id]

        try:
            while True:
                # Get next transcription from queue
                transcription = await message_queue.get_next_transcription()
                item_id = transcription["item_id"]
                text = transcription["text"]
                is_final = transcription["is_final"]

                # Wait for previous messages if this is a final transcription
                if is_final:
                    await message_queue.wait_for_previous_messages(item_id)

                # Process with LLM
                async with self.ws_manager.get_llm_connection() as llm_conn:
                    await llm_conn.send(json.dumps({
                        "text": text,
                        "is_final": is_final,
                        "message_id": item_id
                    }))

                    # Stream LLM response
                    async for response in self.stream_llm_response(llm_conn):
                        yield response

        except asyncio.CancelledError:
            logging.info(f"LLM processing cancelled for session {session_id}")
        except Exception as e:
            logging.error(f"Error in LLM processing: {str(e)}")
            raise

    async def handle_websocket(self, websocket: WebSocket, session_id: str):
        """Main WebSocket handler with streaming support"""
        await websocket.accept()

        # Initialize message queue for this session
        if session_id not in self.message_queues:
            self.message_queues[session_id] = MessageQueue()

        message_queue = self.message_queues[session_id]

        # Start LLM processing task
        self.llm_tasks[session_id] = asyncio.create_task(
            self.process_llm_responses(websocket, session_id)
        )

        try:
            while True:
                message = await websocket.receive()

                if isinstance(message, dict) and "text" in message:
                    data = json.loads(message["text"])

                    if data["type"] == "conversation.item.create":
                        create_event = ConversationItemCreate.model_validate(data)
                        item = create_event.item

                        # Add to message queue
                        await message_queue.add_message(item.id)

                        # If it's a text message, process immediately
                        if item.content and any(c.type == ContentType.INPUT_TEXT for c in item.content):
                            await message_queue.mark_complete(item.id, item.content[0].text)

                        # Send creation confirmation
                        await websocket.send_json({
                            "type": "conversation.item.created",
                            "item": item.dict()
                        })

                elif isinstance(message, dict) and "bytes" in message:
                    # Handle streaming audio data
                    if session_id not in self.audio_buffers:
                        self.audio_buffers[session_id] = AudioBuffer()

                    buffer = self.audio_buffers[session_id]
                    if buffer.append(message["bytes"]):
                        await self.handle_audio_message(
                            websocket,
                            session_id,
                            current_audio_item,  # You'll need to track the current audio item
                            buffer.get_buffer()
                        )
                        buffer.clear()

        except WebSocketDisconnect:
            # Cancel LLM processing
            if session_id in self.llm_tasks:
                self.llm_tasks[session_id].cancel()
            # Clean up other resources
            if session_id in self.message_queues:
                del self.message_queues[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]

    async def process_llm_responses(self, websocket: WebSocket, session_id: str):
        """Process LLM responses and handle TTS if needed"""
        try:
            async for llm_response in self.process_llm_stream(session_id):
                # Send text response
                await websocket.send_json({
                    "type": "response.message",
                    "text": llm_response
                })

                # If TTS is enabled, convert to speech
                if "audio" in self.sessions[session_id].modalities:
                    audio_response = await self.process_tts(session_id, llm_response)
                    await websocket.send_bytes(audio_response)

        except Exception as e:
            logging.error(f"Error processing LLM responses: {str(e)}")
            raise

    async def handle_text_message(
        self,
        websocket: WebSocket,
        session_id: str,
        item: ConversationItem
    ) -> ConversationItem:

        """Handle text input directly"""
        # Text messages can be processed immediately
        return item


    async def update_session(
        self,
        websocket: WebSocket,
        session_id: str,
        update: SessionConfig
    ) -> SessionConfig:

        current_session = self.sessions.get(
            session_id,
            SessionConfig(modalities=["text"])
        )

        # Update session config
        updated_data = current_session.dict()
        update_data = update.dict(exclude_unset=True)
        updated_data.update(update_data)

        updated_session = SessionConfig(**updated_data)
        self.sessions[session_id] = updated_session

        # Create new VAD processor with updated config if needed
        if session_id in self.vad_processors:
            self.vad_processors[session_id] = VADProcessor(updated_session)

        await websocket.send_json({
            "type": "session.updated",
            "session": updated_session.dict()
        })
        return updated_session


    async def stream_stt(
        self,
        conn: ServiceConnections,
        audio_queue: asyncio.Queue,
        session_id: str
    ) -> AsyncGenerator[Tuple[str, bool], None]:

        """Stream audio to STT service and yield transcribed text"""
        if session_id not in self.vad_processors:
            self.vad_processors[session_id] = VADProcessor(
                self.sessions.get(session_id, SessionConfig(modalities=["text"]))
            )

        vad = self.vad_processors[session_id]
        buffer = []

        try:
            while True:
                audio_chunk = await audio_queue.get()

                if audio_chunk is None:  # Signal to stop streaming
                    break

                # Process with VAD
                contains_speech, is_end_of_speech = vad.process_audio_chunk(audio_chunk)

                if contains_speech:
                    # Send audio chunk to STT service only if speech is detected
                    await conn.stt.send(audio_chunk)

                    # Get transcription
                    transcript = await conn.stt.recv()

                    if transcript:
                        buffer.append(transcript)
                        # Yield intermediate results
                        yield buffer[-1], False

                if is_end_of_speech and buffer:
                    # Yield complete utterance when speech ends
                    complete_text = " ".join(buffer)
                    buffer = []  # Reset buffer
                    yield complete_text, True

                audio_queue.task_done()

        except Exception as e:
            logging.error(f"Error in STT streaming: {str(e)}")
            raise
        finally:
            # Clean up VAD processor if session ends
            if session_id in self.vad_processors:
                del self.vad_processors[session_id]

    async def stream_llm(self, conn: ServiceConnections, text_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """Stream text to LLM service and yield responses"""
        try:
            current_utterance = []

            while True:
                # Get transcribed text and end-of-speech flag from queue
                data = await text_queue.get()
                if data is None:  # Signal to stop streaming
                    break

                text, is_end_of_speech = data
                current_utterance.append(text)

                if is_end_of_speech:
                    # Send complete utterance to LLM with end-of-speech signal
                    complete_text = " ".join(current_utterance)
                    await conn.llm.send(json.dumps({
                        "text": complete_text,
                        "end_of_speech": True
                    }))

                    # Stream responses until we get a completion signal
                    while True:
                        response = await conn.llm.recv()
                        response_data = json.loads(response)

                        if response_data.get("type") == "response.done":
                            break

                        if "text" in response_data:
                            yield response_data["text"]

                    # Reset for next utterance
                    current_utterance = []

                text_queue.task_done()

        except Exception as e:
            logging.error(f"Error in LLM streaming: {str(e)}")
            raise

    async def stream_tts(self, conn: ServiceConnections, text_queue: asyncio.Queue) -> AsyncGenerator[bytes, None]:
        """Stream text to TTS service and yield audio chunks"""
        try:
            while True:
                text = await text_queue.get()

                if text is None:  # Signal to stop streaming
                    break

                # Send text to TTS service
                await conn.tts.send(json.dumps({"text": text}))

                # Get audio response (might come in chunks)
                while True:
                    audio = await conn.tts.recv()
                    if not audio:  # Empty response signals end of audio for this text
                        break
                    yield audio

                text_queue.task_done()

        except Exception as e:
            logging.error(f"Error in TTS streaming: {str(e)}")
            raise

    async def process_audio_stream(self, websocket: WebSocket, session_id: str):
        """Coordinate streaming between STT, LLM, and TTS services"""
        audio_queue = asyncio.Queue()
        stt_text_queue = asyncio.Queue()
        llm_text_queue = asyncio.Queue()

        try:
            # Connect to all services
            connections = ServiceConnections(
                stt=await self.ws_manager.connect_to_service(self.ws_manager.STT_URL),
                llm=await self.ws_manager.connect_to_service(self.ws_manager.LLM_URL),
                tts=await self.ws_manager.connect_to_service(self.ws_manager.TTS_URL)
            )

            # Create tasks for each streaming service
            stt_task = asyncio.create_task(
                self.handle_stt_stream(connections, audio_queue, stt_text_queue, session_id)
            )
            llm_task = asyncio.create_task(
                self.handle_llm_stream(connections, stt_text_queue, llm_text_queue)
            )
            tts_task = asyncio.create_task(
                self.handle_tts_stream(connections, llm_text_queue, websocket)
            )

            # Store tasks for potential cancellation
            self.active_streams[session_id] = [stt_task, llm_task, tts_task]

            # Wait for all tasks to complete
            await asyncio.gather(stt_task, llm_task, tts_task)

        except Exception as e:
            logging.error(f"Error in audio stream processing: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "message": f"Processing error: {str(e)}"
            })
        finally:
            # Clean up
            for queue in [audio_queue, stt_text_queue, llm_text_queue]:
                await queue.put(None)

            if session_id in self.active_streams:
                del self.active_streams[session_id]

            # Close service connections
            for conn in [connections.stt, connections.llm, connections.tts]:
                await conn.close()

    async def handle_stt_stream(self, conn: ServiceConnections, audio_queue: asyncio.Queue,
                                text_queue: asyncio.Queue, session_id: str):
        """Handle streaming from STT service with VAD"""
        async for text, is_end_of_speech in self.stream_stt(conn, audio_queue, session_id):
            await text_queue.put((text, is_end_of_speech))

    async def handle_llm_stream(self, conn: ServiceConnections, input_queue: asyncio.Queue,
                                output_queue: asyncio.Queue):
        """Handle streaming from LLM service"""
        async for response in self.stream_llm(conn, input_queue):
            await output_queue.put(response)

    async def handle_tts_stream(self, conn: ServiceConnections, text_queue: asyncio.Queue,
                                websocket: WebSocket):
        """Handle streaming from TTS service"""
        async for audio in self.stream_tts(conn, text_queue):
            await websocket.send_bytes(audio)






# Session manager instance
session_manager = SessionManager()


@router.websocket("/{path:path}")
async def websocket_endpoint(
    websocket: WebSocket,
    path: str,
    model: Optional[str] = None
):
    # TODO this function is only for openAI protocol
    # Accept with OpenAI protocol if requested
    protocols = websocket.headers.get("sec-websocket-protocol", "").split(", ")
    if "openai-beta.realtime-v1" in protocols:
        await websocket.accept(subprotocol="openai-beta.realtime-v1")
    else:
        await websocket.accept()

    session_id = str(id(websocket))

    # for new connection:
    # Create default session configuration
    default_session = SessionConfig(modalities=["text"])
    session_manager.sessions[session_id] = default_session

    # Send session.created event
    await websocket.send_json({
        "event_id": generate_event_id_uuid(),
        "type": "session.created",
        "session": default_session.model_dump()
    })

    try:
        while True:
            data = await websocket.receive_text()

            try:
                event = json.loads(data)
                event_type = event.get("type")

                if event_type == "conversation.item.create":
                    create_event = ConversationItemCreate.model_validate(event)
                    success, item_id = session_manager.conversation_manager.add_item(
                        session_id,
                        create_event.item,
                        create_event.previous_item_id
                    )
                    if success:
                        await websocket.send_json({
                            "event_id": generate_event_id_uuid(),
                            "type": "conversation.item.created",
                            "item": {**create_event.item.dict(), "id": item_id}
                        })
                    else:
                        await websocket.send_json({
                            "event_id": generate_event_id_uuid(),
                            "type": "error",
                            "error": {
                                "code": "invalid_request",
                                "message": f"Failed to create item: {item_id}"
                            }
                        })

                elif event_type == "conversation.item.truncate":
                    truncate_event = ConversationItemTruncate.parse_obj(event)
                    success = session_manager.conversation_manager.truncate_item(
                        session_id,
                        truncate_event.item_id,
                        truncate_event.content_index,
                        truncate_event.audio_end_ms
                    )
                    if success:
                        await websocket.send_json({
                            "type": "conversation.item.truncated",
                            "item_id": truncate_event.item_id
                        })
                    else:
                        await websocket.send_json({
                            "event_id": generate_event_id_uuid(),
                            "type": "error",
                            "error": {
                                "code": "invalid_request",
                                "message": "Failed to truncate item"
                            }
                        })

                elif event_type == "conversation.item.delete":
                    delete_event = ConversationItemDelete.model_validate(event)
                    success = session_manager.conversation_manager.delete_item(
                        session_id,
                        delete_event.item_id
                    )
                    if success:
                        await websocket.send_json({
                            "event_id": generate_event_id_uuid(),
                            "type": "conversation.item.deleted",
                            "item_id": delete_event.item_id
                        })
                    else:
                        await websocket.send_json({
                            "event_id": generate_event_id_uuid(),
                            "type": "error",
                            "error": {
                                "code": "invalid_request",
                                "message": "Item not found"
                            }
                        })

                elif event_type == "response.create":
                    response_event = ResponseCreate.model_validate(event)
                    await session_manager.create_response(
                        websocket,
                        session_id,
                        response_event.response
                    )

                elif event_type == "response.cancel":
                    cancel_event = ResponseCancel.model_validate(event)
                    success = await session_manager.cancel_response(
                        websocket,
                        session_id
                    )
                    if not success:
                        await websocket.send_json({
                            "event_id": generate_event_id_uuid(),
                            "type": "error",
                            "error": {
                                "code": "invalid_request",
                                "message": "No active response to cancel"
                            }
                        })

                elif event_type == "session.update":
                    update = SessionUpdate.model_validate(event)
                    await session_manager.update_session(
                        websocket,
                        session_id,
                        update.session
                    )
                elif event_type == "response.create":
                    # Handle response.create event
                    # You'll implement this in the next part
                    pass
                else:
                    await websocket.send_json({
                        "event_id": generate_event_id_uuid(),
                        "type": "error",
                        "error": {
                            "code": "invalid_request",
                            "message": f"Unsupported event type: {event_type}"
                        }
                    })

            except ValueError as e:
                await websocket.send_json({
                    "event_id": generate_event_id_uuid(),
                    "type": "error",
                    "error": {
                        "code": "invalid_request",
                        "message": str(e)
                    }
                })

    except WebSocketDisconnect:
        # Clean up session, conversation, and audio buffer
        if session_id in session_manager.sessions:
            del session_manager.sessions[session_id]
        if session_id in session_manager.conversation_manager.conversations:
            del session_manager.conversation_manager.conversations[session_id]
        if session_id in session_manager.audio_buffers:
            del session_manager.audio_buffers[session_id]
