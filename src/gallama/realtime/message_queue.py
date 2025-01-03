import asyncio
from collections import OrderedDict
from typing import TypeVar, Optional, Union, Tuple, List
import numpy as np
from starlette.websockets import WebSocket
import base64
import soundfile as sf
import aiohttp
import io
from gallama.data_classes.realtime_client_proto import (
    ConversationItem,
    ConversationItemDelete,
    ConversationItemTruncate,
)
from gallama.data_classes.realtime_server_proto import ConversationItemDeleted, ConversationItemTruncated, \
    MessageContentServer, ConversationItemMessageServer, ConversationItemServer, parse_conversation_item, \
    ConversationItemCreated
from gallama.data_classes.internal_ws import (
    WSInterSTT
)
from gallama.realtime.websocket_client import WebSocketClient
from difflib import SequenceMatcher
import uuid
from ..dependencies_server import get_server_logger

logger = get_server_logger()


T = TypeVar("T", bound=ConversationItemServer)


class MessageQueues:
    """Manages different message queues for the websocket system"""
    def __init__(self):
        # current audio will store into this array.
        # once audio commited, it will move to part of conversation_itm_od
        self.audio_buffer = np.array([], dtype=np.float32)

        # transcription of audio until before commit will be in self.transcription_buffer
        self.transcript_buffer = ""
        self.transcription_complete = False
        self.audio_commited = False
        self.lock_transcript_complete = asyncio.Lock()

        self.sample_rate = 24000    # assuming fixed at this rate
        self.verbose = False  # Control debug logging

        # all non audio events go here
        self.unprocessed = asyncio.Queue()

        # ordered dict of conversation item
        self.conversation_item_od: OrderedDict[str, T] = OrderedDict()

        self.response_queue = asyncio.Queue()
        self.audio_to_client = asyncio.Queue()
        self.response_counter = ""
        self.event_counter = ""
        self.item_counter = ""

        self.lock_conversation_item = asyncio.Lock()
        self.lock_response_counter = asyncio.Lock()
        self.lock_event_counter = asyncio.Lock()
        self.lock_item_counter = asyncio.Lock()

        self.lock_audio_buffer = asyncio.Lock()     # ensure that audio is sync with ws_stt
        self.lock_transcript_buffer = asyncio.Lock()     # ensure that audio is sync with ws_stt
        self.lock_audio_commited = asyncio.Lock()     # ensure that audio is sync with ws_stt

        self.vad_item_id = None
        self.speech_start = None
        self.speech_end = None
        # self.latest_item: Optional[ConversationItem] = None

        # self.uncommitted_audio_data: Optional[bytes] = None
        # self.uncommitted_text: Optional[str] = None

    async def next_event(self) -> str:
        """ return the next counter for event"""
        async with self.lock_event_counter:
            self.event_counter = str(uuid.uuid4().hex)
            return f"event_{self.event_counter}"

    async def next_resp(self) -> str:
        """ return the next counter for response"""
        async with self.lock_response_counter:
            self.response_counter = str(uuid.uuid4().hex)
            return f"resp_{self.response_counter}"

    async def next_item(self, return_current=False) -> Union[str,Tuple[str,Union[str, None]]]:
        """ return the next counter for response"""
        async with self.lock_item_counter:
            self.item_counter = str(uuid.uuid4().hex)
            if return_current:
                if not self.conversation_item_od:
                    return f"item_{self.item_counter}", None
                else:
                    return f"item_{self.item_counter}", next(reversed(self.conversation_item_od.keys()))
            else:
                return f"item_{self.item_counter}"

    async def current_item_id(self) -> str | None:
        """ return the current item id"""
        if not self.conversation_item_od:
            return None
        else:
            return next(reversed(self.conversation_item_od.keys()))     # id of the latest message


    async def get_previous_item_id(self, message_id: str) -> Optional[str]:
        """Get the ID of the message that comes before the given message_id. This only apply to committed messages"""
        async with self.lock_conversation_item:
            previous_id = None
            for id in self.conversation_item_od.keys():
                if id == message_id:
                    return previous_id
                previous_id = id
            return None  # Message ID not found

    async def append_unprocessed_audio(self, base64audio: str, ws_stt: WebSocketClient, audio_float: Optional[np.ndarray] = None):
        """
        Appends base64 encoded audio data to the audio buffer after decoding.
        float32 normalized audio data in numpy array
        Args:
            base64audio (str): Base64 encoded audio string
            ws_stt (WebSocketClient): the websocket client for ws_stt

        Returns:
            None
        """
        try:
            if not audio_float.any():
                # Decode base64 string to bytes
                audio_bytes = base64.b64decode(base64audio)

                # Convert bytes to numpy array of int16 values
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

                # Convert int16 to float32 (normalize to [-1.0, 1.0] range)
                audio_float = audio_data.astype(np.float32) / 32768.0

            # Append to existing buffer
            async with self.lock_audio_buffer:
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])

            # send the audio to ws_stt
            await ws_stt.send_pydantic_message(
                WSInterSTT(
                    type="stt.add_sound_chunk",
                    sound=base64audio,
                )
            )
        except Exception as e:
            # Log error and optionally raise it depending on your error handling needs
            logger.error(f"Error processing audio data: {str(e)}")
            raise


    async def commit_unprocessed_audio(self, ws_stt: WebSocketClient, item_id: str = None, skip_sound_done:bool = False):
        """
        Signal to ws_stt that the audio buffer has been committed.

        If a specific item id to be used, it can be passed in

        Returns:
            None
        """
        try:
            if not skip_sound_done:
                # send the ws_stt the commit signal
                async with self.lock_audio_buffer:
                    await ws_stt.send_pydantic_message(WSInterSTT(type="stt.sound_done"))

            async with self.lock_audio_commited:
                self.audio_commited = True

            self.vad_item_id = item_id
        except Exception as e:
            # Log error and optionally raise it depending on your error handling needs
            logger.error(f"Error processing audio data: {str(e)}")
            raise

    async def clear_unprocessed_audio(self, ws_stt: WebSocketClient):
        """
        Signal to ws_stt that the audio buffer has been committed.
        After stt clear the buffer on its side, it will send the acknowledgement.
        There will be another job that based on the acknowledgement to clear the transcription buffer
        Returns:
            None
        """
        try:
            # send the ws_stt the clear signal
            async with self.lock_audio_buffer:
                await ws_stt.send_pydantic_message(WSInterSTT(type="stt.clear_buffer"))
                # reset audio
                await self.reset_audio()
        except Exception as e:
            # Log error and optionally raise it depending on your error handling needs
            logger.error(f"Error processing audio data: {str(e)}")
            raise

    async def append_transcription(self, transcription_chunk):
        async with self.lock_transcript_buffer:
            self.transcript_buffer += transcription_chunk


    async def mark_transcription_done(self):
        logger.info(f"in MessageQueue: transcription done")
        async with self.lock_transcript_complete:
            self.transcription_complete = True

    async def clear_transcription(self):
        async with self.lock_transcript_buffer:
            self.transcript_buffer = ""
        async with self.lock_transcript_complete:
            self.transcription_complete = False

    async def clear_audio_buffer(self):
        async with self.lock_audio_buffer:
            self.audio_buffer = np.array([], dtype=np.float32)

    async def reset_audio(self):
        try:
            clear_audio_task = self.clear_audio_buffer()
            clear_transcription_buffer = self.clear_transcription()
            self.vad_item_id = None

            await asyncio.gather(clear_audio_task, clear_transcription_buffer)

            async with self.lock_audio_commited:
                self.audio_commited = False

            return True
        except Exception as e:
            logger.error(f"Error reset_audio data: {str(e)}")
            return False

    async def audio_exist(self):
        """ return True or False if there is any audio at all in the conversation"""
        async with self.lock_audio_buffer:
            if self.audio_buffer.any():
                return True
            else:
                return False



    async def reset_after_response(self):
        # for now only audio need to reset
        _ = await self.reset_audio()
        logger.info(f"reset after response completed")


    async def wait_for_transcription_done(self):
        """wait for audio to commit with 10 second timeout

        Returns:
            bool: True if transcription completed, False if timeout occurred
        """
        try:
            async def _wait():
                while not self.transcription_complete:
                    await asyncio.sleep(0.05)
                return True

            await asyncio.wait_for(_wait(), timeout=10.0)
            logger.info(f"transcription done")
            return True, self.vad_item_id
        except asyncio.TimeoutError:
            return False, None
        except Exception as e:
            raise Exception(f"Error in wait_for_transcription_done: {str(e)}")

    async def update_conversation_item_ordered_dict_client(
        self,
        item: ConversationItemServer        # item to create
    ):
        """
        Add an item to the conversation item list.
        If it is a new item, return a ConversationItemCreated object for onward return to front end
        """
        async with self.lock_conversation_item:
            # update to history
            if isinstance(item, ConversationItem):
                self.conversation_item_od[item.id] = item


    async def update_conversation_item_ordered_dict(
        self,
        ws_client: WebSocket,               # web socket for client
        ws_llm: WebSocketClient,            # web socket for llm
        item: ConversationItemServer        # item to create
    ):
        """
        Add an item to the conversation item list.
        If it is a new item, return a ConversationItemCreated object for onward return to front end
        """
        try:
            new_event_id = await self.next_event()
            previous_item_id = await self.get_previous_item_id(item.id)
            # if no previous item, meaning this is new item
            if not previous_item_id:
                previous_item_id = await self.current_item_id()

            item_to_send = item

            # send to ws_llm to sync with this item
            stripped_item = item.strip_audio(mode="remove")
            await ws_llm.send_pydantic_message(stripped_item)

            # send client update about new item created
            async with self.lock_conversation_item:
                if not item.id:
                    item.id = await self.next_item()

                if item.id not in self.conversation_item_od.keys():
                    item_to_send = ConversationItemCreated(**{
                        "event_id": new_event_id,
                        "type": "conversation.item.created",
                        "previous_item_id": previous_item_id,
                        "item": stripped_item.model_dump(exclude_none=True)
                    }).model_dump(exclude_none=True)

                # update to history with server type item
                item_server = parse_conversation_item(item.model_dump())
                self.conversation_item_od[item.id] = item_server

                # send to client
                await ws_client.send_json(item_to_send)
        except Exception as e:
            logger.error(f"Error in update_conversation_item_ordered_dict: {str(e)}")
            raise

    async def update_conversation_item_with_assistant_response(
        self,
        response_id: str,
        _type: str,
        text: str,
        transcript: str,
        audio: np.ndarray,
        ws_llm: WebSocketClient,  # web socket for llm
    ):
        """
        Add an item to the conversation item list.
        If it is a new item, return a ConversationItemCreated object for onward return to front end
        """
        try:
            # update value in place to prevent moving its position
            self.conversation_item_od[response_id].content = [
                MessageContentServer(
                    type=_type,
                    text=text,
                    transcript=transcript,
                    audio=audio,
                )
            ]

            # send to ws_llm to sync with this item
            stripped_item = self.conversation_item_od[response_id].strip_audio()
            await ws_llm.send_pydantic_message(stripped_item)

        except Exception as e:
            logger.error(f"Error in update_conversation_item_with_assistant_response: {str(e)}")
            raise


    async def delete_conversation_item(
            self,
            ws_client: WebSocket,
            ws_llm: WebSocketClient,
            item_id: str,
    ):
        """
        Delete an item from the conversation item list.
        Update previous_item_id for the next item.
        Send this item to ws_llm as well to sync the state.
        Notify the client about the deletion via a ConversationItemDeleted event.

        Args:
            ws_client (WebSocket): The websocket connection to the client
            ws_llm (WebSocketClient): The websocket connection to the LLM service
            item_id (str): The ID of the item to delete

        Raises:
            Exception: If there's an error during the deletion process
        """
        try:
            new_event_id = await self.next_event()

            async with self.lock_conversation_item:
                # Check if item exists before trying to delete
                if item_id not in self.conversation_item_od:
                    logger.warning(f"Attempted to delete non-existent item: {item_id}")
                    return

                # Remove the item from the ordered dictionary
                del self.conversation_item_od[item_id]

                # Create deletion event for the client
                delete_event = ConversationItemDeleted(
                    event_id=new_event_id,
                    type="conversation.item.deleted",
                    item_id=item_id
                )

                # Send deletion event to the client
                await ws_client.send_json(delete_event.model_dump())

                # Send deletion event to the LLM to sync state
                await ws_llm.send_pydantic_message(delete_event)

        except Exception as e:
            logger.error(f"Error in delete_conversation_item: {str(e)}")
            raise


    def get_last_conversation_item(self):
        return next(reversed(self.conversation_item_od.items()))

    @staticmethod
    def find_truncated_match(partial_text, full_transcript):
        # Store original words for output
        full_words_original = full_transcript.split()

        # Convert to lowercase only for matching
        partial_text_lower = partial_text.lower().strip()
        full_transcript_lower = full_transcript.lower().strip()

        partial_words = partial_text_lower.split()
        partial_length = len(partial_words)

        full_words_lower = full_transcript_lower.split()

        best_ratio = 0
        best_end_pos = partial_length

        for end_pos in range(partial_length - 2, partial_length + 3):
            if end_pos <= 0 or end_pos >= len(full_words_lower):
                continue

            current_chunk = ' '.join(full_words_lower[:end_pos])
            sequence_ratio = SequenceMatcher(None, partial_text_lower, current_chunk).ratio()

            if sequence_ratio > best_ratio:
                best_ratio = sequence_ratio
                best_end_pos = end_pos

        return {
            'matched_text': ' '.join(full_words_original[:best_end_pos]),  # Use original text
            'next_words': ' '.join(full_words_original[best_end_pos:best_end_pos + 3]),  # Use original text
            'confidence': best_ratio
        }

    async def _debug_audio_diagnostics(self, audio_data: np.ndarray, truncated_audio: np.ndarray, end_sample: int):
        """Log detailed audio diagnostics and save debug files when verbose is True."""
        if not self.verbose:
            return

        logger.info("=== AUDIO DIAGNOSTIC INFO ===")
        logger.info(f"Original audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        logger.info(f"Sample rate: {self.sample_rate}")
        logger.info(f"Input duration: {len(audio_data) / self.sample_rate:.2f} seconds")
        logger.info(f"Expected samples for 1 second: {self.sample_rate}")
        logger.info(f"Audio min value: {np.min(audio_data)}, max value: {np.max(audio_data)}")
        logger.info(f"Truncating at sample {end_sample}")
        logger.info(f"Calculated duration after truncation: {end_sample / self.sample_rate:.2f} seconds")
        logger.info(f"Truncated audio shape: {truncated_audio.shape}, dtype: {truncated_audio.dtype}")
        logger.info(f"Truncated audio min value: {np.min(truncated_audio)}, max value: {np.max(truncated_audio)}")

        if self.verbose:
            await self._save_debug_audio_files(truncated_audio)

    async def _save_debug_audio_files(self, truncated_audio: np.ndarray):
        """Save audio files for debugging when verbose is True."""
        try:
            # Save as 16-bit PCM WAV at 16kHz
            debug_path = "/home/remichu/work/ML/gallama/experiment/debug_audio.wav"
            sf.write(debug_path, truncated_audio, self.sample_rate, subtype='PCM_16')
            logger.info(f"Successfully saved debug audio file to: {debug_path}")

            # Save another copy at 44.1kHz for comparison
            debug_path_441 = "/home/remichu/work/ML/gallama/experiment/debug_audio_44100.wav"
            import samplerate

            # Resample to 44.1kHz
            audio_441 = samplerate.resample(truncated_audio, 44100 / self.sample_rate, 'sinc_best')
            sf.write(debug_path_441, audio_441, 44100, subtype='PCM_16')
            logger.info(f"Saved 44.1kHz version to: {debug_path_441}")

        except Exception as e:
            logger.error(f"Error saving debug audio files: {str(e)}")

    async def _prepare_audio_buffer(self, audio_data: np.ndarray, audio_end_ms: int) -> tuple[np.ndarray, io.BytesIO]:
        """Prepare truncated audio data and buffer for transcription."""
        # Convert milliseconds to samples
        end_sample = int((audio_end_ms / 1000) * self.sample_rate)

        # Truncate the audio data
        truncated_audio = audio_data[:end_sample]

        # Ensure audio data is in float32 format
        if truncated_audio.dtype != np.float32:
            if truncated_audio.dtype == np.int16:
                truncated_audio = truncated_audio.astype(np.float32) / 32768.0
            elif truncated_audio.dtype == np.int32:
                truncated_audio = truncated_audio.astype(np.float32) / 2147483648.0
            else:
                truncated_audio = truncated_audio.astype(np.float32)

        # Log diagnostics if verbose
        if self.verbose:
            await self._debug_audio_diagnostics(audio_data, truncated_audio, end_sample)

        # Prepare buffer for API call
        buffer = io.BytesIO()
        sf.write(buffer, truncated_audio, self.sample_rate, format='wav', subtype='PCM_16')
        buffer.seek(0)

        return truncated_audio, buffer

    async def _get_transcription(self, audio_buffer: io.BytesIO) -> str:
        """Get transcription for the truncated audio."""
        form_data = aiohttp.FormData()
        form_data.add_field('file', audio_buffer, filename='audio.wav', content_type='audio/wav')
        form_data.add_field('model', 'whisper-1')
        form_data.add_field('response_format', 'json')
        form_data.add_field('temperature', str(0.0))
        form_data.add_field('timestamp_granularities', 'segment')

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    'http://localhost:8001/v1/audio/transcriptions',
                    data=form_data
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    transcription = response_data.get('text')
                    if transcription is None:
                        raise Exception("Transcription response missing 'text' field")
                    return transcription
                else:
                    error_detail = await response.text()
                    raise Exception(f"Transcription failed with status {response.status}: {error_detail}")

    async def truncate_conversation_item(
            self,
            ws_client: WebSocket,
            ws_llm: WebSocketClient,
            event: ConversationItemTruncate,
            user_interrupt_token: str = ""
    ):
        """Truncate an audio item from the conversation item list."""
        try:
            async with self.lock_conversation_item:
                # item truncation should hold a lock until it complete

                new_event_id = await self.next_event()
                item_id = event.item_id
                audio_end_ms = event.audio_end_ms

                # Send acknowledgement to client
                async def send_client_acknowledgement():
                    return await ws_client.send_json(ConversationItemTruncated(
                        event_id=new_event_id,
                        type="conversation.item.truncated",
                        item_id=item_id,
                        audio_end_ms=audio_end_ms
                    ).model_dump())

                async def _update_internal_state():
                    # Get original audio data
                    audio_data = self.conversation_item_od[item_id].content[0].audio

                    # Prepare truncated audio and buffer
                    truncated_audio, audio_buffer = await self._prepare_audio_buffer(audio_data, audio_end_ms)

                    # Get new transcription
                    new_transcription = await self._get_transcription(audio_buffer)

                    # Fuzzy match the transcription
                    new_transcription_matched = self.find_truncated_match(
                        partial_text=new_transcription,
                        full_transcript=self.conversation_item_od[item_id].content[0].transcript,
                    )
                    new_transcription = new_transcription_matched["matched_text"]

                    logger.info(f"Original transcription: {self.conversation_item_od[item_id].content[0].transcript}")
                    logger.info(f"After truncated transcription: {new_transcription}")

                    # Update conversation item
                    current_item = self.conversation_item_od[item_id].content[0]
                    current_item.audio = truncated_audio
                    current_item.text = new_transcription + user_interrupt_token
                    current_item.transcript = new_transcription + user_interrupt_token

                    # Sync with LLM
                    stripped_item = self.conversation_item_od[item_id].strip_audio()
                    await ws_llm.send_pydantic_message(stripped_item)

                # Execute both tasks concurrently
                await asyncio.gather(send_client_acknowledgement(), _update_internal_state())

        except Exception as e:
            logger.error(f"Error in truncate_conversation_item: {str(e)}")
            raise