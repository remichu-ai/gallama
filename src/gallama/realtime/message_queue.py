import asyncio
from collections import OrderedDict
from typing import TypeVar, Optional, Union, Tuple, List
import numpy as np
from starlette.websockets import WebSocket
import base64
from gallama.data_classes.realtime_data_classes import ConversationItemServer, ConversationItem, \
    ConversationItemCreated, parse_conversation_item, ConversationItemMessageServer, MessageContentServer
from gallama.data_classes.internal_ws import (
    WSInterSTT
)
from gallama.realtime.websocket_client import WebSocketClient
from gallama.logger.logger import logger


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


        # all non audio events go here
        self.unprocessed = asyncio.Queue()

        # ordered dict of conversation item
        self.conversation_item_od: OrderedDict[str, T] = OrderedDict()

        self.response_queue = asyncio.Queue()
        self.audio_to_client = asyncio.Queue()
        self.response_counter = 0
        self.event_counter = 0
        self.item_counter = 0

        self.lock_conversation_item = asyncio.Lock()
        self.lock_response_counter = asyncio.Lock()
        self.lock_event_counter = asyncio.Lock()
        self.lock_item_counter = asyncio.Lock()

        self.lock_audio_buffer = asyncio.Lock()     # ensure that audio is sync with ws_stt
        self.lock_transcript_buffer = asyncio.Lock()     # ensure that audio is sync with ws_stt
        self.lock_audio_commited = asyncio.Lock()     # ensure that audio is sync with ws_stt

        self.latest_item: Optional[ConversationItem] = None




        # self.uncommitted_audio_data: Optional[bytes] = None
        # self.uncommitted_text: Optional[str] = None

    async def next_event(self) -> str:
        """ return the next counter for event"""
        async with self.lock_event_counter:
            self.event_counter += 1
            return f"event_{self.event_counter}"

    async def next_resp(self) -> str:
        """ return the next counter for response"""
        async with self.lock_response_counter:
            self.response_counter += 1
            return f"resp_{self.response_counter}"

    async def next_item(self, return_current=False) -> Union[str,Tuple[str,Union[str, None]]]:
        """ return the next counter for response"""
        async with self.lock_item_counter:
            self.item_counter += 1
            if return_current:
                if not self.conversation_item_od:
                    return f"item_{self.item_counter}", None
                else:
                    return f"item_{self.item_counter}", next(reversed(self.conversation_item_od.keys()))
            else:
                return f"item_{self.item_counter}"

    async def get_previous_item_id(self, message_id: str) -> Optional[str]:
        """Get the ID of the message that comes before the given message_id"""
        async with self.lock_conversation_item:
            previous_id = None
            for id in self.conversation_item_od.keys():
                if id == message_id:
                    return previous_id
                previous_id = id
            return None  # Message ID not found

    async def append_unprocessed_audio(self, base64audio: str, ws_stt: WebSocketClient):
        """
        Appends base64 encoded audio data to the audio buffer after decoding.

        Args:
            base64audio (str): Base64 encoded audio string
            ws_stt (WebSocketClient): the websocket client for ws_stt

        Returns:
            None
        """
        try:
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


    async def commit_unprocessed_audio(self, ws_stt: WebSocketClient):
        """
        Signal to ws_stt that the audio buffer has been committed.
        Returns:
            None
        """
        try:
            # send the ws_stt the commit signal
            async with self.lock_audio_buffer:
                await ws_stt.send_pydantic_message(WSInterSTT(type="stt.sound_done"))

            async with self.lock_audio_commited:
                self.audio_commited = True
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
                # clear the audio_buffer
                self.audio_buffer = np.array([], dtype=np.float32)
        except Exception as e:
            # Log error and optionally raise it depending on your error handling needs
            logger.error(f"Error processing audio data: {str(e)}")
            raise

    async def append_transcription(self, transcription_chunk):
        async with self.lock_transcript_buffer:
            self.transcript_buffer += transcription_chunk

    async def mark_transcription_done(self):
        logger.info(f"transcription done")
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
            return True
        except asyncio.TimeoutError:
            return False
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

            # send to ws_llm to sync with this item
            stripped_item = item.strip_audio()
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
                        "item": stripped_item.model_dump()
                    }).model_dump()

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
