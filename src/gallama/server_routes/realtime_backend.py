from fastapi import WebSocket
import websockets
import asyncio
import uuid
from pydantic import BaseModel
from typing import List, Optional, Dict, TypeVar, Generic, Union, Tuple, Callable, Awaitable
from collections import OrderedDict
import json
import io
import soundfile as sf
import numpy as np
from gallama.logger.logger import logger
from websockets.exceptions import WebSocketException
from websockets.protocol import State
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
from ..routes.ws_tts import WSMessageTTS

from gallama.data_classes.realtime_data_classes import *

from gallama.data_classes import (
    ChatCompletionResponse,
    UsageResponse
)


class WebSocketClient:
    def __init__(
        self,
        uri: str,
        reconnect_interval: float = 5.0,
        max_retries: int = 5,
        auto_reconnect: bool = True
    ):
        self.uri = uri
        self.connection: Optional[websockets.WebSocketClientProtocol] = None
        self.reconnect_interval = reconnect_interval
        self.max_retries = max_retries
        self.auto_reconnect = auto_reconnect
        self.retry_count = 0
        self.is_connecting = False

        self.websocket_options = {
            "ping_interval": 20,  # Enable periodic ping to detect connection issues
            "ping_timeout": 20,
            "close_timeout": 300,
            "max_size": None,
            "open_timeout": 60
        }

    async def _wait_for_connection(self, timeout: float = 30.0):
        """Wait for an ongoing connection attempt to complete."""
        start_time = asyncio.get_event_loop().time()
        while self.is_connecting:
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning("Timeout waiting for connection attempt")
                return False
            await asyncio.sleep(0.1)

        # After waiting, check if we actually have a valid connection
        return self.connection is not None and self.connection.state == State.OPEN

    async def ensure_connection(self) -> bool:
        """Ensures that the connection is active and attempts to reconnect if necessary."""
        try:
            # First check if we have an active connection
            # test = self.connection.state
            if self.connection and self.connection.state == State.OPEN:
                try:
                    # Verify connection is truly alive with ping
                    pong_waiter = await self.connection.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    return True
                except Exception as e:
                    logger.debug(f"Ping check failed: {str(e)}")
                    # Connection is not truly alive
                    self.connection = None

            # If another task is already trying to connect, wait for it
            if self.is_connecting:
                logger.debug("Another task is connecting, waiting...")
                return await self._wait_for_connection()

            # Set connecting flag before attempting connection
            self.is_connecting = True
            try:
                return await self.connect()
            finally:
                self.is_connecting = False

        except Exception as e:
            logger.debug(f"Connection check failed: {str(e)}")
            return False

    async def connect(self) -> bool:
        """Establishes a WebSocket connection with retry logic."""
        if self.is_connecting:
            return await self._wait_for_connection()

        self.is_connecting = True
        try:
            while self.retry_count < self.max_retries:
                try:
                    self.connection = await websockets.connect(self.uri, **self.websocket_options)
                    self.retry_count = 0
                    logger.info("Successfully connected to WebSocket server")
                    return True
                except WebSocketException as e:
                    self.retry_count += 1
                    logger.warning(f"Connection attempt {self.retry_count} failed: {str(e)}")

                    if self.retry_count >= self.max_retries:
                        logger.error("Max retry attempts reached")
                        return False

                    await asyncio.sleep(self.reconnect_interval)
            return False
        finally:
            self.is_connecting = False

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Sends a dictionary message, handling reconnection if necessary."""
        try:
            if not await self.ensure_connection():
                return False

            await self.connection.send(json.dumps(message))
            return True
        except WebSocketException as e:
            logger.error(f"Error sending message: {str(e)}")
            if self.auto_reconnect:
                self.connection = None
                return await self.send_message(message)
            return False

    async def send_pydantic_message(self, message: BaseModel) -> bool:
        """Sends a Pydantic model message, handling reconnection if necessary."""
        try:
            if not await self.ensure_connection():
                return False

            await self.connection.send(message.model_dump_json())
            return True
        except WebSocketException as e:
            logger.error(f"Error sending Pydantic message: {str(e)}")
            if self.auto_reconnect:
                self.connection = None
                return await self.send_pydantic_message(message)
            return False

    async def receive_message(self) -> Optional[str]:
        """Receives a message from the WebSocket connection."""
        try:
            if not await self.ensure_connection():
                return None

            return await self.connection.recv()
        except WebSocketException as e:
            logger.error(f"Error receiving message: {str(e)}")
            if self.auto_reconnect:
                self.connection = None
                return await self.receive_message()
            return None

    async def close(self):
        """Closes the WebSocket connection."""
        if self.connection:
            try:
                await self.connection.close()
            except WebSocketException as e:
                logger.error(f"Error closing connection: {str(e)}")
            finally:
                self.connection = None
                self.retry_count = 0

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()



class WebSocketSession:
    """Encapsulates all session-related state and operations"""

    def __init__(self, session_id: str, config: SessionConfig):
        self.id = session_id
        self.config = config
        self.voice_used = False
        self.queues = MessageQueues()
        self.tasks: List[asyncio.Task] = []

    def mark_voice_used(self):
        self.voice_used = True

    async def cleanup(self):
        for task in self.tasks:
            task.cancel()
        # Additional cleanup as needed


class SessionManager:
    """Handles session lifecycle"""

    def __init__(self):
        self.sessions: Dict[str, WebSocketSession] = {}

    def create_session(self, session_id: str, config: Optional[SessionConfig] = None) -> WebSocketSession:
        if config is None:
            config = SessionConfig(modalities=["text"])
        session = WebSocketSession(session_id, config)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[WebSocketSession]:
        return self.sessions.get(session_id)

    async def delete_session(self, session_id: str):
        if session_id in self.sessions:
            await self.sessions[session_id].cleanup()
            del self.sessions[session_id]



class Response:
    """ this class managing the current response to client"""
    def __init__(
        self,
        session: WebSocketSession,
        ws_client: WebSocket,
        ws_llm: WebSocketClient,
        ws_tts: WebSocketClient,
        response_id: str,
        event_id_generator: Callable[[], Awaitable[str]]
    ):
        self.session = session
        self.ws_client = ws_client
        self.ws_llm = ws_llm
        self.ws_tts = ws_tts

        self.tts_queue = asyncio.Queue()

        self.current_session_config: SessionConfig = None
        self.usage: UsageResponseRealTime = UsageResponseRealTime()

        self.response_id = response_id
        self.event_id_generator = event_id_generator

        self.item_id: str = None
        self.previous_item_id: str = None

        self.audio: str = ""
        self.audio_done: bool = False

        self.text: str = ""
        self.text_done: bool = False

        self.transcription: str = ""
        self.transcription_done: bool = False

        # some tracker
        self.content_part_done_status: bool = False
        self.response_done_status: bool = False

    async def content_part_done(self, generation_type: Literal["text", "audio", "transcription"]):
        logger.info(f"content part done is called with type: {generation_type}")
        logger.info(f"transcription: {self.transcription_done}")
        logger.info(f"audio: {self.audio_done}")
        logger.info(f"text: {self.text_done}")

        done_check: bool = False
        content_part: Dict = {}

        # for text type, just the text finish is enough
        allowed_modalities = self.session.config.modalities
        if generation_type == "text" and "text" in allowed_modalities and "audio" not in allowed_modalities:
            if self.text_done:
                done_check = True
                content_part = {
                    "type": "text",
                    "text": self.text
                }
            else:
                raise Exception("text generation not finished but content part is marked as done")

        # for audio type, we need both transcript and audio to finish
        if generation_type in ["audio", "transcription"] and "audio" in allowed_modalities:
            if generation_type=="transcription" and not self.transcription_done:
                raise Exception("Transcription generation not finished but content part is marked as done")

            if generation_type=="audio" and not self.audio_done:
                raise Exception("Audio generation not finished but content part is marked as done")

            if self.transcription_done and self.audio_done:
                done_check = True
                content_part = {
                    "type": "audio",
                    #"audio": self.audio,       # this is not included, despite what said in documentation
                    "transcript": self.transcription
                }

        # if done_check achieved meaning text finished or audio & transcription finished
        if done_check:
            await self.ws_client.send_json(ResponseContentPartDoneEvent(**{
                "event_id": await self.event_id_generator(),
                "type": "response.content_part.done",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "part": content_part
            }).model_dump())

            # mark status of content_part as done
            self.content_part_done_status = True
            logger.info(f"Content part done")


    async def response_done(self):
        try:
            if not self.response_done_status and self.content_part_done_status:   # content part must be done first
                _output = None
                allowed_modalities = self.session.config.modalities

                if "audio" in allowed_modalities:
                    _output = parse_conversation_item({
                        "id": self.item_id,
                        "object": "realtime.item",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "audio",
                                "audio": self.audio,
                                "transcript": self.transcription
                            }
                        ]
                    })
                elif "text" in allowed_modalities and "audio" not in allowed_modalities:
                    # send response done
                    _output = parse_conversation_item({
                        "id": self.item_id,
                        "object": "realtime.item",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": self.text
                            }
                        ]
                    })
                else:
                    raise Exception("Unknown modalities type")

                # send user status
                if _output:
                    _response = ServerResponse(**{
                        "id": self.response_id,
                        "object": "realtime.response",
                        "status": "completed",
                        "usage": self.usage.model_dump(),
                        "output": [_output.model_dump()]        # it must be an array
                    })

                    await self.ws_client.send_json(ResponseDone(**{
                        "event_id": await self.event_id_generator(),
                        "type": "response.done",
                        "response_id": self.response_id,
                        "item_id": self.item_id,
                        "response": _response.model_dump()
                    }).model_dump())
                else:
                    raise Exception("Invalid Response Type")

                # update response_done tracker
                self.response_done_status = True
            elif self.response_done_status:
                raise Exception("Response already sent")
            elif not self.content_part_done_status:
                raise Exception("Content part done must be sent before response done")

        except Exception as e:
            logger.error(f"{e}")


    async def response_initialize(self):
        """ send empty event to client to signal response is created """
        await self.ws_client.send_json(ResponseCreated(**{
            "event_id": await self.event_id_generator(),
            "type": "response.created",
            "response": {
                "id": self.response_id,
                "status": "in_progress",
            }
        }).model_dump())

        # also create an empty conversation item
        self.item_id, self.previous_item_id = await self.session.queues.next_item(return_current=True)

        initial_item_created_object = ConversationItemCreated(**{
            "event_id": await self.event_id_generator(),
            "type": "conversation.item.created",
            "previous_item_id": self.previous_item_id,
            "item": {
                "id": self.item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": []
            }
        })
        await self.session.queues.update_conversation_item_ordered_dict(
            ws_client=self.ws_client,
            ws_llm=self.ws_llm,
            item=initial_item_created_object.item
        )

        # send initial output_item created
        await self.ws_client.send_json(ResponseOutput_ItemAdded(**{
            "event_id": await self.event_id_generator(),
            "response_id": self.response_id,
            "output_index": 0,
            "item": initial_item_created_object.item
        }).model_dump())


    async def update_delta(self, mode: Literal["text", "transcription", "audio"], chunk: str):
        """ send text chunk to client as well as update internal state """

        if mode=="text":
            _type = "response.text.delta"
            self.text += chunk
        elif mode=="transcription":
            _type = "response.audio_transcript.delta"
            self.transcription += chunk
        elif mode=="audio":
            _type = "response.audio.delta"
            self.audio += chunk

        # send chunk delta
        await self.ws_client.send_json(ResponseDelta(**{
            "event_id": await self.event_id_generator(),
            "type": _type,
            "response_id": self.response_id,
            "item_id": self.item_id,
            "delta": chunk
        }).model_dump())


    async def update_usage(self, usage: UsageResponse):
        """ send usage to client as well as update internal state """
        self.usage = usage

    async def update_content_done(self, content_type: Literal["text", "audio", "transcription"]):
        """update client that generation is done"""
        # send done status for one type of content

        # text content
        if content_type=="text":
            await self.ws_client.send_json(ResponseTextDone(**{
                "event_id": await self.event_id_generator(),
                "type": "response.text.done",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "text": self.text
            }).model_dump())
            self.text_done = True

        elif content_type=="audio":
            await self.ws_client.send_json(ResponseAudioDone(**{
                "event_id": await self.event_id_generator(),
                "type": "response.audio.done",
                "response_id": self.response_id,
                "item_id": self.item_id,
            }).model_dump())
            self.audio_done = True

        elif content_type=="transcription":
            await self.ws_client.send_json(ResponseTranscriptDone(**{
                "event_id": await self.event_id_generator(),
                "type": "response.audio_transcript.done",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "transcript": self.transcription
            }).model_dump())
            self.transcription_done = True



    async def update_text_or_transcription_task(self, mode: Literal["text", "audio"]):
        try:
            while True:
                # Ensure LLM connection is active
                # if not self.ws_llm.connection:
                #     await self.ws_llm.connect()

                if not await self.ws_llm.ensure_connection():
                    raise Exception("Could not establish LLM connection")

                # Receive message from LLM
                message = await self.ws_llm.connection.recv()

                # Parse the message
                try:
                    llm_response = json.loads(message)
                    if llm_response.get("object") == "chat.completion.chunk":
                        chat_completion_chunk = ChatCompletionResponse(**llm_response)
                        text_chunk = chat_completion_chunk.choices[0].delta.content

                        if mode=="audio":
                            # put generated text into tts queue for converting to audio
                            await self.tts_queue.put(text_chunk)

                        await self.update_delta(
                            mode="transcription" if mode=="audio" else "text",
                            chunk=text_chunk
                        )

                    elif llm_response.get("type") == "usage.update":
                        usage_completion = ChatCompletionResponse(**json.loads(llm_response.get("usage"))).usage
                        usage_data = UsageResponseRealTime(
                            input_tokens=usage_completion.prompt_tokens,
                            output_tokens=usage_completion.completion_tokens,
                            total_tokens=usage_completion.total_tokens,
                        )
                        await self.update_usage(usage_data)
                    elif llm_response.get("type") == "generation.complete":
                        await self.tts_queue.put(None)  # mark that text generation completed for TTS

                        await self.update_content_done(content_type="transcription" if mode=="audio" else "text")
                        await self.content_part_done(generation_type="transcription")
                        await self.response_done()
                    else:
                        pass

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM message: {message}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to parse LLM message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.error("LLM websocket connection closed unexpectedly")
            # Send error message to client
            await self.ws_client.send_json({
                "type": "response.failed",
                "response": {
                    "id": self.response_id,
                    "object": "realtime.response",
                    "status": "failed",
                    "last_error": {
                        "code": "internal_error",
                        "message": "LLM connection terminated unexpectedly"
                    }
                }
            })
        except Exception as e:
            logger.error(f"Unexpected error occurred: {str(e)}")

    async def send_audio_to_client_task(self):
        """Listen for audio chunks from TTS websocket and forward them to the client"""
        first_chunk = True
        try:
            if not await self.ws_tts.ensure_connection():
                raise Exception("Could not establish TTS connection")

            while True:
                try:
                    message = await self.ws_tts.receive_message()
                    if message is None:
                        logger.error("Failed to receive message from TTS service")
                        break

                    if isinstance(message, str):
                        try:
                            tts_response = json.loads(message)
                            if tts_response.get("type") == "tts_complete":
                                logger.info("TTS response complete")
                                # mark audio generation as done
                                self.audio_done = True
                                await self.update_content_done(content_type="audio")
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Received unexpected string message: {message[:100]}...")

                    elif isinstance(message, bytes):
                        try:
                            # Convert to base64 and send
                            audio_base64 = base64.b64encode(message).decode('utf-8')
                            await self.update_delta(
                                mode="audio",
                                chunk=audio_base64,
                            )
                        except Exception as e:
                            logger.error(f"Error processing audio chunk: {str(e)}")
                            continue


                except asyncio.TimeoutError:
                    continue

                except websockets.exceptions.ConnectionClosed:
                    logger.error("TTS websocket connection closed")
                    break
                except Exception as e:
                    raise Exception(f"Unexpected error occurred in send_audio_to_client_task: {str(e)}")


            await self.content_part_done(generation_type="audio")
            await self.response_done()

        except Exception as e:
            logger.error(f"Error in audio_to_client_task: {str(e)}")
            await self.ws_client.send_json({
                "type": "response.failed",
                "response": {
                    "id": self.response_id,
                    "object": "realtime.response",
                    "status": "failed",
                    "last_error": {
                        "code": "internal_error",
                        "message": str(e)
                    }
                }
            })

    async def send_text_to_ws_tts(self):
        """Forward text chunks to TTS service"""
        try:
            # Use the built-in connection management
            if not await self.ws_tts.ensure_connection():
                raise Exception("Could not establish TTS connection")

            while True:
                # Get text chunk from queue
                text = await self.tts_queue.get()

                if text is None:  # End of stream marker
                    await self.ws_tts.send_pydantic_message(WSMessageTTS(**{
                        "type": "text_done"
                    }))
                    break

                # Send text to TTS service
                success = await self.ws_tts.send_pydantic_message(WSMessageTTS(**{
                    "type": "add_text",
                    "text": text
                }))

                if not success:
                    logger.error("Failed to send text to TTS service")
                    break

        except Exception as e:
            logger.error(f"Error in send_text_to_ws_tts: {str(e)}")
        finally:
            # Signal end of text stream
            if self.ws_tts.connection and self.ws_tts.connection.state == State.OPEN:
                await self.ws_tts.send_pydantic_message(WSMessageTTS(**{
                    "type": "text_done"
                }))

    async def response_start(self, response_create_request: ResponseCreate, session_config: SessionConfig):
        """ send empty event to client to signal response is created """

        # send response_create_request to LLM
        user_config = response_create_request.response
        if not user_config:
            user_config = {}
        else:
            user_config = user_config.model_dump()

        config_for_this_generation = SessionConfig(**{
            **session_config.model_dump(),
            **user_config
        })

        # update the response config with this
        self.current_session_config = config_for_this_generation

        response_create_request.response = config_for_this_generation
        await self.ws_llm.send_pydantic_message(response_create_request)

        # send content_part added for text to let front end know that there is text coming
        if "audio" in self.current_session_config.modalities:
            # Send initial audio content part added event
            await self.ws_client.send_json(ResponseContentPartAddedEvent(**{
                "event_id": await self.event_id_generator(),
                "type": "response.content_part.added",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "part": {
                    "type": "audio",
                    "audio": "",
                    "transcript": ""
                }
            }).model_dump())
        else:
            await self.ws_client.send_json(ResponseContentPartAddedEvent(**{
                "event_id": await self.event_id_generator(),
                "type": "response.content_part.added",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "part": {
                   "type": "text",
                    "text": ""
                }
            }).model_dump())

        task_list = []
        if "audio" in self.current_session_config.modalities:
            text_task = self.update_text_or_transcription_task(mode="audio")
            text_to_ws_tts_task = self.send_text_to_ws_tts()
            audio_to_client_task = self.send_audio_to_client_task()
            task_list = [text_to_ws_tts_task, audio_to_client_task, text_task]
        else:
            text_task =self.update_text_or_transcription_task(mode="text")
            task_list = [text_task]

        await asyncio.gather(*task_list)






T = TypeVar("T", bound=ConversationItemServer)

class MessageQueues:
    """Manages different message queues for the websocket system"""
    def __init__(self):
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
        new_event_id = await self.next_event()
        previous_item_id = await self.get_previous_item_id(item.id)

        # async with self.lock_conversation_item:
        await ws_llm.send_pydantic_message(item)

        # send client update about new item created
        async with self.lock_conversation_item:
            if not item.id:
                item.id = await self.next_item()

            if item.id not in self.conversation_item_od.keys():
                item_to_send = ConversationItemCreated(**{
                    "event_id": new_event_id,
                    "type": "conversation.item.created",
                    "previous_item_id": previous_item_id,
                    "item": item.model_dump()
                }).model_dump()

            # update to history with server type item
            item_server = parse_conversation_item(item.model_dump())
            self.conversation_item_od[item.id] = item_server

        # send to client
        await ws_client.send_json(item_to_send)


    def get_snapshot(self) -> List[ConversationItem]:
        snapshot = self.conversation_item_od.copy()
        # if self.latest_item:
        #     snapshot.append(self.latest_item)
        return snapshot




class WebSocketMessageHandler:
    """Handles different message types"""

    def __init__(self, stt_url: str, llm_url: str, tts_url: str):
        self.stt_url = stt_url
        self.llm_url = llm_url
        self.tts_url = tts_url

        self.ws_stt = WebSocketClient(stt_url)
        self.ws_llm = WebSocketClient(llm_url)
        self.ws_tts = WebSocketClient(tts_url)

        self.initialize_ws = False

        self.handlers = {
            "session.update": self.handle_session_update,
            "conversation.item.create": self.handle_conversation_item_create,
            "response.create": self.handle_response_create
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


    async def handle_session_update(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        update_config = SessionConfig(**message["session"])

        # Don't update voice if it's been used
        # This is because voice will stick for the whole session following OpenAI spec
        # can consider change this down the road
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
    async def handle_conversation_item_create(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        # for conversation.item.created, it is a completed item, hence no need to do streaming
        event = ConversationItemCreate(**message)
        await session.queues.unprocessed.put(event.item)

    async def handle_response_create(self, websocket: WebSocket, session: WebSocketSession, message: dict):
        # add a Conversation Item into unprocess queue
        # once the processing reach this model, it will trigger llm -> tts
        item = ResponseCreate(**message)
        await session.queues.unprocessed.put(item)

    async def process_stt(self, ws, audio_data: bytes) -> str:
        await ws.send(audio_data)
        return await ws.recv()

    async def process_llm(self, ws, snapshot: List[ConversationItem]) -> str:
        await ws.send(json.dumps({
            "messages": [item.dict() for item in snapshot]
        }))
        return await ws.recv()

    async def process_tts(self, ws, text: str) -> bytes:
        await ws.send(json.dumps({"text": text}))
        return await ws.recv()





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

        # elif item.content[0].type == ContentType.INPUT_AUDIO:
        # # Process audio through STT
        # self.queues.latest_item = item
        # audio_data = base64.b64decode(item.content[0].audio)
        #
        # async with websockets.connect(self.stt_url) as stt_ws:
        #     transcription = await self.process_stt(stt_ws, audio_data)
        #
        #     # Update item with transcription
        #     item.content.append(MessageContent(
        #         type=ContentType.INPUT_TEXT,
        #         text=transcription
        #     ))
        #
        #     # Move to history and clear latest_item
        #     await self.queues.add_to_history(item)
        #     self.queues.latest_item = None

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
                    await response.response_initialize()
                    await response.response_start(response_create_request=item, session_config=session.config)



                    # response_create_object_json = response_created_object.model_dump_json()
                    # send response created event to user
                    # task1 = websocket.send_json(response_create_object_json)

                    # start generation
                    # task2 = self.message_handler.ws_llm.send_message(response_create_object_json)


                    # send the opening conversation.item.created
                    # task3 = session.queues.update_conversation_item_ordered_dict(
                    #     ws_client=websocket,
                    #     ws_llm=self.message_handler.ws_llm,
                    #     item=initial_item_created_object
                    # )

                    # send the opening response.output_item.added


                    # collecting the received text and send to user
                    # task3 = self.send_llm_result_to_client(websocket, self.message_handler.ws_llm, response_id, session)

                    # await asyncio.gather(task1, task2, task3)



            except Exception as e:
                logger.error(f"Error processing queue item: {str(e)}")
            finally:
                session.queues.unprocessed.task_done()

    # async def send_client_conversation_item_created(self, client_ws: WebSocket):
    #     pass
    #
    # async def send_llm_result_to_client(self, client_ws: WebSocket, llm_ws: WebSocketClient, response_id: str, session: WebSocketSession):
    #     """
    #     Forward LLM websocket responses to the client websocket.
    #
    #     Args:
    #         client_ws (WebSocket): The client's websocket connection
    #         llm_ws (WebSocketClient): The LLM websocket client
    #         response_id (str): The ID of the response being processed
    #     """
    #     item_id = response_id       # TODO not sure how this is used
    #     final_text = ""
    #     first_message = False
    #
    #     try:
    #         while True:
    #             # Ensure LLM connection is active
    #             if not llm_ws.connection:
    #                 await llm_ws.connect()
    #
    #             # send the initial content part added with empty text
    #             if not first_message:
    #                 event_id = await session.queues.next_resp()
    #                 await client_ws.send_json(ResponseContentPartAddedEvent(**{
    #                     "event_id": event_id,
    #                     "type": "response.content_part.added",
    #                     "response_id": response_id,
    #                     "item_id": "msg_" + str(item_id),
    #                     "output_index": 0,
    #                     "content_index": 0,
    #                     "part": {
    #                         "type": "text",
    #                         "text": ""          # initial response is empty text
    #                     }
    #                 }).model_dump())
    #                 first_message = True
    #
    #             # Receive message from LLM
    #             message = await llm_ws.connection.recv()
    #
    #             try:
    #                 # Parse the message
    #                 llm_response = json.loads(message)
    #
    #                 try:
    #                     chat_completion_chunk = ChatCompletionResponse(**llm_response)
    #                     text_chunk = chat_completion_chunk.choices[0].delta.content
    #
    #                     # Forward completion message to client
    #                     # item_id = item_id + 1
    #                     await client_ws.send_json(ResponseDelta(**{
    #                         "event_id": event_id,
    #                         "type": "response.text.delta",
    #                         "response_id": response_id,
    #                         "item_id": "msg_" + str(item_id),
    #                         "output_index": 0,
    #                         "content_index": 0,
    #                         "delta": text_chunk
    #                     }).model_dump())
    #                     final_text = final_text + text_chunk
    #
    #                 except Exception as e:
    #                     # send response Done
    #                     # item_id = item_id + 1
    #                     await client_ws.send_json(ResponseTextDone(**{
    #                         "event_id": event_id,
    #                         "type": "response.text.delta",
    #                         "response_id": response_id,
    #                         "item_id": "msg_" + str(item_id),
    #                         "output_index": 0,
    #                         "content_index": 0,
    #                         "text": final_text
    #                     }).model_dump())
    #
    #
    #
    #                 # # Check if this is a function call
    #                 # elif llm_response.get("type") == "function_call":
    #                 #     await client_ws.send_json({
    #                 #         "type": "conversation.item.created",
    #                 #         "item": {
    #                 #             "id": f"item_{uuid.uuid4().hex[:20]}",
    #                 #             "type": "function_call",
    #                 #             "function_call": llm_response.get("function_call"),
    #                 #             "status": "in_progress"
    #                 #         }
    #                 #     })
    #                 #
    #                 # # Handle streaming text chunks
    #                 # elif llm_response.get("type") == "content_block_delta":
    #                 #     await client_ws.send_json({
    #                 #         "type": "conversation.item.message.created",
    #                 #         "item": {
    #                 #             "id": f"item_{uuid.uuid4().hex[:20]}",
    #                 #             "type": "message",
    #                 #             "role": "assistant",
    #                 #             "content": [{
    #                 #                 "type": "text",
    #                 #                 "text": llm_response.get("delta", "")
    #                 #             }],
    #                 #             "status": "in_progress"
    #                 #         }
    #                 #     })
    #
    #             except json.JSONDecodeError:
    #                 logger.error(f"Failed to parse LLM message: {message}")
    #                 continue
    #
    #     except websockets.exceptions.ConnectionClosed:
    #         logger.error("LLM websocket connection closed unexpectedly")
    #         # Send error message to client
    #         await client_ws.send_json({
    #             "type": "response.failed",
    #             "response": {
    #                 "id": response_id,
    #                 "object": "realtime.response",
    #                 "status": "failed",
    #                 "last_error": {
    #                     "code": "internal_error",
    #                     "message": "LLM connection terminated unexpectedly"
    #                 }
    #             }
    #         })
    #
    #     except Exception as e:
    #         logger.error(f"Error in send_llm_result_to_client: {str(e)}")
    #         # Send error message to client
    #         await client_ws.send_json({
    #             "type": "response.failed",
    #             "response": {
    #                 "id": response_id,
    #                 "object": "realtime.response",
    #                 "status": "failed",
    #                 "last_error": {
    #                     "code": "internal_error",
    #                     "message": str(e)
    #                 }
    #             }
    #         })
    #
    # async def process_llm_queue(self, session: WebSocketSession, websocket: WebSocket):
    #     """Process LLM responses"""
    #     # Implementation similar to your original process_llm_queue
    #     pass
    #
    #
    # async def process_audio_to_client(self, session: WebSocketSession, websocket: WebSocket):
    #     """Process audio responses"""
    #     # Implementation similar to your original process_audio_to_client
    #     pass

# def generate_event_id_uuid():
#     return f"event_{uuid.uuid4().hex[:20]}"