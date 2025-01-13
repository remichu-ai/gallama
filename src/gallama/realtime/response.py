import asyncio
import base64
import json
from typing import Callable, Awaitable, Literal, Dict
import numpy as np
import websockets
from starlette.websockets import WebSocket
from websockets.protocol import State
import samplerate
from gallama.data_classes import UsageResponse, ChatCompletionResponse, WSInterTTS, ChoiceDeltaToolCall
from gallama.data_classes.realtime_client_proto import ResponseCancel,\
    ResponseCreate, SessionConfig
from gallama.data_classes.realtime_server_proto import *
from gallama.data_classes.internal_ws import WSInterCancel
from gallama.realtime.websocket_client import WebSocketClient

from ..dependencies_server import get_server_logger

logger = get_server_logger()

class Response:
    """ this class managing the current response to client"""
    def __init__(
        self,
        session: "WebSocketSession",
        ws_client: WebSocket,
        ws_stt: WebSocketClient,
        ws_llm: WebSocketClient,
        ws_tts: WebSocketClient,
        response_id: str,
        event_id_generator: Callable[[], Awaitable[str]]
    ):
        self.session = session
        self.ws_client = ws_client      # this ws is end user
        self.ws_stt = ws_stt
        self.ws_llm = ws_llm
        self.ws_tts = ws_tts

        self.tts_queue = asyncio.Queue()

        # self.current_session_config: SessionConfig = None
        self.usage: UsageResponseRealTime = UsageResponseRealTime()

        self.response_id = response_id
        self.event_id_generator = event_id_generator

        self.item_id: str = None
        self.previous_item_id: str = None

        self.audio: np.ndarray = np.array([], dtype=np.float32)
        self.audio_done: bool = False

        self.text: str = ""
        self.text_done: bool = False

        self.transcription: str = ""
        self.transcription_done: bool = False

        self.function_calling_arguments = None
        self.function_calling_name = None
        self.function_calling_tool_id = None


        # some tracker
        self.content_part_done_status: bool = False
        self.output_item_done_status: bool = False
        self.response_done_status: bool = False
        self.conversation_item_created_status: bool = False     # track the initial conversation created

        # Add cancellation control
        self.cancel_event = asyncio.Event()


    async def cancel(self, cancel_event: ResponseCancel):
        """Cancel the current response while preserving partial results"""
        try:
            self.cancel_event.set()

            # send event to llm and tts to stop generation
            if self.ws_llm and self.ws_llm.connection:
                await self.ws_llm.send_pydantic_message(WSInterCancel(type="common.cancel"))

            # Cancel TTS generation
            if self.ws_tts and self.ws_tts.connection:
                await self.ws_tts.send_pydantic_message(WSInterCancel(type="common.cancel"))
        except Exception as e:
            logger.error(f"Error during response cancellation: {str(e)}")
            raise

    async def content_part_done(self, generation_type: Literal["text", "audio", "transcription", "function_call"]):
        logger.info(f"content part done is called with type: {generation_type}")
        logger.debug(f"transcription: {self.transcription_done}")
        logger.debug(f"function_call: {self.function_calling_name}")
        logger.debug(f"audio: {self.audio_done}")
        logger.debug(f"text: {self.text_done}")

        done_check: bool = False
        content_part: Dict = {}

        # for text type, just the text finish is enough
        allowed_modalities = self.session.config.modalities
        if generation_type == "function_call":
            # currently no content part done response for function call
            done_check = True
            pass

        elif generation_type == "text" and "text" in allowed_modalities and "audio" not in allowed_modalities:
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
                    #"audio": self.audio,       # this is not included in data sent to client, despite what said in documentation
                    "transcript": self.transcription
                }

        # if done_check achieved meaning text finished or audio & transcription finished
        if done_check:
            # no content part done for function call
            if not generation_type=="function_call":
                await self.ws_client.send_json(ResponseContentPartDoneEvent(**{
                    "event_id": await self.event_id_generator(),
                    "type": "response.content_part.done",
                    "response_id": self.response_id,
                    "item_id": self.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": content_part
                }).model_dump(exclude_unset=True))  # if not, None will be set as audio to front end

            # mark status of content_part as done
            self.content_part_done_status = True
            logger.info(f"Content part done")

            # trigger response output item done
            await self.response_output_item_done(mode=generation_type)

    async def response_output_item_done(self, mode:Literal["text", "audio", "transcription", "function_call"]):

        if mode=="audio" or mode=="transcription":
            _type = "audio"
        elif mode=="text":
            _type = "text"
        elif mode=="function_call":
            _type = "function_call"
        else:
            raise Exception("Unknown mode in response_output_item_done()")

        item = None

        if mode != "function_call":
            item = ConversationItemMessageServer(
                id=self.item_id,
                object="realtime.item",
                type="message",
                status="completed",
                role="assistant",
                content=[MessageContentServer(
                    type=_type,
                    text=self.text if _type=="text" else None,
                    transcript=self.transcription if _type=="audio" else None,
                )]
            )
        else:
            item = ConversationItemFunctionCallServer(
                id=self.item_id,
                object="realtime.item",
                status="completed",
                type="function_call",
                call_id=self.function_calling_tool_id,
                name=self.function_calling_name,
                arguments=self.function_calling_arguments
            )

        # send user response output_item done
        await self.ws_client.send_json(ResponseOutput_ItemDone(**{
            "event_id": await self.event_id_generator(),
            "type": "response.output_item.done",
            "response_id": self.response_id,
            "output_index": 0,
            "item": item.model_dump(exclude_unset=True)
        }).model_dump())

        self.output_item_done_status = True

        # trigger Response done
        await self.response_done()


    async def response_done(self):
        try:
            content_part_status = "completed" if not self.cancel_event.is_set() else "incomplete"
            server_response_status = "completed" if not self.cancel_event.is_set() else "cancelled"

            # content part and response item must be done first
            if not self.response_done_status and self.content_part_done_status and self.output_item_done_status:
                _output = None
                allowed_modalities = self.session.config.modalities

                if self.function_calling_name:  # function calling check first
                    _output = parse_conversation_item({
                        "id": self.item_id,
                        "object": "realtime.item",
                        "type": "function_call",
                        "status": content_part_status,
                        "name": self.function_calling_name,
                        "call_id": self.function_calling_tool_id,
                        "arguments": self.function_calling_arguments,
                    })
                elif "audio" in allowed_modalities:
                    _output = parse_conversation_item({
                        "id": self.item_id,
                        "object": "realtime.item",
                        "type": "message",
                        "status": content_part_status,
                        "role": "assistant",
                        "content": [
                            {
                                "type": "audio",
                                # "audio": self.audio,
                                "audio": None,      # different to documentation, audio is not resent, also this is in bytes, not base64
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
                        "status": content_part_status,
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": self.text
                            }
                        ]
                    })
                else:
                    raise Exception("Unknown modalities type in response_done()")

                # send user status
                if _output:
                    _response = ServerResponse(**{
                        "id": self.response_id,
                        "object": "realtime.response",
                        "status": server_response_status,
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
                raise Exception("Response already sent error in response_done()")
            elif not self.content_part_done_status:
                raise Exception("Content part done must be sent before response done, in response_done()")

        except Exception as e:
            logger.error(f"Error in response_done(): {e}")


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

        # register the item id
        self.item_id, self.previous_item_id = await self.session.queues.next_item(return_current=True)


    async def send_conversation_item_created(
        self,
        mode: Literal["text", "transcription", "audio", "function_call"],
        function_name: str = None,  # for function calling
        call_id: str = None,        # for function calling
    ):
        if mode != "function_call":
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

        elif mode == "function_call":   # function call
            initial_item_created_object = ConversationItemCreated(**{
                "event_id": await self.event_id_generator(),
                "type": "conversation.item.created",
                "previous_item_id": self.previous_item_id,
                "item": {
                    "id": self.item_id,
                    "object": "realtime.item",
                    "type": "function_call",
                    "status": "in_progress",
                    "name": function_name,
                    "call_id": call_id,
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
        else:
            logger.error(f"Unknown mode in send_conversation_item_created() of mode: {mode}")


        # send content_part added for text to let front end know that there is text coming
        if mode == "audio" or mode == "transcription":
            # Send initial audio content part added event
            await self.ws_client.send_json(ResponseContentPartAddedEvent(**{
                "event_id": await self.event_id_generator(),
                "type": "response.content_part.added",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "audio",
                    "audio": "",
                    "transcript": ""
                }
            }).model_dump())
        elif mode == "text":
            await self.ws_client.send_json(ResponseContentPartAddedEvent(**{
                "event_id": await self.event_id_generator(),
                "type": "response.content_part.added",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                   "type": "text",
                    "text": ""
                }
            }).model_dump())


        # mark this as done
        self.conversation_item_created_status = True


    async def update_delta(
        self,
        mode: Literal["text", "transcription", "audio", "function_call"],
        chunk: str | ChoiceDeltaToolCall,
        chunk_in_byte: bytes = None,
    ):
        """ send text chunk to client as well as update internal state """

        # the first delta update will trigger conversation item creation accordingly

        # for function_call mode, this will be sent in the if below where the details of the function call is obtained
        if not self.conversation_item_created_status and mode != "function_call":
            await self.send_conversation_item_created(mode=mode)

        def convert_to_json_string(input_string):
            # Convert the input string to a dictionary
            data_dict = json.loads(input_string)

            # Convert the dictionary to a JSON string with escaped quotation marks
            json_string = json.dumps(data_dict)

            return json_string


        # for overwriting response. Currently use for tool calling
        overwrite_response_dict = {}

        if mode == "text":
            _type = "response.text.delta"
            self.text += chunk
        elif mode == "transcription":
            _type = "response.audio_transcript.delta"
            self.transcription += chunk
        elif mode == "function_call":
            _type = "response.function_call_arguments.delta"
            self.function_calling_arguments = convert_to_json_string(chunk.function.arguments)
            self.function_calling_name = chunk.function.name
            self.function_calling_tool_id = chunk.id
            overwrite_response_dict = {
                "call_id": chunk.id,
                "delta": self.function_calling_arguments,
            }

            # create item
            await self.send_conversation_item_created(
                mode=mode,
                function_name=self.function_calling_name,
                call_id=self.function_calling_tool_id
            )
        elif mode == "audio":
            _type = "response.audio.delta"
            if chunk_in_byte is None:
                raise Exception("chunk_in_byte must be provided for audio mode")

            try:
                # STT generated audio at 24000. However, keep it at 16000 to be consistent with source

                # Convert bytes to numpy array as int16 first (PCM16 format)
                chunk_array = np.frombuffer(chunk_in_byte, dtype=np.int16)

                # Convert to float32 and normalize to [-1, 1]
                chunk_array = chunk_array.astype(np.float32) / 32768.0

                # Ensure chunk_array is 1D
                if chunk_array.ndim == 0:
                    chunk_array = chunk_array.reshape(1)

                # Validate chunk size
                if chunk_array.size == 0:
                    logger.warning("Received empty audio chunk, skipping")
                    return

                # no longer resample here, cause audio in main ws is kept to source
                # Perform sample rate conversion from 24000 Hz to Internal working rate Hz
                # ratio = 16000 / 24000  # Target rate / Source rate
                # chunk_array = samplerate.resample(chunk_array, ratio, 'sinc_best')

                # Concatenate arrays
                if self.audio.size == 0:
                    self.audio = chunk_array
                else:
                    try:
                        # Validate shapes before concatenation
                        if chunk_array.shape[0] == 0:
                            logger.warning("Skipping empty chunk")
                            return

                        self.audio = np.concatenate([self.audio, chunk_array])

                    except Exception as e:
                        logger.error(f"Error concatenating audio chunks: {str(e)}")
                        raise

                # Log some debug information periodically
                if hasattr(self, '_chunk_counter'):
                    self._chunk_counter += 1
                else:
                    self._chunk_counter = 1

                if self._chunk_counter % 10 == 0:
                    logger.debug(f"Audio stats - shape: {self.audio.shape}, "
                                 f"min: {np.min(self.audio)}, max: {np.max(self.audio)}")

            except Exception as e:
                logger.error(f"Error update delta chunk: {str(e)}")
                raise

        # send chunk delta
        response_dict = {
            "event_id": await self.event_id_generator(),
            "type": _type,
            "response_id": self.response_id,
            "item_id": self.item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": chunk  # for audio, send to client base64 string
        }
        if mode=="function_call":
            response_dict.update(overwrite_response_dict)
            response_dict.pop("content_index")

        # logger.info(f"response_dict: {response_dict}")

        await self.ws_client.send_json(ResponseDelta(**response_dict).model_dump(exclude_unset=True))



    async def update_usage(self, usage: UsageResponse):
        """ send usage to client as well as update internal state """
        self.usage = usage

    async def update_content_done(self, content_type: Literal["text", "audio", "transcription", "function_call"]):
        """update client that generation is done"""
        # send done status for one type of content

        # text content
        if content_type=="text":
            await self.ws_client.send_json(ResponseTextDone(**{
                "event_id": await self.event_id_generator(),
                "type": "response.text.done",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "output_index": 0,
                "content_index": 0,
                "text": self.text
            }).model_dump())
            self.text_done = True
        elif content_type=="function_call":
            await self.ws_client.send_json(ResponseFunctionCallArgumentsDone(**{
                "event_id": await self.event_id_generator(),
                "type": "response.function_call_arguments.done",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "output_index": 0,
                "call_id": self.function_calling_tool_id,
                "name": self.function_calling_name,
                "arguments": self.function_calling_arguments,
            }).model_dump())
            self.text_done = True
        elif content_type=="audio":
            await self.ws_client.send_json(ResponseAudioDone(**{
                "event_id": await self.event_id_generator(),
                "type": "response.audio.done",
                "response_id": self.response_id,
                "output_index": 0,
                "content_index": 0,
                "item_id": self.item_id,
            }).model_dump())
            self.audio_done = True

        elif content_type=="transcription":
            await self.ws_client.send_json(ResponseTranscriptDone(**{
                "event_id": await self.event_id_generator(),
                "type": "response.audio_transcript.done",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "output_index": 0,
                "content_index": 0,
                "transcript": self.transcription
            }).model_dump())
            self.transcription_done = True



    async def update_text_or_transcription_task(self, mode: Literal["text", "audio"]):
        content_done_tracker = False
        function_call_tracker = False

        try:
            while not self.cancel_event.is_set():

                if not await self.ws_llm.ensure_connection():
                    raise Exception("Could not establish LLM connection")

                # Receive message from LLM
                message = await asyncio.wait_for(self.ws_llm.connection.recv(), timeout=20)

                # Parse the message
                try:
                    llm_response = json.loads(message)
                    if llm_response.get("object") == "chat.completion.chunk":
                        chat_completion_chunk = ChatCompletionResponse(**llm_response)
                        _delta = chat_completion_chunk.choices[0].delta

                        # finished reason returned with empty delta
                        if _delta.content or (_delta.tool_calls and len(_delta.tool_calls) > 0):
                            text_chunk = _delta.content
                            tool_calls = _delta.tool_calls

                            if text_chunk:      # text answer
                                if mode=="audio":
                                    # put generated text into tts queue for converting to audio
                                    await self.tts_queue.put(text_chunk)

                                await self.update_delta(
                                    mode="transcription" if mode=="audio" else "text",
                                    chunk=text_chunk
                                )
                            elif tool_calls and len(tool_calls) > 0:
                                # for now only return the 1 st function calling even if LLM return multiple

                                if tool_calls[0].function:
                                    await self.update_delta(
                                        mode="function_call",
                                        chunk=tool_calls[0],
                                    )
                                function_call_tracker = True
                            else:
                                logger.error(f"Unknown response type in update_text_or_transcription_task: {llm_response}")

                    elif llm_response.get("type") == "conversation.update.ack":
                        logger.info("Received conversation update acknowledgement from ws_llm")
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
                        if function_call_tracker:
                            _content_type = "function_call"
                        elif mode=="audio":
                            _content_type = "transcription"
                        else:
                            _content_type = "text"

                        await self.update_content_done(content_type=_content_type)
                        await self.content_part_done(generation_type=_content_type)
                        content_done_tracker = True
                        break   # exit the function

                    else:
                        raise Exception(f"Unknown response type in update_text_or_transcription_task: {llm_response}")

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM message to dict: {message}")
                    continue
                except asyncio.TimeoutError:
                    logger.error("LLM connection timed out")
                    continue
                except Exception as e:
                    logger.error(f"Failed to parse LLM message: {e}")
                    raise
        except asyncio.CancelledError:
            logger.info("Text generation task cancelled")
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.error("LLM websocket connection closed unexpectedly")
        except Exception as e:
            logger.error(f"Unexpected error occurred: {str(e)}")


        if not content_done_tracker and not self.cancel_event.is_set():
            logger.error("Text generation task did not complete due to task cancelling")
            await self.tts_queue.put(None)  # mark that text generation completed for TTS

            await self.update_content_done(content_type="transcription" if mode == "audio" else "text")
            await self.content_part_done(generation_type="transcription")
            content_done_tracker = True



    async def send_audio_to_client_task(self):
        """Listen for audio chunks from TTS websocket and forward them to the client"""

        content_done_tracker = False
        try:
            while not self.cancel_event.is_set():
                if not await self.ws_tts.ensure_connection():
                    raise Exception("Could not establish TTS connection")

                try:
                    message = await self.ws_tts.receive_message(timeout=0.05, max_retries=1, disable_warning = True)

                    if isinstance(message, str):
                        try:
                            tts_response = json.loads(message)
                            if tts_response.get("type") == "tts_complete":
                                logger.info("TTS response complete")
                                # mark audio generation as done
                                self.audio_done = True
                                content_done_tracker = True
                                await self.update_content_done(content_type="audio")

                                break   # exit the function
                        except json.JSONDecodeError:
                            logger.warning(f"Received unexpected string message: {message[:100]}...")

                    elif isinstance(message, bytes):
                        try:
                            # Convert to base64 and send
                            audio_base64 = base64.b64encode(message).decode('utf-8')
                            await self.update_delta(
                                mode="audio",
                                chunk=audio_base64,
                                chunk_in_byte=message
                            )
                        except Exception as e:
                            logger.error(f"Error processing audio chunk: {str(e)}")
                            continue


                except asyncio.TimeoutError:
                    # Check if self.function_calling_id is not None
                    if self.function_calling_tool_id is not None:
                        logger.info("Function calling ID is set, considering task completed and no audio needed")
                        # self.audio_done = False
                        # content_done_tracker = False
                        # await self.update_content_done(content_type="audio")
                        break
                    else:
                        continue  # Continue waiting
                except websockets.exceptions.ConnectionClosed:
                    logger.error("TTS websocket connection closed")
                    break
                except asyncio.CancelledError:
                    logger.info("Audio generation task cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error occurred in send_audio_to_client_task: {str(e)}")
                    raise

            # check if function calling as well
            if not self.function_calling_tool_id:
                if not content_done_tracker and not self.cancel_event.is_set():
                    logger.info("Audio generation task `did not complete due to task cancelling")
                    self.audio_done = True
                    await self.update_content_done(content_type="audio")
                    content_done_tracker = True

                await self.content_part_done(generation_type="audio")

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

            while not self.cancel_event.is_set():
                # Get text chunk from queue
                text = await self.tts_queue.get()

                if text is None:  # End of stream marker
                    await self.ws_tts.send_pydantic_message(WSInterTTS(**{
                        "type": "tts.text_done"
                    }))
                    break

                # Send text to TTS service
                success = await self.ws_tts.send_pydantic_message(WSInterTTS(**{
                    "type": "tts.add_text",
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
                await self.ws_tts.send_pydantic_message(WSInterTTS(**{
                    "type": "tts.text_done"
                }))

    async def response_start(self, response_create_request: ResponseCreate, session_config: SessionConfig):
        """ send empty event to client to signal response is created """
        try:
            # send response_create_request to LLM
            request_session_config = response_create_request.response.model_dump() if response_create_request.response else {}

            # find the merged config for this generation
            current_session_config = session_config.merge(request_session_config)
            response_create_request.response = current_session_config


            # at this point, the ws_llm state should already sync with user audio
            await self.ws_llm.send_pydantic_message(response_create_request)


            tasks = []
            if "audio" in current_session_config.modalities:
                text_task = self.update_text_or_transcription_task(mode="audio")
                text_to_ws_tts_task = self.send_text_to_ws_tts()
                audio_to_client_task = self.send_audio_to_client_task()
                tasks = [text_to_ws_tts_task, audio_to_client_task, text_task]
            else:
                text_task = self.update_text_or_transcription_task(mode="text")
                tasks = [text_task]

            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Error in response_start: {str(e)}")
            raise