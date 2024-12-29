import asyncio
import base64
import json
from typing import Callable, Awaitable, Literal, Dict
import numpy as np
import websockets
from starlette.websockets import WebSocket
from websockets.protocol import State
import samplerate
from gallama.logger import logger
from gallama.data_classes import UsageResponse, ChatCompletionResponse, WSInterTTS
from gallama.data_classes.realtime_data_classes import UsageResponseRealTime, ResponseContentPartDoneEvent, \
    parse_conversation_item, ServerResponse, ResponseDone, ResponseCreated, ConversationItemCreated, \
    ResponseOutput_ItemAdded, ResponseDelta, ResponseTextDone, ResponseAudioDone, ResponseTranscriptDone, ResponseCancel,\
    ResponseCreate, SessionConfig, ResponseContentPartAddedEvent
from gallama.data_classes.internal_ws import WSInterCancel
from gallama.realtime.websocket_client import WebSocketClient


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

        # some tracker
        self.content_part_done_status: bool = False
        self.response_done_status: bool = False

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
                    #"audio": self.audio,       # this is not included in data sent to client, despite what said in documentation
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

            # trigger Response done
            await self.response_done()



    async def response_done(self):
        try:
            content_part_status = "completed" if not self.cancel_event.is_set() else "incomplete"
            server_response_status = "completed" if not self.cancel_event.is_set() else "cancelled"

            if not self.response_done_status and self.content_part_done_status:   # content part must be done first
                _output = None
                allowed_modalities = self.session.config.modalities

                if "audio" in allowed_modalities:
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

    async def update_delta(self, mode: Literal["text", "transcription", "audio"], chunk: str,
                           chunk_in_byte: bytes = None):
        """ send text chunk to client as well as update internal state """

        if mode == "text":
            _type = "response.text.delta"
            self.text += chunk
        elif mode == "transcription":
            _type = "response.audio_transcript.delta"
            self.transcription += chunk
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

                # Perform sample rate conversion from 24000 Hz to 16000 Hz
                ratio = 16000 / 24000  # Target rate / Source rate
                chunk_array = samplerate.resample(chunk_array, ratio, 'sinc_best')

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
                logger.error(f"Error processing audio chunk: {str(e)}")
                raise

        # send chunk delta
        await self.ws_client.send_json(ResponseDelta(**{
            "event_id": await self.event_id_generator(),
            "type": _type,
            "response_id": self.response_id,
            "item_id": self.item_id,
            "delta": chunk  # for audio, send to client base64 string
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
        content_done_tracker = False

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
                        text_chunk = chat_completion_chunk.choices[0].delta.content

                        if mode=="audio":
                            # put generated text into tts queue for converting to audio
                            await self.tts_queue.put(text_chunk)

                        await self.update_delta(
                            mode="transcription" if mode=="audio" else "text",
                            chunk=text_chunk
                        )
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

                        await self.update_content_done(content_type="transcription" if mode=="audio" else "text")
                        await self.content_part_done(generation_type="transcription")
                        content_done_tracker = True
                        break   # exit the function

                    else:
                        raise Exception(f"Unknown response type in update_text_or_transcription_task: {llm_response}")

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM message: {message}")
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
                                content_done_tracker = True
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
                    logger.error("TTS websocket connection timed out")
                    continue  # Changed from break to continue to retry
                except websockets.exceptions.ConnectionClosed:
                    logger.error("TTS websocket connection closed")
                    break
                except asyncio.CancelledError:
                    logger.info("Audio generation task cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error occurred in send_audio_to_client_task: {str(e)}")
                    raise

            if not content_done_tracker and not self.cancel_event.is_set():
                logger.info("Audio generation task did not complete due to task cancelling")
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

            # send content_part added for text to let front end know that there is text coming
            if "audio" in current_session_config.modalities:
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