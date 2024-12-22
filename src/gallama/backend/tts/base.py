from typing import Any, AsyncIterator, Dict, List
import asyncio
from .text_processor import TextToTextSegment
from .TTSQueueHandler import TTSQueueHandler
from ...data_classes import ModelSpec
from ...logger import logger
from ...routes.ws_tts import TTSEvent
import time


class TTSBase:
    """this is the base interface for all TTS models"""

    def __init__(self, model_spec: ModelSpec):
        self.model_name = model_spec.model_name
        self.model = None

        self.voice_list: List[str] = None
        self.default_voice: Dict = None

        self.voice: Dict = model_spec.voice
        if self.voice:
            self.voice_list = list(self.voice.keys())
            self.default_voice = self.voice[self.voice_list[0]]

        # backend specific arguments
        self.backend_extra_args = model_spec.backend_extra_args

        # each subclass must implement the model loading method
        self.load_model(model_spec)


    def load_model(self, model_spec: ModelSpec):
        raise NotImplementedError(
            "The load_model method must be implemented by the derived class."
        )

    async def text_to_speech(
        self,
        text: str,
        language:str = "auto",
        stream: bool = False,    # non stream return numpy array of whole speech, True will return iterator instead
        speed_factor: float = 1.0,
        queue: asyncio.Queue = None,
        **kwargs: Any
    ):
        # If stream=False: Tuple of (sample_rate, concatenated_audio_data)
        # If stream=True: None (audio chunks are sent to the provided queue)

        raise NotImplementedError(
            "The synthesize method must be implemented by the derived class."
        )

    async def text_stream_to_speech_to_queue(
            self,
            text_queue: asyncio.Queue,
            queue: asyncio.Queue,
            language: str = "auto",
            speed_factor: float = 1.0,
            text_stream_end: asyncio.Event = None,
            audio_stream_end: asyncio.Event = None,
            **kwargs: Any
    ) -> None:

        pipeline = TextToTextSegment(
            quick_start=True,
            initial_segment_size=1,  # Start with single sentences
            max_segment_size=4,  # Maximum of 4 sentences per segment
            segment_size_increase_interval=1  # Increase segment size every 3 segments
        )

        # Create a set to track generations specific to this streaming session
        session_generations = set()

        try:
            processing_task = asyncio.create_task(
                pipeline.process_text_stream_async(input_queue=text_queue)
            )

            last_log_time = time.time()

            while True:
                try:
                    segment = await pipeline.get_next_segment(timeout=0.1, raise_timeout=True)

                    if segment:
                        segment = segment.strip()
                        await self.text_to_speech(
                            queue=queue,
                            text=segment,
                            language=language,
                            stream=True,
                            speed_factor=speed_factor,
                            session_tracker=session_generations,
                        )

                except asyncio.TimeoutError:
                    # current_time = time.time()
                    # # Only log if 3 seconds have passed since the last log
                    # if current_time - last_log_time >= 3.0:
                    #     queue_size = pipeline.processing_queue.qsize()
                    #     logger.info(f"-------------------Timeout reached, queue is {pipeline.processing_queue}")
                    #     logger.info(f"-------------------Queue size: {queue_size}-------------------")
                    #     logger.info(f"-------------------Text buffer is: {pipeline.text_buffer}-------------------")
                    #     logger.info(f"-------------------Text Processing task is: {processing_task.done()}-------------------")
                    #     # Update the last log time
                    #     last_log_time = current_time


                    if not session_generations and pipeline.processing_queue.qsize()==0 and pipeline.text_buffer=="" and processing_task.done():
                        # logger.info("------------------ break frog here------------------------")
                        # queue_size = pipeline.processing_queue.qsize()
                        # logger.info(f"-------------------Timeout reached, queue is {pipeline.processing_queue}")
                        # logger.info(f"-------------------Queue size: {queue_size}-------------------")
                        # logger.info(f"-------------------Text buffer is: {pipeline.text_buffer}-------------------")
                        # logger.info(f"-------------------Text Processing task is: {processing_task.done()}-------------------")
                        break
                    else:
                        continue
        except Exception as e:
            logger.error(f"Error in text_stream_to_speech_to_queue: {str(e)}", exc_info=True)
            await queue.put(Exception(f"Text-to-speech error: {str(e)}"))
            if hasattr(self, 'model') and hasattr(self.model, 'stop'):
                self.model.stop()
            raise
        finally:
            logger.info("Cleaning up pipeline")
            await queue.put(TTSEvent(type="text_end"))
            await pipeline.stop_processing()
            await pipeline.reset()
            await pipeline.clear_queue()