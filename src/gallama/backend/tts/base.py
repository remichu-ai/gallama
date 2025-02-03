from typing import Any, AsyncIterator, Dict, List
import asyncio
from .text_processor import TextToTextSegment
from .TTSQueueHandler import TTSQueueHandler
from ...data_classes import ModelSpec, TTSEvent
from ...logger import logger
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

        # for stopping
        self._stop_event = asyncio.Event()
        self._current_session_tracker: set = None


    def load_model(self, model_spec: ModelSpec):
        raise NotImplementedError(
            "The load_model method must be implemented by the derived class."
        )

    def stop(self):
        """Stop current processing"""
        if self._current_session_tracker is not None:
            self._stop_event.set()
            if self._current_session_tracker:
                self._current_session_tracker.clear()
            self._current_session_tracker = None


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
        stream: bool = True,
        **kwargs: Any
    ) -> None:
        pipeline = TextToTextSegment(
            quick_start=True,
            initial_segment_size=1,  # Start with single sentences
            max_segment_size=4,  # Maximum of 4 sentences per segment
            segment_size_increase_interval=1  # Increase segment size every 3 segments
        )

        # Create a set to track generations specific to this streaming session
        self._current_session_tracker = set()
        self._stop_event.clear()

        try:
            processing_task = asyncio.create_task(
                pipeline.process_text_stream_async(input_queue=text_queue)
            )

            last_log_time = time.time()

            while not self._stop_event.is_set():
                try:
                    segment = await pipeline.get_next_segment(timeout=0.1, raise_timeout=True)

                    if segment and not self._stop_event.is_set():
                        segment = segment.strip()
                        await self.text_to_speech(
                            queue=queue,
                            text=segment,
                            language=language,
                            stream=True,
                            speed_factor=speed_factor,
                            session_tracker=self._current_session_tracker,
                        )

                except asyncio.TimeoutError:
                    # Check if processing is complete
                    if (not self._current_session_tracker and
                            pipeline.processing_queue.qsize() == 0 and
                            pipeline.text_buffer == "" and
                            processing_task.done()):
                        break
                    elif self._stop_event.is_set():
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
            # Cancel the processing task if it's still running
            if not processing_task.done():
                processing_task.cancel()
                try:
                    await processing_task
                except asyncio.CancelledError:
                    pass

            await queue.put(TTSEvent(type="text_end"))
            await pipeline.stop_processing()
            await pipeline.reset()
            await pipeline.clear_queue()

            # Clear session tracker
            self._current_session_tracker = None
            self._stop_event.clear()
