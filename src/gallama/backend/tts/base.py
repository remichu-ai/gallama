from typing import Any, AsyncIterator, Dict, List
import asyncio
from .text_processor import TextToTextSegment
from .TTSQueueHandler import TTSQueueHandler
from ...data_classes import ModelSpec
from ...logger import logger
from ...routes.ws_tts import TTSEvent

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
        queue: asyncio.Queue,           # audio chunk will be put to this queue
        language: str = "auto",
        speed_factor: float = 1.0,
        text_stream_end: asyncio.Event = None,        # let parent function know that text done
        audio_stream_end: asyncio.Event = None,       # let parent function know that audio done
        **kwargs: Any
    ) -> None:

        pipeline = TextToTextSegment(quick_start=True)

        try:
            # logger.info("Starting text-to-speech pipeline")
            # Create task but don't wait for it
            processing_task = asyncio.create_task(
                pipeline.process_text_stream_async(input_queue=text_queue)
            )

            while True:
                try:
                    # Shorter timeout for more responsive processing
                    segment = await pipeline.get_next_segment(timeout=0.1, raise_timeout=True)
                    # if segment:
                        # logger.info(f"---------------------Segment start with space: {segment.startswith(' ')}")
                        # logger.info(f"---------------------Segment end with space: {segment.endswith(' ')}")
                        # segment = segment.strip()

                    if segment:
                        await self.text_to_speech(
                            queue=queue,
                            text=segment,
                            language=language,
                            stream=True,
                            speed_factor=speed_factor,
                        )
                    # logger.info("Segment processed successfully")

                except asyncio.TimeoutError:
                    # Check if we should continue processing
                    if processing_task.done():
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
            await queue.put(TTSEvent(type="text_end"))  # signal that audio complete
            await pipeline.stop_processing()
            await pipeline.reset()
            await pipeline.clear_queue()
