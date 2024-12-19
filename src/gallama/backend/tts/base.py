from typing import Any, AsyncIterator
import asyncio
from .text_processor import TextToTextSegment
from ...data_classes import ModelSpec
from ...logger import logger

class TTSBase:
    """this is the base interface for all TTS models"""

    def __init__(self, model_spec: ModelSpec):
        self.model_name = model_spec.model_name
        self.model = None

        self.voice = model_spec.voice
        # TODO, get a list of available voice

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
        text_stream: AsyncIterator,
        queue: asyncio.Queue,           # audio chunk will be put to this queue
        language: str = "auto",
        speed_factor: float = 1.0,
        **kwargs: Any
    ) -> None:

        pipeline = TextToTextSegment(quick_start=True)

        try:
            # logger.info("Starting text-to-speech pipeline")
            # Create task but don't wait for it
            processing_task = asyncio.create_task(
                pipeline.process_text_stream_async(text_stream, end_stream=False)
            )

            while True:
                try:
                    # Shorter timeout for more responsive processing
                    segment = await pipeline.get_next_segment(timeout=0.1)

                    if segment is None:
                        # Check if processing task is done and queue is empty
                        if processing_task.done() and pipeline.processing_queue.empty():
                            break
                        continue

                    else:
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
                    if processing_task.done() and pipeline.processing_queue.empty():
                        # Process any remaining text in buffer before exiting
                        if pipeline.get_buffer_contents().strip():
                            final_text = pipeline.get_buffer_contents()
                            segments = pipeline.segment_text(final_text)
                            for segment in segments:
                                await self.text_to_speech(
                                    queue=queue,
                                    text=segment,
                                    language=language,
                                    stream=True,
                                    speed_factor=speed_factor,
                                )
                        break
                    continue

        except Exception as e:
            logger.error(f"Error in text_stream_to_speech_to_queue: {str(e)}", exc_info=True)
            await queue.put(Exception(f"Text-to-speech error: {str(e)}"))
            if hasattr(self, 'model') and hasattr(self.model, 'stop'):
                self.model.stop()
            raise
        finally:
            logger.info("Cleaning up pipeline")
            await pipeline.stop_processing()
            await pipeline.reset()
            await pipeline.clear_queue()
            # Signal completion to the queue
            await queue.put(None)