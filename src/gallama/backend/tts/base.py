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
        raise NotImplementedError(
            "The synthesize method must be implemented by the derived class."
        )

    async def text_to_speech_to_queue(
        self,
        queue: asyncio.Queue,
        text: str,
        language:str = "auto",
        speed_factor: float = 1.0,
        **kwargs: Any    # use for overwriting any other parameter below
    ):
        """
        Generate audio chunks from text and put them into an asyncio Queue.

        Args:
            queue: asyncio.Queue to put the audio chunks into
            text: Text to convert to speech
            language: Language of the text (default: "auto")
            stream: Whether to stream the audio in chunks (default: False)
            speed_factor: Speed factor for the audio playback (default: 1.0)
            kwargs: Additional parameters to pass to the text_to_speech function
        """

        try:
            # Use the existing text_to_speech method
            audio_result = await self.text_to_speech(
                    text=text,
                    language=language,
                    stream=True,
                    speed_factor=speed_factor,
                    queue=queue,
                    **kwargs
            )

            # # Process the audio chunks
            # chunks_processed = 0
            # async for sampling_rate, audio_data in audio_result:
            #     if audio_data.shape[0] == 0:
            #         break
            #
            #     chunks_processed += 1
            #     logger.info(f"Processing chunk #{chunks_processed}")
            #     logger.info(f"  - Queue size before put: {queue.qsize()}")
            #     logger.info(f"  - Audio shape: {audio_data.shape}")
            #
            #     # Put both sampling rate and audio data into queue
            #     await asyncio.wait_for(queue.put((sampling_rate, audio_data)), timeout=5.0)
            #     logger.info(f"  - Queue size after put: {queue.qsize()}")
            #
            # # Signal completion by putting None into the queue
            # await asyncio.wait_for(queue.put(None), timeout=5.0)

        except Exception as e:
            logger.error(f"Error in text_to_speech_to_queue: {e}")
            # Put the error in the queue to notify consumers
            await queue.put(Exception(f"Text-to-speech error: {str(e)}"))
            self.model.stop()
            raise

    async def text_stream_to_speech_to_queue(
        self,
        text_stream: AsyncIterator,
        queue: asyncio.Queue,
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
                        await self.text_to_speech_to_queue(
                            queue=queue,
                            text=segment,
                            language=language,
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
                                await self.text_to_speech_to_queue(
                                    queue=queue,
                                    text=segment,
                                    language=language,
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