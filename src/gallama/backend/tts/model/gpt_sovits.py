from ..base import TTSBase
from .gpt_sovits_source import TTS, TTS_Config
from typing import Dict, Any, AsyncGenerator, Tuple, AsyncIterator
import numpy as np
import asyncio
from ..text_processor import TextToTextSegment
from concurrent.futures import ThreadPoolExecutor
from gallama.data_classes import (
    ModelSpec
)

class TSS_ConfigModified(TTS_Config):
    def __init__(self, backend_extra_args):
        """ overwrite the original logic to change the path to point to gallama folder"""

        self.config = backend_extra_args

        # get argument set in config file:
        backend_arg_list = ["bert_base_path", "cnhuhbert_base_path", "device", "is_half", "t2s_weights_path", "version", "vits_weights_path"]

        for arg_name in backend_arg_list:
            value_to_set = backend_extra_args.get(arg_name)

            if value_to_set is None:
                raise ValueError(f"For GPT_SoVits argument {arg_name} is required and must be set in model_config.yaml")

            # set this as attribute of the class
            setattr(self, arg_name, value_to_set)

        self.max_sec = None
        self.hz:int = 50
        self.semantic_frame_rate:str = "25hz"
        self.segment_size:int = 20480
        self.filter_length:int = 2048
        self.sampling_rate:int = 32000
        self.hop_length:int = 640
        self.win_length:int = 2048
        self.n_speakers:int = 300

    def save_configs(self, configs_path:str=None)->None:
        pass


class AsyncTTSWrapper:
    def __init__(self, tts_instance, executor: ThreadPoolExecutor = None):
        self.tts = tts_instance
        self.executor = executor or ThreadPoolExecutor(max_workers=1)

    async def stream_audio(self, inputs: Dict[str, Any]) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Creates an async generator that yields audio fragments from the TTS system.

        Args:
            inputs: Dictionary of TTS parameters including text, ref_audio_path, etc.
                   Make sure return_fragment=True is set in the inputs.

        Yields:
            Tuples of (sample_rate, audio_data)
        """
        # Create a queue to receive audio fragments
        queue = asyncio.Queue()

        # Function to run in the executor that will feed the queue
        def run_tts():
            try:
                for audio_chunk in self.tts.run(inputs):
                    # Handle stop condition if needed
                    if self.tts.stop_flag:
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(audio_chunk), loop)
            finally:
                # Signal that we're done by putting None in the queue
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop = asyncio.get_running_loop()
        # Start the TTS processing in the executor
        loop.run_in_executor(self.executor, run_tts)

        # Yield audio fragments as they become available
        while True:
            chunk = await queue.get()
            if chunk is None:  # End of stream
                break
            yield chunk

    def stop(self):
        """Stops the TTS processing"""
        self.tts.stop()


class TTS_GPT_SoVITS(TTSBase):
    def __init__(self, model_spec: ModelSpec):
        super().__init__(model_spec)

        # create the config object with parameters from gallama yaml config file
        self.config = TSS_ConfigModified(self.backend_extra_args)
        self.model = AsyncTTSWrapper(TTS(self.config))

        # set other required parameter for speech generation
        self.ref_audio_path = self.backend_extra_args.get('ref_audio_path')
        self.ref_audio_transcription = self.backend_extra_args.get('ref_audio_transcription')
        self.chunk_size_in_s = self.backend_extra_args.get('chunk_size_in_s')

        if self.ref_audio_path is None or self.ref_audio_transcription is None:
            raise ValueError("Both ref_audio_path and ref_audio_transcription must be set for GPT_SoVITS")

    async def text_to_speech(
        self,
        text: str,
        language:str = "auto",
        stream: bool = False,    # non stream return numpy array of whole speech, True will return iterator instead
        speed_factor: float = 1.0,
        batching: bool = False,
        batch_size: int = 1,
        *kwargs: Any    # use for overwriting any other parameter below
    ):
        """
        Generate audio chunks from text and put them into an asyncio Queue.

        Args:
            text: Text to convert to speech
            language: Language of the text (default: "auto")
            stream: Whether to stream the audio in chunks (default: False)
            speed_factor: Speed factor for the audio playback (default: 1.0)
            batching: whether to using parallel batching, will be faster but require more memmory (default: False)
            batch_size: batch size if use batching (default: 1),
            kwargs: Additional parameters to pass to the text_to_speech function
        """


        params = {
            "text": text,
            "text_lang": "en",
            "ref_audio_path": self.ref_audio_path,
            "aux_ref_audio_paths": [],
            "prompt_text": self.ref_audio_transcription,
            "prompt_lang": "en",
            "top_k": 5,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut5",
            "batch_size": batch_size,
            "batch_threshold": 0.75,
            "split_bucket": not stream,  # Disable split_bucket when streaming
            "return_fragment": stream,  # Enable fragments when streaming
            "speed_factor": 1.0,
            "fragment_interval": self.chunk_size_in_s,  # Use the provided chunk size
            "seed": -1,
            "parallel_infer": batching,
            "repetition_penalty": 1.35
        }

        # over any other arguments set from the api call
        params.update(kwargs)

        if stream:
            async def audio_stream() -> AsyncGenerator[np.ndarray, None]:
                try:
                    async for sampling_rate, audio_data in self.model.stream_audio(params):
                        if audio_data.shape[0] == 0:
                            break
                        yield sampling_rate, audio_data
                except Exception as e:
                    print(f"Error during audio streaming: {e}")
                    self.model.stop()
                    raise

            return audio_stream()   # return an iterator instead
        else:
            try:
                # Collect all audio chunks into a single array
                audio_chunks = []
                generated_sampling_rate = 0
                async for sample_rate, audio_data in self.model.stream_audio(params):
                    if audio_data.shape[0] == 0:
                        break
                    audio_chunks.append(audio_data)
                    generated_sampling_rate = sample_rate

                if not audio_chunks:
                    return 0, np.array([], dtype=np.int16)

                return generated_sampling_rate, np.concatenate(audio_chunks)
            except Exception as e:
                print(f"Error during audio generation: {e}")
                self.model.stop()
                raise


    async def text_to_speech_to_queue(
        self,
        queue: asyncio.Queue,
        text: str,
        language:str = "auto",
        speed_factor: float = 1.0,
        *kwargs: Any    # use for overwriting any other parameter below
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
                stream=True,  # Always use stream mode to get chunks
                speed_factor=speed_factor,
                **kwargs
            )

            # Process the audio chunks
            async for sampling_rate, audio_data in audio_result:
                if audio_data.shape[0] == 0:
                    break

                # Put both sampling rate and audio data into queue
                await queue.put((sampling_rate, audio_data))

            # Signal completion by putting None into the queue
            await queue.put(None)

        except Exception as e:
            print(f"Error in text_to_speech_to_queue: {e}")
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
        """
        Process a stream of text chunks and convert them to speech, putting audio chunks into a queue.

        Args:
            text_stream: AsyncIterator yielding text chunks
            queue: asyncio.Queue to receive audio chunks
            language: Language of the text
            speed_factor: Speed factor for speech
            **kwargs: Additional parameters for text_to_speech
        """
        try:
            pipeline = TextToTextSegment(quick_start=True)

            # Process the incoming text stream
            await pipeline.process_text_stream_async(
                text_stream,
                end_stream=True
            )

            # Process segments as they become available
            while True:
                segment = await pipeline.get_next_segment(timeout=0.5)
                if segment is None:
                    break

                # Use existing method to handle audio conversion and queueing
                await self.text_to_speech_to_queue(
                    queue=queue,
                    text=segment,
                    language=language,
                    speed_factor=speed_factor,
                    **kwargs
                )

        except Exception as e:
            print(f"Error in text_stream_to_speech_to_queue: {e}")
            await queue.put(Exception(f"Text-to-speech error: {str(e)}"))
            self.model.stop()
            raise
        finally:
            # Cleanup
            await pipeline.stop_processing()
            await pipeline.reset()
            await pipeline.clear_queue()