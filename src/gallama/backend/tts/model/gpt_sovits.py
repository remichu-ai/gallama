from ..base import TTSBase
from .gpt_sovits_source import TTS, TTS_Config
from typing import Dict, Any, AsyncGenerator, Tuple, AsyncIterator
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from gallama.data_classes import ModelSpec
from gallama.logger.logger import logger
import time


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

    async def stream_audio(
        self,
        inputs: Dict[str, Any],
        queue: asyncio.Queue,
    ) -> None:
        """
        Feeds audio fragments directly into the provided queue for upstream consumption.

        Args:
            inputs: Dictionary of TTS parameters including text, ref_audio_path, etc.
            queue: asyncio.Queue where audio chunks will be placed
        """
        loop = asyncio.get_running_loop()

        # Function to run in the executor that will feed the queue
        def run_tts():
            try:


                for audio_chunk in self.tts.run(inputs):
                    # Put the chunk in the queue from the thread
                    future = asyncio.run_coroutine_threadsafe(
                        queue.put(audio_chunk),
                        loop
                    )
                    # Ensure the put operation completes
                    future.result()


            except Exception as e:
                logger.error(f"Error in TTS generation: {e}")
                # Put the error in the queue
                future = asyncio.run_coroutine_threadsafe(
                    queue.put(Exception(f"TTS error: {str(e)}")),
                    loop
                )
                future.result()
            finally:
                # Signal that we're done by putting None in the queue
                pass

        # Start the TTS processing in the executor and return immediately
        loop.run_in_executor(self.executor, run_tts)

    def stop(self):
        """Stops the TTS processing"""
        self.tts.stop()


class TTS_GPT_SoVITS(TTSBase):
    def __init__(self, model_spec: ModelSpec):
        super().__init__(model_spec)


    def load_model(self, model_spec: ModelSpec):
        # create the config object with parameters from gallama yaml config file
        config = TSS_ConfigModified(self.backend_extra_args)
        self.model = AsyncTTSWrapper(TTS(config))

        # set other required parameter for speech generation
        # self.ref_audio_path = self.backend_extra_args.get('ref_audio_path')
        # self.ref_audio_transcription = self.backend_extra_args.get('ref_audio_transcription')
        self.chunk_size_in_s = self.backend_extra_args.get('chunk_size_in_s', 0.5)


    async def text_to_speech(
        self,
        text: str,
        language: str = "auto",
        stream: bool = False,
        batching: bool = False,
        batch_size: int = 1,
        queue: asyncio.Queue = None,
        voice: str = None,
        speed_factor: float = None,
        end_signal: str = "conversion_done",
        **kwargs: Any
    ) -> None | Tuple[int, np.ndarray] | AsyncIterator[Tuple[int, np.ndarray]] | str:
        """
        Generate audio chunks from text and put them into an asyncio Queue.

        Args:
            text: Text to convert to speech
            language: Language of the text (default: "auto")
            stream: Whether to stream the audio in chunks (default: False)
            speed_factor: Speed factor for the audio playback (default: 1.0)
            batching: whether to using parallel batching, will be faster but require more memory (default: False)
            batch_size: batch size if use batching (default: 1)
            queue: Optional asyncio Queue to receive audio chunks
            kwargs: Additional parameters to pass to the text_to_speech function
            end_signal: str = "conversion_done" -> the string to put into queue when conversion is done

        Returns:
            If stream=False: Tuple of (sample_rate, concatenated_audio_data)
            If stream=True: None (audio chunks are sent to the provided queue)
        """
        logger.info(f"----------Converting: {text}")

        if stream and not queue:
            # for streaming, the result will be written directly to the queue hence it is required parameter
            raise Exception("For streaming mode, the queue must be provided")

        if voice:
            if voice in self.voice_list:
                voice_to_use = self.voice[voice]
            else:
                voice_to_use = self.default_voice
        else:
            voice_to_use = self.default_voice

        if speed_factor:
            speed_factor_to_use = speed_factor
        else:
            speed_factor_to_use = voice_to_use.speed_factor or 1.0

        params = {
            "text": text,
            "text_lang": "en",
            "ref_audio_path": voice_to_use.ref_audio_path,
            "aux_ref_audio_paths": [],
            "prompt_text": voice_to_use.ref_audio_transcription,
            "prompt_lang": "en",
            "top_k": 5,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut0",
            "batch_size": batch_size,
            "batch_threshold": 0.75,
            # "split_bucket": not stream,  # Disable split_bucket when streaming
            "split_bucket": True,  # Disable split_bucket when streaming
            "return_fragment": stream,  # Enable fragments when streaming
            # "return_fragment": False,  # Enable fragments when streaming
            "speed_factor": speed_factor_to_use,  # Use the provided speed_factor
            "fragment_interval": self.chunk_size_in_s,
            "seed": -1,
            # "parallel_infer": batching,
            "parallel_infer": 3,
            "repetition_penalty": 1.35,
            **kwargs  # Allow overriding of any parameters
        }

        if stream:
            if queue is None:
                queue = asyncio.Queue()
            await self.model.stream_audio(inputs=params, queue=queue)
            return None  # Since we're using a queue, no direct return value needed
        else:
            # For non-streaming mode, create a temporary queue and collect all chunks
            temp_queue = asyncio.Queue()
            await self.model.stream_audio(inputs=params, queue=temp_queue)

            try:
                # Collect all audio chunks into a single array
                audio_chunks = []
                sample_rate = None

                while True:
                    chunk = await temp_queue.get()
                    if chunk is None:  # End of stream
                        break
                    if isinstance(chunk, Exception):
                        raise chunk

                    current_sample_rate, audio_data = chunk
                    if sample_rate is None:
                        sample_rate = current_sample_rate
                    elif sample_rate != current_sample_rate:
                        raise ValueError(f"Inconsistent sample rates detected: {sample_rate} vs {current_sample_rate}")

                    if audio_data.shape[0] > 0:  # Only add non-empty chunks
                        audio_chunks.append(audio_data)

                if not audio_chunks:
                    pass
                    # return 0, np.array([], dtype=np.int16)

                return sample_rate, np.concatenate(audio_chunks)

            except Exception as e:
                logger.error(f"Error during audio generation: {e}")
                self.model.stop()
                raise