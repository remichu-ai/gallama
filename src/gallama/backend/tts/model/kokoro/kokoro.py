from ...base import TTSBase
import os
from kokoro import KPipeline
from kokoro.pipeline import LANG_CODES
from kokoro.model import KModel
import torch
from gallama.data_classes import ModelSpec
from gallama.logger import logger
import asyncio
from typing import Dict, Any, AsyncGenerator, Tuple, AsyncIterator, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from split_lang import LangSplitter
from split_lang.model import SubString

from huggingface_hub import hf_hub_download


class KPipelineModified(KPipeline):
    """
    KPipeline class with modify method that not use another path that can work offline
    """

    def __init__(self, *args, **kwargs):
        self.voice_path = kwargs.pop("model_path", None)    # extra argument for path of the voices pt
        super().__init__(*args, **kwargs)


    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith('.pt'):
            f = voice
        else:

            # modification from here
            use_local_voice = False
            try:
                logger.info(f"Attempt to use voice from: {self.voice_path}/voices/{voice}.pt")
                if self.voice_path and os.path.exists(f"{self.voice_path}/voices/{voice}.pt"):
                    f = f"{self.voice_path}/voices/{voice}.pt"
                    use_local_voice = True
                else:
                    raise FileNotFoundError
            except Exception as e:
                logger.info(f"Error when loading voice {voice} from local: {e}. Fall back to download from huggingface")

            # fall back to download from hugging face
            if not use_local_voice:
                f = hf_hub_download(repo_id=KModel.REPO_ID, filename=f'voices/{voice}.pt')

            if not voice.startswith(self.lang_code):
                v = LANG_CODES.get(voice, voice)
                p = LANG_CODES.get(self.lang_code, self.lang_code)
                logger.warning(f'Language mismatch, loading {v} voice into {p} pipeline.')


        pack = torch.load(f, weights_only=True)
        self.voices[voice] = pack
        return pack


class TTSKokoro(TTSBase):
    def __init__(self, model_spec: ModelSpec):
        super().__init__(model_spec)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.sample_rate = 24000

    def convert_language_code(self, lan:str) -> str:
        if lan=="en":
            return "a"
        elif lan=="en-b":
            return "b"
        elif lan=="zh":
            return "z"
        elif lan=="ja":
            return "j"
        elif lan=="fr":
            return "f"
        elif lan=="hi":
            return "h"
        elif lan=="it":
            return "i"
        elif lan=="pt":
            return "p"
        else:
            logger.error(f"Unknown language code {lan}")
            return "a"  # return english

    @property
    def languages(self):
        return ["zh", "en", "en-b", "ja", "fr", "hi", "it", "pt"]

    def load_model(self, model_spec: ModelSpec):

        self.model = {}

        # create English pipeline:
        model_object = None

        for lang_code in self.languages:
            kokoro_lang_code = self.convert_language_code(lang_code)
            if model_object is None:
                pipeline = KPipelineModified(lang_code=kokoro_lang_code, model_path=model_spec.model_id)
                model_object = pipeline.model
            else:
                pipeline = KPipelineModified(lang_code=kokoro_lang_code, model=model_object, model_path=model_spec.model_id)

            self.model[lang_code] = pipeline

    async def stream_audio(
        self,
        inputs: Dict[str, Any],
        queue: asyncio.Queue,
        session_tracker: set = None,  # Set to track generations for a specific session
    ) -> None:
        """
        Feeds audio fragments directly into the provided queue for upstream consumption.

        Args:
            inputs: Dictionary of TTS parameters including text, ref_audio_path, etc.
            queue: asyncio.Queue where audio chunks will be placed
            session_tracker: Optional set to track generations for this specific session
        """

        task_id = str(time.time())
        if session_tracker is not None:
            session_tracker.add(task_id)

        loop = asyncio.get_running_loop()

        pipeline_to_use: KPipeline = self.model.get(inputs['text_lang'], self.model['en'])  # load en if language not found

        # remove language field as it is not part of KPipeline __call__ argument
        inputs.pop('text_lang')

        def run_tts():
            try:
                for gs, ps, audio_chunk in pipeline_to_use(**inputs):
                    # gs => graphemes/text
                    # ps => phonemes
                    future = asyncio.run_coroutine_threadsafe(
                        queue.put((self.sample_rate, audio_chunk.numpy())),     # kokoro return torch.Tensor, hence convert to numpy
                        loop
                    )
                    future.result()

            except Exception as e:
                logger.error(f"Error in TTS generation {task_id}: {e}")
                future = asyncio.run_coroutine_threadsafe(
                    queue.put(Exception(f"TTS error: {str(e)}")),
                    loop
                )
                future.result()
            finally:
                # Remove this task from session tracker if it exists
                if session_tracker is not None:
                    session_tracker.discard(task_id)

        loop.run_in_executor(self.executor, run_tts)

    def check_language(self, text_list: List[SubString]):
        supported_languages = self.languages

        for text in text_list:
            if text.lang not in supported_languages:
                text.lang = "en"

        if not text_list:
            return text_list

        grouped_list = [text_list[0]]
        for current in text_list[1:]:
            last = grouped_list[-1]
            # If current and last have the same language, merge them.
            if last.lang == current.lang:
                last.text += current.text
                last.length += current.length
            else:
                grouped_list.append(current)

        return grouped_list


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
        session_tracker: set = None,
        **kwargs: Any
    ) -> None | Tuple[int, np.ndarray] | AsyncIterator[Tuple[int, np.ndarray]] | str:


        if language == "auto":
            DEFAULT_LANG = "en"
            lang_splitter = LangSplitter(
                merge_across_punctuation=True
            )

            text_segmented_by_lan = self.check_language(
                lang_splitter.split_by_lang(text)
            )


            if len(text_segmented_by_lan) == 1:
                logger.debug(f"single language: {text_segmented_by_lan[0].lang}")
                return await self.text_to_speech_single_language(
                    text,
                    text_segmented_by_lan[0].lang,   # the detected language
                    stream,
                    batching,
                    batch_size,
                    queue,
                    voice,
                    speed_factor,
                    session_tracker,
                    ** kwargs
                )
            else:
                # example of each item in text_segmented_by_lan:
                # {'lang': 'zh', 'text': '我喜欢在雨天里听音乐。\n', 'score': 1.0}
                logger.info(f"Processing multi-language text with {len(text_segmented_by_lan)} segments")

                for text_chunk in text_segmented_by_lan:
                    _lang = text_chunk.lang
                    _text = text_chunk.text
                    # Multiple languages case

                    if stream and not queue:
                        raise ValueError("For streaming mode, a queue must be provided")

                    if not stream:
                        # Non-stream: Aggregate results from segments
                        full_audio = []
                        sample_rate = None

                        for segment in text_segmented_by_lan:
                            lang = segment.lang
                            seg_text = segment.text
                            if not seg_text.strip():
                                continue

                            # Process each segment with stream=False
                            sr, audio = await self.text_to_speech_single_language(
                                seg_text,
                                lang,
                                stream=False,
                                batching=batching,
                                batch_size=batch_size,
                                queue=None,
                                voice=voice,
                                speed_factor=speed_factor,
                                session_tracker=session_tracker,
                                **kwargs
                            )

                            if sample_rate is None:
                                sample_rate = sr
                            elif sr != sample_rate:
                                raise ValueError(f"Sample rate mismatch between segments: {sr} vs {sample_rate}")

                            full_audio.append(audio)

                        if not full_audio:
                            return sample_rate or self.sample_rate, np.array([], dtype=np.int16)

                        return sample_rate, np.concatenate(full_audio)
                    else:
                        # Stream: Process segments sequentially using shared queue
                        for segment in text_segmented_by_lan:
                            lang = segment.lang
                            seg_text = segment.text
                            if not seg_text.strip():
                                continue

                            # Wait until all previous tasks are completed
                            while session_tracker:
                                await asyncio.sleep(0.1)

                            await self.text_to_speech_single_language(
                                seg_text,
                                lang,
                                stream=True,
                                batching=batching,
                                batch_size=batch_size,
                                queue=queue,
                                voice=voice,
                                speed_factor=speed_factor,
                                session_tracker=session_tracker,
                                **kwargs
                            )
                        return None
        else:

            return await self.text_to_speech_single_language(
                text,
                language,
                stream,
                batching,
                batch_size,
                queue,
                voice,
                speed_factor,
                session_tracker,
                ** kwargs
            )


    async def text_to_speech_single_language(
        self,
        text: str,
        language: str = "auto",
        stream: bool = False,
        batching: bool = False,
        batch_size: int = 1,
        queue: asyncio.Queue = None,
        voice: str = None,
        speed_factor: float = None,
        session_tracker: set = None,
        **kwargs: Any
    ) -> None | Tuple[int, np.ndarray] | AsyncIterator[Tuple[int, np.ndarray]] | str:
        """
        Generate audio chunks from text and put them into an asyncio Queue.

        Args:
            text: Text to convert to speech
            language: Language of the text (default: "auto")
            stream: Whether to stream the audio in chunks (default: False)
            voice: the name of the voice to use (default: None), if None, will use default voice
            speed_factor: Speed factor for the audio playback (default: 1.0)
            batching: whether to using parallel batching, will be faster but require more memory (default: False)
            batch_size: batch size if use batching (default: 1)
            queue: Optional asyncio Queue to receive audio chunks
            session_tracker: Optional set to track generations for this specific session
            kwargs: Additional parameters to pass to the text_to_speech function

        Returns:
            If stream=False: Tuple of (sample_rate, concatenated_audio_data)
            If stream=True: None (audio chunks are sent to the provided queue)
        """
        logger.debug(f"Converting TTS: {text}")

        if stream and not queue:
            # for streaming, the result will be written directly to the queue hence it is required parameter
            raise Exception("For streaming mode, the queue must be provided")

        if voice:
            voice_to_use = voice
        else:
            voice_to_use = "af_heart"
            speed_factor = 1

        if speed_factor:
            speed_factor_to_use = speed_factor
        else:
            raise Exception("There must be at least one voice to use")

        params = {
            "text": text,
            "text_lang": language,
            "voice": voice_to_use,
            "speed": speed_factor_to_use,
        }

        if stream:
            if queue is None:
                queue = asyncio.Queue()
            await self.stream_audio(
                inputs=params,
                queue=queue,
                session_tracker=session_tracker
            )
            return None  # Since we're using a queue, no direct return value needed
        else:
            # For non-streaming mode, create a temporary queue and collect all chunks
            temp_queue = asyncio.Queue()
            temp_session_tracker = set()  # Create temporary session tracker if none provided
            session_tracker_to_use = session_tracker if session_tracker is not None else temp_session_tracker

            await self.stream_audio(
                inputs=params,
                queue=temp_queue,
                session_tracker=session_tracker_to_use
            )

            try:
                # Collect all audio chunks into a single array
                audio_chunks = []
                sample_rate = None

                # Process chunks while the session is active or queue has items
                while session_tracker_to_use or not temp_queue.empty():
                    try:
                        # Use asyncio.wait_for with a timeout for the queue.get()
                        chunk = await asyncio.wait_for(temp_queue.get(), timeout=0.1)
                        if isinstance(chunk, Exception):
                            raise chunk

                        current_sample_rate, audio_data = chunk
                        if sample_rate is None:
                            sample_rate = current_sample_rate
                        elif sample_rate != current_sample_rate:
                            raise ValueError(
                                f"Inconsistent sample rates detected: {sample_rate} vs {current_sample_rate}")

                        if audio_data.shape[0] > 0:  # Only add non-empty chunks
                            audio_chunks.append(audio_data)
                        temp_queue.task_done()
                    except asyncio.TimeoutError:
                        # Timeout is expected when queue is empty, just continue the loop
                        continue

                if not audio_chunks:
                    return 0, np.array([], dtype=np.int16)

                return sample_rate, np.concatenate(audio_chunks)

            except Exception as e:
                logger.error(f"Error during audio generation: {e}")
                # self.model.stop()
                raise