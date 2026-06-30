import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

import numpy as np

from omegaconf import OmegaConf

from ....data_classes import LanguageType, ModelSpec, TranscriptionResponse
from ....logger import logger
from ..base import ASRBase

try:
    import torch
except ImportError:  # pragma: no cover - optional backend dependency
    torch = None

try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
except ImportError:  # pragma: no cover - optional backend dependency
    nemo_asr = None
    Hypothesis = None
    CacheAwareStreamingAudioBuffer = None


NEMOTRON_DEFAULT_LOCALES = {
    "en": "en-US",
    "es": "es-US",
    "fr": "fr-FR",
    "it": "it-IT",
    "pt": "pt-BR",
    "nl": "nl-NL",
    "de": "de-DE",
    "tr": "tr-TR",
    "ru": "ru-RU",
    "ar": "ar-AR",
    "hi": "hi-IN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "vi": "vi-VN",
    "uk": "uk-UA",
    "pl": "pl-PL",
    "sv": "sv-SE",
    "cs": "cs-CZ",
    "nb": "nb-NO",
    "no": "nb-NO",
    "da": "da-DK",
    "bg": "bg-BG",
    "fi": "fi-FI",
    "hr": "hr-HR",
    "sk": "sk-SK",
    "zh": "zh-CN",
    "hu": "hu-HU",
    "ro": "ro-RO",
    "et": "et-EE",
    "el": "el-GR",
    "lt": "lt-LT",
    "lv": "lv-LV",
    "mt": "mt-MT",
    "sl": "sl-SI",
    "he": "he-IL",
    "th": "th-TH",
    "nn": "nn-NO",
}


@dataclass
class NemoSegment:
    start: float
    end: float
    text: str
    words: Optional[List[Tuple[float, float, str]]] = None


class ASRNeMo(ASRBase):
    """NeMo cache-aware ASR backend for Nemotron streaming checkpoints."""

    sampling_rate = 16000

    def load_model(self, model_spec: ModelSpec):
        if nemo_asr is None:
            raise ImportError(
                "nemo_asr backend requires NVIDIA NeMo. Install with: "
                "pip install Cython packaging && "
                "pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'"
            )

        extra = model_spec.backend_extra_args or {}
        self.model_id = model_spec.model_id
        self.model_name = model_spec.model_name
        self.sample_rate = int(extra.get("sample_rate", self.sampling_rate))
        self.device = extra.get("device") or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.compute_dtype = extra.get("compute_dtype", "float32")
        self.use_amp = bool(extra.get("amp", False))
        self.att_context_size = extra.get("att_context_size", [56, 3])
        self.target_lang = self._normalize_language(
            extra.get("target_lang") or model_spec.language or "auto"
        )
        self.strip_lang_tags = bool(extra.get("strip_lang_tags", True))
        self.online_normalization = bool(extra.get("online_normalization", False))
        self.pad_and_drop_preencoded = bool(extra.get("pad_and_drop_preencoded", False))

        model_id_path = Path(str(self.model_id)).expanduser() if self.model_id else None
        if model_id_path and model_id_path.is_dir():
            nemo_files = sorted(model_id_path.glob("*.nemo"))
            if len(nemo_files) != 1:
                raise ValueError(
                    f"nemo_asr model_id directory must contain exactly one .nemo file: {model_id_path}"
                )
            self.model_id = str(nemo_files[0])

        if self.model_id and str(self.model_id).endswith(".nemo"):
            model = nemo_asr.models.ASRModel.restore_from(restore_path=self.model_id)
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)

        self._configure_model(model, self.target_lang)

        # Monkey-patch _setup_transcribe_dataloader to force default_prompt_mode="auto".
        # NeMo's LhotseSpeechToTextBpeDatasetWithPromptIndex defaults to "unified" mode,
        # which tries to read language from cut metadata. Plain audio files have no
        # language metadata (None), causing "Unknown prompt key: 'None'" errors.
        self._patch_transcribe_dataloader(model)

        if hasattr(model, "to"):
            model = model.to(device=self.device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    @staticmethod
    def _patch_transcribe_dataloader(model):
        """Monkey-patch _setup_transcribe_dataloader to inject default_prompt_mode='auto'.

        NeMo's _setup_transcribe_dataloader builds a fresh dl_config dict that omits
        default_prompt_mode, so LhotseSpeechToTextBpeDatasetWithPromptIndex defaults to
        "unified" mode. In unified mode it sometimes reads cut.supervisions[0].language
        which is None for plain audio files (no manifest), causing ValueError.

        We monkey-patch _setup_dataloader_from_config instead, which creates the actual
        dataset, to inject default_prompt_mode='auto' into the dataset config.
        """
        original_setup = model._setup_dataloader_from_config

        def patched_setup(config):
            config = OmegaConf.to_container(config, resolve=True) if hasattr(config, 'keys') else dict(config or {})
            config.setdefault("default_prompt_mode", "auto")
            return original_setup(config)

        model._setup_dataloader_from_config = patched_setup

    def _configure_model(self, model, target_lang: Optional[str] = None):
        if self.att_context_size is not None and hasattr(model.encoder, "set_default_att_context_size"):
            model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)

        if target_lang and hasattr(model, "set_inference_prompt"):
            model.set_inference_prompt(target_lang)
            if hasattr(model, "decoding") and hasattr(model.decoding, "set_strip_lang_tags"):
                model.decoding.set_strip_lang_tags(self.strip_lang_tags)

    @staticmethod
    def _normalize_language(language: Optional[LanguageType]) -> str:
        if not language:
            return "auto"
        if isinstance(language, list):
            language = language[0] if language else "auto"
        if language == "auto":
            return "auto"
        return NEMOTRON_DEFAULT_LOCALES.get(str(language), str(language))

    @staticmethod
    def _extract_text(item) -> str:
        if hasattr(item, "text"):
            return item.text or ""
        if isinstance(item, dict):
            return item.get("text") or item.get("pred_text") or ""
        return str(item or "")

    def _extract_transcriptions(self, outputs) -> List[str]:
        if outputs is None:
            return []
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if not isinstance(outputs, list):
            outputs = [outputs]
        return [self._extract_text(item) for item in outputs]

    def _load_audio_array(self, audio: Union[str, BinaryIO, np.ndarray]) -> np.ndarray:
        import librosa

        if isinstance(audio, np.ndarray):
            audio_array = audio.astype(np.float32, copy=False)
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            return np.ascontiguousarray(audio_array, dtype=np.float32)

        if isinstance(audio, (str, os.PathLike)):
            audio_array, _ = librosa.load(str(audio), sr=self.sample_rate, mono=True)
            return np.ascontiguousarray(audio_array, dtype=np.float32)

        name = getattr(audio, "name", None) or ""
        suffix = Path(name).suffix or ".audio"
        current_pos = None
        if hasattr(audio, "tell") and hasattr(audio, "seek"):
            try:
                current_pos = audio.tell()
                audio.seek(0)
            except Exception:
                current_pos = None

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio.read())
            tmp_path = tmp.name

        if current_pos is not None:
            try:
                audio.seek(current_pos)
            except Exception:
                pass

        try:
            audio_array, _ = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
            return np.ascontiguousarray(audio_array, dtype=np.float32)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @contextlib.contextmanager
    def _audio_as_wav_path(self, audio: Union[str, BinaryIO, np.ndarray]):
        import soundfile as sf

        if isinstance(audio, (str, os.PathLike)):
            yield str(audio)
            return

        audio_array = self._load_audio_array(audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio_array, self.sample_rate, subtype="PCM_16")
            yield tmp_path
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_to_segment(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        init_prompt: str = "",
        temperature: float = 0.0,
        language: LanguageType = None,
        batch: bool = False,
        batch_size: int = 8,
    ) -> List[NemoSegment]:
        target_lang = self._normalize_language(language) if language else self.target_lang
        self._configure_model(self.model, target_lang)

        audio_array = self._load_audio_array(audio)
        duration = len(audio_array) / self.sample_rate if self.sample_rate else 0.0

        with self._audio_as_wav_path(audio_array) as audio_path:
            outputs = self.model.transcribe([audio_path])
        texts = self._extract_transcriptions(outputs)
        text = texts[0] if texts else ""
        return [NemoSegment(start=0.0, end=duration, text=text)]

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        init_prompt: str = "",
        temperature: float = 0.0,
        language: LanguageType = None,
        include_segments: bool = False,
        batch: bool = False,
    ) -> TranscriptionResponse:
        segments = self.transcribe_to_segment(
            audio,
            init_prompt=init_prompt,
            temperature=temperature,
            language=language,
            batch=batch,
        )
        text = self.segment_to_long_text(segments)
        if not include_segments:
            return TranscriptionResponse(text=text)
        return TranscriptionResponse(
            text=text,
            segments=[{"start": s.start, "end": s.end, "text": s.text} for s in segments],
        )

    def segment_to_timestamped_words(self, segments: List[NemoSegment]) -> List[Tuple[float, float, str]]:
        words = []
        for segment in segments:
            if segment.words:
                words.extend(segment.words)
                continue
            if segment.text:
                words.append((segment.start, segment.end, segment.text))
        return words

    def segment_to_long_text(self, segments: List[NemoSegment]) -> str:
        return self.sep.join(segment.text for segment in segments if segment.text).strip()

    def segments_end_ts(self, res):
        return [segment.end for segment in res]

    def use_vad(self):
        return None

    def create_streaming_session(self, language: LanguageType = None):
        target_lang = self._normalize_language(language) if language else self.target_lang
        self._configure_model(self.model, target_lang)
        return NemoStreamingSession(self, target_lang=target_lang)


class NemoStreamingSession:
    def __init__(self, backend: ASRNeMo, target_lang: str):
        if CacheAwareStreamingAudioBuffer is None:
            raise ImportError("NeMo CacheAwareStreamingAudioBuffer is unavailable")

        self.backend = backend
        self.model = backend.model
        self.target_lang = target_lang
        self.compute_dtype = torch.float32 if torch else None
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = (
            self.model.encoder.get_initial_cache_state(batch_size=1)
        )
        self.previous_hypotheses = None
        self.previous_pred_out = None
        self.step_num = 0
        self.full_text = ""
        self.closed = False

        right_context = int((backend.att_context_size or [56, 3])[1])
        self.chunk_seconds = (right_context + 1) * 0.08
        self.chunk_samples = max(1, int(self.chunk_seconds * backend.sample_rate))
        self.buffer = np.array([], dtype=np.float32)

    def reset(self):
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = (
            self.model.encoder.get_initial_cache_state(batch_size=1)
        )
        self.previous_hypotheses = None
        self.previous_pred_out = None
        self.step_num = 0
        self.full_text = ""
        self.buffer = np.array([], dtype=np.float32)

    def accept_audio(self, audio: np.ndarray, is_final: bool = False) -> str:
        if self.closed:
            return ""
        if audio is not None and len(audio) > 0:
            self.buffer = np.concatenate([self.buffer, audio.astype(np.float32, copy=False)])

        delta = ""
        while len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[: self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples :]
            delta += self._process_array(chunk, is_final=False)

        if is_final and len(self.buffer) > 0:
            delta += self._process_array(self.buffer, is_final=True)
            self.buffer = np.array([], dtype=np.float32)
        return delta

    def finish(self) -> str:
        return self.accept_audio(np.array([], dtype=np.float32), is_final=True)

    def close(self):
        self.closed = True

    def _process_array(self, audio: np.ndarray, is_final: bool) -> str:
        if len(audio) == 0 and not is_final:
            return ""

        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self.model,
            online_normalization=self.backend.online_normalization,
            pad_and_drop_preencoded=self.backend.pad_and_drop_preencoded,
        )
        if len(audio) > 0:
            streaming_buffer.append_audio(audio, stream_id=-1)

        transcribed_texts = None
        for chunk_audio, chunk_lengths in streaming_buffer:
            if chunk_audio.numel() == 0 or (chunk_lengths is not None and chunk_lengths.sum() == 0):
                continue
            if self.compute_dtype is not None and hasattr(chunk_audio, "to"):
                chunk_audio = chunk_audio.to(self.compute_dtype)
            try:
                (
                    self.previous_pred_out,
                    transcribed_texts,
                    self.cache_last_channel,
                    self.cache_last_time,
                    self.cache_last_channel_len,
                    self.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self.cache_last_channel,
                    cache_last_time=self.cache_last_time,
                    cache_last_channel_len=self.cache_last_channel_len,
                    keep_all_outputs=is_final or streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=self.previous_pred_out,
                    drop_extra_pre_encoded=self._drop_extra_pre_encoded(),
                    return_transcription=True,
                )
                self.step_num += 1
            except RuntimeError:
                logger.debug("conformer_stream_step skipped empty/invalid chunk")
                continue

        texts = self.backend._extract_transcriptions(transcribed_texts)
        if not texts:
            return ""
        new_full_text = texts[0]
        delta = self._delta_from_full_text(new_full_text)
        self.full_text = new_full_text
        return delta

    def _drop_extra_pre_encoded(self) -> int:
        if self.step_num == 0 and not self.backend.pad_and_drop_preencoded:
            return 0
        return getattr(self.model.encoder.streaming_cfg, "drop_extra_pre_encoded", 0)

    def _delta_from_full_text(self, new_full_text: str) -> str:
        if not new_full_text:
            return ""
        if new_full_text.startswith(self.full_text):
            return new_full_text[len(self.full_text):]
        return new_full_text
