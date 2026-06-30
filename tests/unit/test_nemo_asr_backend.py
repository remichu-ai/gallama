from types import SimpleNamespace

import numpy as np

from gallama.backend.stt.asr_processor import ASRProcessor
from gallama.backend.stt.nemo_asr import model as nemo_module
from gallama.data_classes.data_class import ModelSpec
from gallama.data_classes.realtime_client_proto import TurnDetectionConfig


class FakeDecoding:
    def __init__(self):
        self.strip_lang_tags = None

    def set_strip_lang_tags(self, value):
        self.strip_lang_tags = value


class FakeEncoder:
    def __init__(self):
        self.att_context_size = None
        self.streaming_cfg = SimpleNamespace(drop_extra_pre_encoded=3)

    def set_default_att_context_size(self, att_context_size):
        self.att_context_size = att_context_size

    def get_initial_cache_state(self, batch_size):
        return f"channel-{batch_size}", f"time-{batch_size}", f"length-{batch_size}"


class FakeModel:
    def __init__(self):
        self.encoder = FakeEncoder()
        self.decoding = FakeDecoding()
        self.prompts = []
        self.transcribe_calls = []
        self.stream_step_calls = []
        self.stream_texts = ["hello", "hello world"]

    def set_inference_prompt(self, target_lang):
        self.prompts.append(target_lang)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.did_eval = True
        return self

    def transcribe(self, audio_paths):
        self.transcribe_calls.append(audio_paths)
        return [SimpleNamespace(text="xin chao")]

    def conformer_stream_step(self, **kwargs):
        self.stream_step_calls.append(kwargs)
        index = min(len(self.stream_step_calls) - 1, len(self.stream_texts) - 1)
        return (
            "pred",
            [SimpleNamespace(text=self.stream_texts[index])],
            kwargs["cache_last_channel"],
            kwargs["cache_last_time"],
            kwargs["cache_last_channel_len"],
            "hyp",
        )


class FakeASRModel:
    restored = []
    model = None

    @classmethod
    def restore_from(cls, restore_path):
        cls.restored.append(restore_path)
        cls.model = FakeModel()
        return cls.model

    @classmethod
    def from_pretrained(cls, model_name):
        cls.model = FakeModel()
        cls.model.pretrained_name = model_name
        return cls.model


class FakeTensor:
    def to(self, _dtype):
        return self


class FakeStreamingBuffer:
    empty_final_created = False

    def __init__(self, model, online_normalization=False, pad_and_drop_preencoded=False):
        self.audio = None

    def append_audio(self, audio, stream_id=-1):
        self.audio = audio

    def __iter__(self):
        if self.audio is None or len(self.audio) == 0:
            FakeStreamingBuffer.empty_final_created = True
            return iter([])
        return iter([(FakeTensor(), 1)])

    def is_buffer_empty(self):
        return True


def make_backend(monkeypatch, backend_extra_args=None):
    FakeASRModel.restored = []
    FakeStreamingBuffer.empty_final_created = False
    monkeypatch.setattr(
        nemo_module,
        "nemo_asr",
        SimpleNamespace(models=SimpleNamespace(ASRModel=FakeASRModel)),
    )
    monkeypatch.setattr(nemo_module, "CacheAwareStreamingAudioBuffer", FakeStreamingBuffer)
    return nemo_module.ASRNeMo(
        ModelSpec(
            model_name="nemotron",
            model_id="/models/nemotron.nemo",
            backend="nemo_asr",
            backend_extra_args=backend_extra_args or {"device": "cpu", "att_context_size": [56, 0]},
        )
    )


def test_model_spec_maps_nemo_asr_to_stt():
    assert ModelSpec.get_model_type_from_backend("nemo_asr") == "stt"
    spec = ModelSpec.from_dict(
        {
            "model_name": "nemotron",
            "model_id": "/models/nemotron.nemo",
            "backend": "nemo_asr",
        }
    )
    assert spec.model_type == "stt"


def test_nemo_backend_loads_local_checkpoint_and_sets_language_prompt(monkeypatch):
    backend = make_backend(
        monkeypatch,
        {"device": "cpu", "target_lang": "vi", "att_context_size": [56, 3]},
    )

    assert FakeASRModel.restored == ["/models/nemotron.nemo"]
    assert backend.model.prompts[-1] == "vi-VN"
    assert backend.model.encoder.att_context_size == [56, 3]
    assert backend.model.decoding.strip_lang_tags is True


def test_nemo_backend_transcribes_numpy_audio(monkeypatch):
    backend = make_backend(monkeypatch)
    response = backend.transcribe(np.zeros(1600, dtype=np.float32), language="vi")

    assert response.text == "xin chao"
    assert backend.model.prompts[-1] == "vi-VN"
    assert len(backend.model.transcribe_calls) == 1


def test_asr_processor_exposes_native_streaming_session(monkeypatch):
    backend = make_backend(monkeypatch)
    processor = ASRProcessor(asr=backend, vad_config=TurnDetectionConfig(create_response=False))

    assert processor.supports_native_streaming() is True
    session = processor.create_native_streaming_session(language="vi")

    first_delta = session.accept_audio(np.ones(session.chunk_samples, dtype=np.float32))
    second_delta = session.accept_audio(np.ones(session.chunk_samples, dtype=np.float32))

    assert first_delta == "hello"
    assert second_delta == " world"
    assert session.accept_audio(np.zeros(0, dtype=np.float32), is_final=True) == ""
    assert FakeStreamingBuffer.empty_final_created is False
    assert backend.model.prompts[-1] == "vi-VN"


def test_asr_processor_disables_vad_when_optional_dependency_is_missing(monkeypatch):
    class MissingDependencyVAD:
        def __init__(self, _vad_config):
            raise ModuleNotFoundError("No module named 'torchaudio'")

    monkeypatch.setattr(
        "gallama.backend.stt.asr_processor.VADProcessor",
        MissingDependencyVAD,
    )

    backend = make_backend(monkeypatch)
    processor = ASRProcessor(asr=backend, vad_config=TurnDetectionConfig(create_response=True))

    assert processor.vad_enable is False
    assert processor.vad is None
    assert processor.supports_native_streaming() is True
