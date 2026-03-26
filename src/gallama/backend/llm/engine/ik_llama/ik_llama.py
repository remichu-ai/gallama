from typing import Any, Dict

from ..llamacpp_server import ModelLlamaCppServer


class ModelIKLlama(ModelLlamaCppServer):
    DEFAULT_MULTIMODAL_MARKER = "<__media__>"

    @classmethod
    def apply_backend_defaults(cls, backend_extra_args: Dict[Any, Any] | None) -> Dict[Any, Any]:
        resolved = dict(backend_extra_args or {})
        resolved.setdefault("multimodal_marker", cls.DEFAULT_MULTIMODAL_MARKER)
        return resolved

    def load_model(self):
        self.backend_extra_args = self.apply_backend_defaults(self.backend_extra_args)
        return super().load_model()
