from typing import Any, Dict

from ..llamacpp_server import ModelLlamaCppServer


class ModelIKLlama(ModelLlamaCppServer):

    def load_model(self):
        self.backend_extra_args = self.apply_backend_defaults(self.backend_extra_args)
        return super().load_model()
