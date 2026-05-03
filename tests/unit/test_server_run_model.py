import asyncio

from gallama.data_classes.data_class import ModelSpec
from gallama import server


class _CaptureLogger:
    def __init__(self):
        self.exceptions = []
        self.infos = []

    def exception(self, message, *args, **kwargs):
        self.exceptions.append(message)

    def info(self, message, *args, **kwargs):
        self.infos.append(message)


def test_run_model_handles_error_before_port_assignment(monkeypatch):
    logger = _CaptureLogger()
    stopped_ports = []
    cleaned_models = []

    def raise_config_error(model_name):
        raise RuntimeError("config boom")

    async def stop_model_instance(model_name, port):
        stopped_ports.append(port)

    async def cleanup_after_model_load(model_name):
        cleaned_models.append(model_name)

    monkeypatch.setattr(server, "server_logger", logger)
    monkeypatch.setattr(server.config_manager, "get_effective_model_config", raise_config_error)
    monkeypatch.setattr(server, "stop_model_instance", stop_model_instance)
    monkeypatch.setattr(server, "cleanup_after_model_load", cleanup_after_model_load)

    asyncio.run(server.run_model(ModelSpec(model_name="broken-model")))

    assert stopped_ports == []
    assert cleaned_models == ["broken-model"]
    assert len(logger.exceptions) == 1
    assert "config boom" in logger.exceptions[0]
    assert "on port" not in logger.exceptions[0]
    assert logger.infos[-1] == "Exiting run_model for broken-model"
