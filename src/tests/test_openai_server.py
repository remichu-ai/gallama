# TODO
# import pytest
# from fastapi.testclient import TestClient
# from gallama.server import run_from_script, manager_app
#
# class TestArgs:
#     def __init__(self):
#         self.model_id = [{"model_id": "mistral"}]
#         self.detached = True
#         self.host = "127.0.0.1"
#         self.port = 8000
#         self.draft_model_id = "mistral"
#         self.verbose = False
#         self.strict_mode = False
#
# args = TestArgs()
#
# app = run_from_script(args)     # start an api service
#
# client = TestClient(manager_app)
#
# def test_read_root():
#     response = client.get("/")
#     assert response.status_code == 200
#     assert response.json() == {"Hello": "World"}
#
# def test_chat_completion():
#     payload = {
#         "model": "Mixtral-8x7B",
#         "messages": [{"role": "user", "content": "Hello, how are you?"}],
#         "temperature": 0.7,
#         "stream": False
#     }
#     response = client.post("/v1/chat/completions", json=payload)
#     assert response.status_code == 200
#     assert "choices" in response.json()
#     assert len(response.json()["choices"]) > 0
#     assert "message" in response.json()["choices"][0]
#
# def test_generate():
#     payload = {
#         "prompt": "Once upon a time",
#         "model": "Mixtral-8x7B",
#         "stream": False,
#         "max_tokens": 50
#     }
#     response = client.post("/v1/completions", json=payload)
#     assert response.status_code == 200
#     assert "choices" in response.json()
#     assert len(response.json()["choices"]) > 0
#     assert "text" in response.json()["choices"][0]
#
# def test_embeddings():
#     payload = {
#         "input": "The quick brown fox jumps over the lazy dog",
#         "model": "Mixtral-8x7B"
#     }
#     response = client.post("/v1/embeddings", json=payload)
#     assert response.status_code == 200
#     assert "data" in response.json()
#     assert len(response.json()["data"]) > 0
#     assert "embedding" in response.json()["data"][0]
#
# def test_get_models():
#     response = client.get("/v1/models")
#     assert response.status_code == 200
#     assert "data" in response.json()
#     assert len(response.json()["data"]) > 0
#     assert "id" in response.json()["data"][0]
#
# def test_load_model():
#     payload = {
#         "model_id": "Mixtral-8x7B",
#         "gpus": [1.0],
#         "cache_size": 2048
#     }
#     response = client.post("/load_model", json=payload)
#     assert response.status_code == 200
#
# def test_delete_model():
#     response = client.post("/delete_model", json={"model_name": "Mixtral-8x7B"})
#     assert response.status_code == 200
#
# def test_get_status():
#     response = client.get("/status")
#     assert response.status_code == 200
#     assert "status" in response.json()
#
# def test_health_check():
#     response = client.get("/health")
#     assert response.status_code == 200
#     assert response.json() == {"status": "healthy"}