import asyncio
import json
import os
import sys
import types

import pytest
from pydantic import ValidationError


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def device_count():
            return 0

    torch_stub.cuda = _Cuda()
    sys.modules["torch"] = torch_stub

if "colorama" not in sys.modules:
    colorama_stub = types.ModuleType("colorama")
    colorama_stub.Fore = types.SimpleNamespace(CYAN="", GREEN="", YELLOW="", RED="", BLUE="")
    colorama_stub.Back = types.SimpleNamespace(WHITE="")
    colorama_stub.Style = types.SimpleNamespace(RESET_ALL="")
    colorama_stub.init = lambda autoreset=True: None
    sys.modules["colorama"] = colorama_stub

if "zmq" not in sys.modules:
    zmq_stub = types.ModuleType("zmq")

    class _Again(Exception):
        pass

    class _ZMQError(Exception):
        pass

    class _Socket:
        def setsockopt(self, *args, **kwargs):
            return None

        def connect(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def send_json(self, *args, **kwargs):
            return None

    class _Context:
        def socket(self, *args, **kwargs):
            return _Socket()

        def term(self):
            return None

    zmq_stub.Context = _Context
    zmq_stub.PUSH = 0
    zmq_stub.LINGER = 0
    zmq_stub.NOBLOCK = 0
    zmq_stub.Again = _Again
    zmq_stub.error = types.SimpleNamespace(ZMQError=_ZMQError)
    sys.modules["zmq"] = zmq_stub

if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class _HTTPException(Exception):
        def __init__(self, status_code=None, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _Request:
        pass

    fastapi_stub.FastAPI = object
    fastapi_stub.HTTPException = _HTTPException
    fastapi_stub.Query = lambda default=None, **kwargs: default
    fastapi_stub.Request = _Request
    sys.modules["fastapi"] = fastapi_stub

if "fastapi.responses" not in sys.modules:
    fastapi_responses_stub = types.ModuleType("fastapi.responses")

    class _Response:
        def __init__(self, content=b"", status_code=200, headers=None, media_type=None):
            self.content = content
            self.status_code = status_code
            self.headers = headers or {}
            self.media_type = media_type

    class _StreamingResponse(_Response):
        pass

    fastapi_responses_stub.Response = _Response
    fastapi_responses_stub.StreamingResponse = _StreamingResponse
    sys.modules["fastapi.responses"] = fastapi_responses_stub


from gallama.api_response.api_formatter import ResponsesFormatter
from gallama.api_response.chat_response import chat_completion_response_stream
from gallama.conversation_store import ConversationStore
from gallama.data_classes import (
    ChatMLQuery,
    ConversationCreateRequest,
    ResponseInputItem,
    ResponsesCreateRequest,
    TagDefinition,
)
from gallama.data_classes.generation_data_class import GenEnd, GenQueueDynamic, GenStart, GenText


def test_responses_request_rejects_previous_response_id_with_conversation():
    with pytest.raises(ValidationError):
        ResponsesCreateRequest(
            model="qwen3",
            input="hello",
            previous_response_id="resp_prev",
            conversation="conv_123",
        )


def test_responses_formatter_includes_conversation_object():
    request_model = ResponsesCreateRequest(
        model="qwen3",
        input="hello",
        conversation="conv_123",
    )

    formatter = ResponsesFormatter(
        model_name="qwen3",
        request_model=request_model,
        unique_id="resp_123",
    )

    response = formatter.final_stream_response()

    assert response.id == "resp_123"
    assert response.conversation is not None
    assert response.conversation.id == "conv_123"


def test_responses_formatter_emits_live_reasoning_events():
    request_model = ResponsesCreateRequest(
        model="qwen3",
        input="hello",
        stream=True,
    )

    formatter = ResponsesFormatter(
        model_name="qwen3",
        request_model=request_model,
        unique_id="resp_123",
    )

    assert formatter.open_block("reasoning") == []

    first_events = formatter.stream_chunk(api_tag="reasoning", text="First", role="assistant")
    second_events = formatter.stream_chunk(api_tag="reasoning", text=" second", role="assistant")
    done_events = formatter.close_block()

    first_payloads = [json.loads(event["data"]) for event in first_events]
    second_payloads = [json.loads(event["data"]) for event in second_events]
    done_payloads = [json.loads(event["data"]) for event in done_events]

    assert [payload["type"] for payload in first_payloads] == [
        "response.output_item.added",
        "response.reasoning_text.delta",
    ]
    assert first_payloads[1]["delta"] == "First"
    assert [payload["type"] for payload in second_payloads] == ["response.reasoning_text.delta"]
    assert second_payloads[0]["delta"] == " second"
    assert [payload["type"] for payload in done_payloads] == [
        "response.reasoning_text.done",
        "response.output_item.done",
    ]
    assert done_payloads[0]["text"] == "First second"

    response = formatter.final_stream_response()

    assert response.output[0].type == "reasoning"
    assert response.output[0].content[0].text == "First second"


def test_responses_stream_emits_reasoning_text_deltas():
    async def scenario():
        reasoning_tag = TagDefinition(
            start_marker="<think>",
            end_marker="</think>",
            tag_type="thinking",
            api_tag="reasoning",
            role="assistant",
        )
        gen_queue = GenQueueDynamic()
        gen_queue.put_nowait(GenStart(gen_type=reasoning_tag))

        request_model = ResponsesCreateRequest.model_validate(
            {
                "model": "test-model",
                "input": "hello",
                "stream": True,
            }
        )

        query = ChatMLQuery.model_validate(
            {
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            }
        )

        async def producer():
            await asyncio.sleep(0.01)
            gen_queue.put_nowait(GenText(content="<think>First"))
            await asyncio.sleep(0.02)
            gen_queue.put_nowait(GenText(content=" second"))
            await asyncio.sleep(0.02)
            gen_queue.put_nowait(GenText(content="</think>"))
            await asyncio.sleep(0.02)
            gen_queue.put_nowait(GenEnd())

        producer_task = asyncio.create_task(producer())

        events = []
        async for event in chat_completion_response_stream(
            query=query,
            gen_queue=gen_queue,
            model_name="test-model",
            request=None,
            provider="responses",
            tag_definitions=[reasoning_tag],
            formatter_kwargs={"request_model": request_model},
        ):
            events.append({"event": event.get("event"), "data": json.loads(event["data"])})

        await producer_task

        reasoning_deltas = [
            payload["data"]["delta"]
            for payload in events
            if payload["event"] == "response.reasoning_text.delta"
        ]
        reasoning_done = next(
            payload["data"]
            for payload in events
            if payload["event"] == "response.reasoning_text.done"
        )
        completed_response = next(
            payload["data"]["response"]
            for payload in events
            if payload["event"] == "response.completed"
        )

        assert "".join(reasoning_deltas) == "First second"
        assert reasoning_done["text"] == "First second"
        assert [item["type"] for item in completed_response["output"]] == ["reasoning"]
        assert completed_response["output"][0]["content"][0]["text"] == "First second"

    asyncio.run(scenario())


def test_conversation_create_request_converts_initial_items():
    request = ConversationCreateRequest(
        items=[
            "hello",
            ResponseInputItem(role="assistant", content="hi there"),
        ]
    )

    messages = request.to_messages()

    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == ["hello", "hi there"]


def test_responses_request_normalizes_instruction_messages_to_front():
    request = ResponsesCreateRequest(
        model="qwen3",
        input=[
            ResponseInputItem(
                role="assistant",
                content=[{"type": "output_text", "text": "Previous assistant reply."}],
            ),
            ResponseInputItem(
                role="developer",
                content=[{"type": "input_text", "text": "Follow repository conventions."}],
            ),
            ResponseInputItem(
                role="user",
                content=[{"type": "input_text", "text": "Make the next edit."}],
            ),
            ResponseInputItem(
                role="system",
                content=[{"type": "input_text", "text": "Be concise."}],
            ),
        ],
    )

    messages = request.to_input_messages()

    assert [message.role for message in messages] == ["system", "assistant", "user"]
    assert messages[0].content == "Follow repository conventions.\n\nBe concise."


def test_responses_request_ignores_item_reference_inputs():
    request = ResponsesCreateRequest(
        model="qwen3",
        input=[
            ResponseInputItem(type="item_reference", id="item_123"),
            ResponseInputItem(
                role="user",
                content=[{"type": "input_text", "text": "Hello"}],
            ),
        ],
    )

    messages = request.to_input_messages()

    assert [message.role for message in messages] == ["user"]
    assert messages[0].content == "Hello"


def test_responses_request_can_insert_codex_user_fallback():
    request = ResponsesCreateRequest(
        model="qwen3",
        input=[
            ResponseInputItem(
                role="assistant",
                content=[{"type": "output_text", "text": "Tool call pending."}],
            ),
            ResponseInputItem(
                type="function_call_output",
                call_id="call_123",
                output={"status": "ok"},
            ),
        ],
    )

    messages = request.to_input_messages(ensure_user=True)

    assert [message.role for message in messages] == ["assistant", "tool", "user"]
    assert messages[-1].content == "Please continue based on the conversation above."


def test_conversation_store_crud_round_trip():
    async def scenario():
        store = ConversationStore()

        created = await store.create(metadata={"topic": "demo"})
        assert created.id.startswith("conv_")

        fetched = await store.get(created.id)
        assert fetched is not None
        assert fetched.to_resource().metadata == {"topic": "demo"}

        updated = await store.update_metadata(created.id, {"topic": "updated"})
        assert updated is not None
        assert updated.to_resource().metadata == {"topic": "updated"}

        deleted = await store.delete(created.id)
        assert deleted is not None
        assert deleted.id == created.id
        assert await store.get(created.id) is None

    asyncio.run(scenario())
