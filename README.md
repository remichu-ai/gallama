# gallama - Guided Agentic Llama

__gallama__ is an opinionated Python library that provides a LLM inference API service backend optimized for local agentic tasks.
It focuses on model serving, realtime, multimodal, and local inference integrations rather than multi-agent orchestration.

Gallama is predominantly tested with the Exllama V3 workflow at this point. Other backends are still available, but they may have bugs or rough edges depending on the model and feature path.

Currently, the backend is mainly using Exllama-family backends. Llama.cpp support is under experiment. 

Do checkout [TabbyAPI](https://github.com/theroyallab/tabbyAPI) if you want a reliable and pure ExllamaV3 API backend.

# Key Feature:
- Native Tool Calling
- OpenAI chat completion API
- Anthropic message API
- Compatible with Claude Code
- ExLlamaV3 speculative decoding, including DFlash draft models with `exllamav3>=0.0.31`

## Native Tool Calling
Gallama supports native tool calling. Instead of forcing every model into one synthetic format, Gallama uses the model's own tool-calling format when that format is supported by a parser in [`src/gallama/backend/llm/prompt_engine/by_model`](./src/gallama/backend/llm/prompt_engine/by_model).

Current models with custom native tool parsers:

- `Qwen JSON family`
  Covers `qwen2`, `qwen2_5_vl`, `qwen3`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `qwen3_vl_moe`
- `Qwen XML family`
  Covers `qwen3_5`, `qwen3_5_moe`, `step3p5`, `nemotron_h`
- `GPT-OSS Harmony family`
  Covers `gpt_oss`
- `GLM-4 family`
  Covers `glm4`, `glm4_moe`, `glm4v`, `glm4v_moe`
- `MiniMax family`
  Covers `minimax`, `minimax_m2`
- `Ministral / Mistral 3 family`
  Covers `ministral3`, `mistral3` (including Devstral-style `mistral3` models)

For these models, Gallama expects the model to emit its native tool-call structure, and Gallama parses that structure back into OpenAI-compatible `tool_calls` or Anthropic-compatible `tool_use` blocks.

If you want to use a new model with a different native tool-calling format, Gallama will usually need a new parser added under [`src/gallama/backend/llm/prompt_engine/by_model`](./src/gallama/backend/llm/prompt_engine/by_model) so the backend can interpret that model correctly. Without a matching parser, tool calling may fail or be decoded incorrectly even if the model itself knows how to call tools.

## Reasoning Output
Gallama also returns model reasoning when the model emits it.

With the OpenAI-compatible API, reasoning is returned on the assistant message as `reasoning` in the raw response payload:

```python
completion = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Solve 27 * 43. Give only the answer."}],
)

message = completion.choices[0].message

print(message.content)

# Depending on the SDK version, custom fields may be available either directly
# or through a raw/model-extra view of the response object.
print(getattr(message, "reasoning", None))
print(getattr(message, "model_extra", {}).get("reasoning") if getattr(message, "model_extra", None) else None)
```

With the Anthropic-compatible API, reasoning is returned as `thinking` blocks inside `response.content`:

```python
response = client.messages.create(
    model="qwen3",
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 1024},
    messages=[{"role": "user", "content": "Solve 27 * 43. Give only the answer."}],
)

thinking_blocks = [block for block in response.content if block.type == "thinking"]
reasoning_text = "\n".join(block.thinking for block in thinking_blocks)

print(reasoning_text)
```

This makes it possible to inspect the model's intermediate reasoning while still using standard OpenAI or Anthropic client libraries against Gallama.


# Quick Start
   Head down to the installation guide at the bottom of this page.
   Then check out the [Examples_Notebook.ipynb](https://github.com/remichu-ai/gallama/blob/main/examples/Examples_Notebook.ipynb) in the examples folder
   A simple python streamlit frontend chat UI code is included in the examples folder [streamlit](https://github.com/remichu-ai/gallama/blob/main/examples/streamlit_chatbot.py)
   Or checkout [GallamaUI](https://github.com/remichu-ai/gallamaUI.git)
   You can also refer to `src/tests` folder for more example using OpenAI and Anthropic client.

# Features


## OpenAI Compatible Server
Fully compatible with the OpenAI client.

Install openai client and overwrite its base setting as follow:

```shell
pip install openai
```

```python
import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = 'test'
client = OpenAI(base_url='http://127.0.0.1:8000/v1')
```

```python
messages = [{"role": "user", "content": "Which is faster in terms of reaction speed: a cat or a dog?"}]

completion = client.chat.completions.create(
    model="mistral",
    messages=messages,
    tool_choice="auto"
)

print(completion)
```

See [`tests/live/test_openai.py`](tests/live/test_openai.py) and [`src/tests/test_openai_server.py`](./src/tests/test_openai_server.py) for more complete examples.

## Anthropic Compatible Server
Gallama also exposes an Anthropic-compatible Messages endpoint.

Install the Anthropic SDK and point it at your local server:

```shell
pip install anthropic
```

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:8000",
    api_key="test",
)
```

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Which is faster in terms of reaction speed: a cat or a dog?"}
    ],
)

print(response.content)
```

See [`tests/live/test_anthropic.py`](tests/live/test_anthropic.py) for a more complete Anthropic client example suite.

### Claude Code
You can also point Claude Code at a local Gallama server by overriding the Anthropic base URL and auth token:

```shell
ANTHROPIC_BASE_URL="http://127.0.0.1:8000/" ANTHROPIC_AUTH_TOKEN="local" claude --model minimax
```

This lets Claude Code talk to your local model through Gallama's Anthropic-compatible API.

### Claude Code With Local MCP Web Search
If you want Claude Code to use a local MCP server for web search instead of Claude Code's built-in `WebSearch`, you can run the MCP server included in this repo and wrap `claude` with a local helper command.

1. Create a local env file:

```shell
cp examples/mcp/.env_sample examples/mcp/.env
```

2. Fill in whichever search providers you want to use:

- `EXA_API_KEY`
- `TAVILY_API_KEY`
- `BRAVE_API_KEY`

3. Set `LOCAL_VISION_MODEL` if you also want the `understand_image` MCP tool.

4. Start the MCP server from the repo root:

```shell
python examples/mcp/server.py
```

The MCP server exposes:

- `web_search`
  Uses one unified wrapper over Exa, Tavily, and Brave. In `provider="auto"` mode it rotates across configured providers and returns the `provider` used in the response.
- `understand_image`
  Sends image understanding requests to your local OpenAI-compatible vision model.

To run Claude Code against your local Gallama server and inject this MCP server automatically, add a wrapper like this to `~/.zshrc`:

```shell
claudel() {
    local model="${CLAUDEL_MODEL:-minimax}"
    local base_url="${CLAUDEL_BASE_URL:-http://127.0.0.1:8000/}"
    local auth_token="${CLAUDEL_AUTH_TOKEN:-local}"
    local mcp_url="${CLAUDEL_MCP_URL:-http://127.0.0.1:18011/mcp}"
    local enable_mcp="${CLAUDEL_ENABLE_MCP:-1}"
    local search_tool="${CLAUDEL_MCP_SEARCH_TOOL:-mcp__local-coding-plan__web_search}"
    local args=(--model "$model")

    if [[ "$enable_mcp" != "0" ]]; then
        args+=(--mcp-config "{\"mcpServers\":{\"local-coding-plan\":{\"type\":\"http\",\"url\":\"$mcp_url\"}}}")
        args+=(
            --disallowedTools "WebSearch"
            --append-system-prompt "For internet search, use the MCP tool ${search_tool}. Do not use the built-in WebSearch tool."
        )
    fi

    ANTHROPIC_BASE_URL="$base_url" \
    ANTHROPIC_AUTH_TOKEN="$auth_token" \
    command claude "${args[@]}" "$@"
}
```

Reload your shell:

```shell
source ~/.zshrc
```

Then run:

```shell
claudel
```

Useful wrapper overrides:

- `CLAUDEL_MODEL=qwen2.5-vl-instruct claudel`
- `CLAUDEL_ENABLE_MCP=0 claudel`
- `CLAUDEL_MCP_URL=http://127.0.0.1:18011/mcp claudel`

Notes:

- The wrapper leaves your normal `claude` command untouched.
- `examples/mcp/.env` is ignored by git, so your local API keys stay out of the repo.
- The MCP server stores monthly provider usage in `examples/mcp/.search_provider_usage.json`, which is also ignored by git.

## MCP
Gallama can discover and execute tools from a remote streamable HTTP MCP server on the server side. The request shape depends on which client surface you use:

- OpenAI Chat Completions: add a tool with `"type": "mcp"`
- OpenAI Responses: add a tool with `"type": "mcp"`
- Anthropic Messages: define `mcp_servers` and reference them with a `"type": "mcp_toolset"` entry in `tools`

Current limitations:

- MCP works for both non-streaming and streaming requests
- `require_approval` is only supported as `"never"` right now
- Mixing MCP tool calls and normal function tool calls in the same model turn is not supported yet

When you use MCP with streaming on the Responses API, Gallama emits MCP trace items in the stream as `response.output_item.added` / `response.output_item.done` events with `mcp_list_tools` and `mcp_call` items before the final assistant text. Streaming Chat Completions still suppresses the intermediate MCP tool-call turns and only streams the final assistant output.

### OpenAI Chat Completions
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="test",
)

completion = client.chat.completions.create(
    model="qwen3",
    max_tokens=3000,
    messages=[
        {
            "role": "user",
            "content": "Use the MCP weather tool and tell me the result.",
        }
    ],
    tools=[
        {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "http://127.0.0.1:18001/mcp",
            "allowed_tools": ["get_weather"],
            "require_approval": "never",
        }
    ],
)

print(completion.choices[0].message.content)
```

### OpenAI Responses API
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="test",
)

response = client.responses.create(
    model="qwen3",
    input="Use the MCP weather tool and tell me the result.",
    max_output_tokens=300,
    tools=[
        {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "http://127.0.0.1:18001/mcp",
            "allowed_tools": ["get_weather"],
            "require_approval": "never",
        }
    ],
)

print(response.output_text)
```

Gallama also prepends MCP trace items to the Responses output, so you will see `mcp_list_tools` and `mcp_call` entries alongside the assistant output.

Streaming also works on the Responses API:

```python
stream = client.responses.create(
    model="qwen3",
    input="Use the MCP weather tool and stream the final answer.",
    max_output_tokens=300,
    stream=True,
    tools=[
        {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "http://127.0.0.1:18001/mcp",
            "allowed_tools": ["get_weather"],
            "require_approval": "never",
        }
    ],
)

for event in stream:
    if event.type == "response.output_item.added" and event.item.type in {"mcp_list_tools", "mcp_call"}:
        print(event.item)
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

If you set `store=True`, you can later retrieve the full Responses object, including the MCP trace items. This is useful when you want both streamed text for the live client and the full MCP conversation history afterward.

```python
created = client.responses.create(
    model="qwen3",
    input="Use the MCP weather tool and stream the final answer.",
    max_output_tokens=300,
    stream=True,
    store=True,
    tools=[
        {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "http://127.0.0.1:18001/mcp",
            "allowed_tools": ["get_weather"],
            "require_approval": "never",
        }
    ],
)

response_id = None
for event in created:
    if event.type in {"response.created", "response.completed"}:
        response_id = event.response.id

stored = client.responses.retrieve(response_id)
for item in stored.output:
    if item.type in {"mcp_list_tools", "mcp_call"}:
        print(item)
```

If you continue a stored Responses conversation with `previous_response_id`, the saved history includes those MCP trace items as part of the recorded response. That means you can inspect prior MCP tool calls later, not just the final assistant text.

### Anthropic Messages API
Gallama accepts an Anthropic-compatible MCP request shape on `/v1/messages`, but this is not a byte-for-byte mirror of Anthropic's current hosted MCP connector beta. In Anthropic's official API, MCP is documented separately under the MCP connector docs and requires a beta header. Gallama's local compatibility layer does not require that beta header.

```python
import json
import urllib.request

payload = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 3000,
    "messages": [
        {
            "role": "user",
            "content": "Use the MCP weather tool and tell me the result.",
        }
    ],
    "mcp_servers": [
        {
            "type": "url",
            "name": "weather",
            "url": "http://127.0.0.1:18001/mcp",
        }
    ],
    "tools": [
        {
            "type": "mcp_toolset",
            "mcp_server_name": "weather",
            "allowed_tools": ["get_weather"],
        }
    ],
}

request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/messages",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "content-type": "application/json",
        "x-api-key": "test",
        "anthropic-version": "2023-06-01",
    },
    method="POST",
)

with urllib.request.urlopen(request) as response:
    data = json.loads(response.read().decode("utf-8"))

for block in data["content"]:
    print(block)
```

When using the Anthropic-compatible endpoint, Gallama returns MCP activity as `mcp_tool_use` and `mcp_tool_result` blocks before the normal text block.

If your MCP server requires auth, include `authorization_token` or `headers` on the MCP server/tool definition.

If you are targeting Anthropic's hosted API instead of Gallama, use Anthropic's MCP connector docs and beta versioning instead of this local Gallama example.

See [`tests/live/test_openai.py`](tests/live/test_openai.py), [`tests/live/test_anthropic.py`](tests/live/test_anthropic.py), and [`tests/live/test_responses.py`](tests/live/test_responses.py) for live end-to-end MCP examples against a dummy MCP server.

## Function Calling
Supports function calling for all models, mimicking OpenAI's behavior for tool_choice="auto" where if tool usage is not applicable, model will generate normal response.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    }
]

messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

completion = client.chat.completions.create(
    model="mistral",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print(completion.choices[0].message.tool_calls[0].function)
```

## Multiple Concurrent Models
Run multiple models (different or same) with automatic load balancing and request routing. 
Model VRAM usage can be auto_loaded or with specific GPUs spliting.
Each model will be run as a dedicated FastAPI to ensure no threading issue and guarantee speed.
However, do note that this will be more demanding on the system as there will be multiple FastAPI running

Basic
```shell
gallama run -id "model_id=llama-3.1-8B" -id "model_id=mistral"
```

Customize GPUs split
```shell
gallama run -id "model_id=qwen2-72B gpus=20,15,15,0" -id "model_id=Llama3.1-8B gpus=0,0,0,20"
```

## OpenAI Embedding Endpoint
Utilize Infinity Embedding library for both embedding via OpenAI client.

```python
response = client.embeddings.create(
    input="Your text string for embedding goes here",
    model="Alibaba-NLP/gte-large-en-v1.5"
)

print(response.data[0].embedding)
```

## Legacy OpenAI Completion Endpoint
Support for the Legacy Completion Endpoint.

```python
client.completions.create(
    model="mistral",
    prompt="Tell me a story about a Llama in 200 words",
    max_tokens=1000,
    temperature=0
)
```

## Format Enforcement
Ensure output conforms to specified patterns with a following options that can be specified in the `extra_body` when using OpenAI client.

```python
completion = client.chat.completions.create(
    model="mistral",
    messages=[{"role": "user", "content": "Is smoking bad for health? Answer with Yes or No"}],
    temperature=0.1,
    max_tokens=200,
    extra_body={
        # "leading_prompt": leading_prompt,         # prefix the generation with some string
        # "regex_pattern": regex_pattern,           # define the regex for the whole generation
        # "regex_prefix_pattern": '(Yes|No)\.',     # define the regex to match the starting words
        # "stop_words": stop_words,                 # define the word to stop generation
    },
)
```

## Streaming
Streaming is fully supported.

```python
messages = [{"role": "user", "content": "Tell me a 200-word story about a Llama"}]

completion = client.chat.completions.create(
    model="mistral",
    messages=messages,
    stream=True,
    temperature=0.1,
)

for chunk in completion:
    print(chunk.choices[0].delta.content, end='')
```

## Remote Model Management
Load and unload models via API calls.

start gallama server if it is not current running:

```shell
gallama run
```

```python
import requests

api_url = "http://127.0.0.1:8000/add_model"

payload = [
    {
        "model_id": "qwen-2-72B",
        "gpus": [22,22,4,0],
        "cache_size": 32768,
    },
    {
        "model_id": "gemma-2-9b",
        "gpus": [0,0,0,12],
        "cache_size": 32768,
    },
    {
        "model_id": "multilingual-e5-large-instruct",
        "gpus": [0,0,0,5],
    },
]

response = requests.post(api_url, json=payload)
```



## Installation

gallama requires certain components to be installed and functioning. 

Ensure that you have a working backend installed before using Gallama. In practice, Exllama V3 is the backend I test against most often. Other backends may still work, but they may need extra debugging depending on the model and feature set.

OS level package required as followed:
For Speech to Text, You will need to install the dependency as required by faster whisper
Most notably is CuDNN
https://developer.nvidia.com/cudnn

For Text to Speech, install the following package:
```shell
apt-get install portaudio19-dev ffmpeg
```

Now to install gallama from pip
```shell
pip install gallama
```

Optional extras are available if you only want specific components:
```shell
pip install "gallama[all]"
pip install "gallama[exl2]"
pip install "gallama[exl3]"
pip install "gallama[llama-cpp]"
pip install "gallama[transformers-backend]"
pip install "gallama[utils]"
pip install "gallama[embedding]"
pip install "gallama[stt]"
pip install "gallama[tts]"
pip install "gallama[video]"
pip install "gallama[vllm]"
pip install "gallama[sglang]"
```

Extras can be combined as needed:
```shell
pip install "gallama[exl3,tts]"
pip install "gallama[llama-cpp,stt]"
```

For newer model support, the latest `transformers` release is often needed. If a model is not loading correctly, update it with:
```shell
pip install -U transformers
```

Or, install from source:

```shell
git clone https://github.com/remichu-ai/gallama.git
cd gallama
pip install .
```

If you're starting from scratch and don't have these dependencies yet, follow these steps:

1. Create a virtual environment (recommended):
   Recommend to use python 3.12 if you can or minimally 3.11 for future tensor parallel compatibility.
   ```shell
   conda create --name genv python=3.12
   conda activate genv
   ```

2. Install and verify your backend:
   - Exllama V3 is the recommended path if you want the setup closest to what is actively tested.
   - DFlash speculative decoding requires `exllamav3>=0.0.31`.
   - Exllama V2, llama.cpp, transformers, vLLM, sglang, and other backends are still available, but expect some backend-specific rough edges.

   For ExLlamaV3 with DFlash support:
   ```shell
   pip install -U "exllamav3>=0.0.31"
   ```

   (Optional) Install llama cpp-python:
   - Follow instructions at [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
   - Test with examples from [llama-cpp-python Examples](https://github.com/abetlen/llama-cpp-python/blob/main/examples/high_level_api/high_level_api_streaming.py)

3. (Optional) Install Flash Attention for improved performance:
   - Follow instructions at [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)

4. (Optional) Install Llama.cpp:
   - Follow instructions at [Llama-cpp-python](https://github.com/abetlen/Llama-cpp-python)
   - Note: ExLlamaV2 is currently recommended. Llama.cpp support is under development.

5. Install gallama:
   ```shell
   pip install gallama
   ```
   Or install with only the components you need:
   ```shell
   pip install "gallama[all]"
   pip install "gallama[exl3]"
   pip install "gallama[exl2,stt]"
   ```
   Or, install from source:
   ```shell
   git clone https://github.com/remichu-ai/gallama.git
   cd gallama
   pip install ".[all]"
   ```

## Usage

Follow these steps to use the model.

### Setup

1. Initialize gallama:
   ```shell
   gallama run
   ```
   This creates a `model_config.yaml` file in `~/gallama`.

2. Update `~/gallama/model_config.yaml` with your model configurations.

3. Launch the model:
   Simple method
      ```shell
      gallama run mistral
      ```
   Advanced method
      ```shell
      gallama run -id "model_id=mistral"
      ```

### Environment rollback

If you use a repo-local `.venv`, you can snapshot it and restore it later:

```shell
make env-snapshot
make env-restore
```

The snapshot is written to `.base-env/requirements.txt` with basic interpreter metadata in `.base-env/metadata.txt`. The snapshot command only targets `.venv` and refuses to write an empty lock file.

### model_config.yaml

Each top-level key is the model name that Gallama will expose through the API. The value under that key is the configuration used to load the backend.

Optional `_global` settings apply to every model entry. This is useful for subprocess environment variables such as `CUDA_VISIBLE_DEVICES`.

Minimal Exllama example:

```yaml
_global:
  env:
    CUDA_VISIBLE_DEVICES: "1,0"

mistral:
  backend: exllama
  model_id: /home/your-user/gallama/models/Mistral-7B-instruct-v0.3-4.5bpw-exl2
  prompt_template: Mistral_large
  gpus: auto
```

Typical keys:

- `backend`: backend name such as `exllama`, `exllamav3`, `llama_cpp`, `llama_cpp_server`, `ik_llama`, `transformers`, `embedding`, or `kokoro`
- `model_id`: local path to the model or model directory
- `prompt_template`: prompt formatter to use for the model family
- `gpus`: usually `auto`, but can also be a per-GPU split
- `reserve_vram`: ExLlamaV3 auto-mode reserve in GB per visible GPU. Scalar applies to all visible GPUs; list values follow the final logical CUDA order after `CUDA_VISIBLE_DEVICES`
- `env`: optional environment variables for the model subprocess. Per-model `env` overrides `_global.env`
- `warmup_prompt`: optional ChatML-style mapping for startup warmup. Can also be set under `_global` and overridden per model. Use `false` on a model to disable inherited warmup.
- `max_seq_len`: override context length if needed
- `cache_quant`: KV cache quantization such as `FP16`, `Q4`, `Q6`, or `Q8`
- `quant`: optional metadata for the model quantization you downloaded
- `eos_token_list`: optional extra EOS tokens for models that need them
- `default_sampling`: optional per-model sampling defaults. Rules with `condition: thinking` apply only to the dedicated reasoning pass; omitted `condition` is the normal default. API request values override YAML values per field.
- `backend_extra_args`: backend-specific options, commonly used for `transformers`, `sglang`, `kokoro`, and similar backends
- `draft_model_id`: optional draft model path for speculative decoding. With ExLlamaV3 this can be a normal flash draft model or a DFlash draft model.
- `draft_model_name`: optional name of another `model_config.yaml` entry to use as the draft model.
- `draft_gpus`: optional GPU split for the draft model. If omitted, Gallama uses `auto`.
- `draft_cache_quant`: draft KV cache quantization. Defaults to `FP16`; use `Q4`, `Q6`, or `Q8` only if you intentionally want a quantized draft cache.

Example with default sampling:

```yaml
qwen35:
  backend: transformers
  model_id: /home/your-user/gallama/models/qwen3.5
  prompt_template: Qwen3.5
  gpus: auto
  default_sampling:
    - temperature: 0.7
      top_p: 0.85
      top_k: 20
      min_p: 0.0
      presence_penalty: 0.0
      frequency_penalty: 0.0
      repetition_penalty: 1.0
    - condition: thinking
      temperature: 1.0
      top_p: 0.95
      top_k: 20
      min_p: 0.0
      presence_penalty: 1.5
      repetition_penalty: 1.0
```

Example with global GPU reordering and a per-model override:

```yaml
_global:
  env:
    CUDA_VISIBLE_DEVICES: "1,0"

qwen25-vl:
  backend: exllamav3
  model_id: /home/your-user/gallama/models/qwen25-vl
  prompt_template: Qwen2-VL
  gpus: auto
  reserve_vram: [1.0, 0.0]

text-only-model:
  backend: exllama
  model_id: /home/your-user/gallama/models/text-only
  prompt_template: Llama3
  gpus: auto
  env:
    CUDA_VISIBLE_DEVICES: "0,1"
```

When `gpus: auto` is used, Gallama preserves the configured `CUDA_VISIBLE_DEVICES` order exactly. When `gpus` is an explicit split list, Gallama now interprets that split relative to the configured visible-device order.

Example with a global warmup prompt loaded from an external file:

```yaml
_global:
  env:
    CUDA_VISIBLE_DEVICES: "0,2,3,1,4,5"
  warmup_prompt:
    path: /home/your-user/.config/claude-code/warmup.yaml
    max_completion_tokens: 64
    reasoning_effort: minimal

claude-code-model:
  backend: transformers
  model_id: /home/your-user/gallama/models/claude-code
  warmup_prompt:
    max_completion_tokens: 32

another-model:
  backend: exllama
  model_id: /home/your-user/gallama/models/another-model
  warmup_prompt: false
```

The external file should contain a mapping that Gallama can validate as a `ChatMLQuery`, for example:

```yaml
messages:
  - role: developer
    content: You are Claude Code.
  - role: user
    content: Reply with OK.
```

Example with a `transformers` backend:

```yaml
llama-3.2-Vision-11B_transformers:
  backend: transformers
  model_id: /home/your-user/gallama/models/llama-3.2-Vision-11B-4.0bpw-transformers
  prompt_template: Llama3.2-VL
  gpus: auto
  cache_quant: Q4
  quant: 4.0
  backend_extra_args:
    model_class: transformers.MllamaForConditionalGeneration
    tokenizer_class: transformers.AutoTokenizer
    processor_class: transformers.AutoProcessor
    model_class_extra_kwargs:
      attn_implementation: sdpa
```

Example with a `llama_cpp` backend:

```yaml
codestral_llama_cpp:
  backend: llama_cpp
  model_id: /home/your-user/gallama/models/codestral-4.0bpw-llama_cpp/Codestral-22B-v0.1-Q4_K_M.gguf
  prompt_template: Mistral
  gpus: auto
  cache_quant: Q4
  quant: 4.0
```

Example with a `llama_cpp_server` backend:

```yaml
codestral_llama_cpp_server:
  backend: llama_cpp_server
  model_id: mistralai/Codestral-22B-v0.1
  prompt_template: Mistral
  max_seq_len: 32768
  backend_extra_args:
    base_url: http://127.0.0.1:8080
    cache_prompt: true
    use_server_tokenizer: true
```

This backend keeps prompt templating in Gallama and uses `llama-server` mainly as a generation engine through `/completion` and `/tokenize`.

Start `llama-server` separately, for example:

```shell
llama-server -m /path/to/model.gguf --port 8080 --ctx-size 32768
```

Notes for `llama_cpp_server`:

- `backend_extra_args.base_url` is required.
- `model_id` is still used by Gallama's prompt engine. If `prompt_template` is omitted, `model_id` must be a valid Hugging Face model/tokenizer identifier or a local tokenizer directory so Gallama can load the chat template.
- If you want to avoid Hugging Face tokenizer loading, set an explicit `prompt_template` such as `Mistral`, `Llama3`, or another template from `src/gallama/data/model_token.yaml`.
- Gallama tokenizes prompts through `llama-server` with `add_special=false`, then sends token arrays to `/completion` for text-only requests.
- Image inputs are supported by switching `/completion` into prompt-object mode with `prompt_string + multimodal_data`.
- Audio inputs are not supported yet.
- Direct video input is not sent to `llama-server`, but Gallama can still fall back to converting video frames into images for backends that support images.
- `use_server_tokenizer` must stay `true` in the current implementation.

Example with an `ik_llama` backend:

```yaml
qwen_ik_llama:
  backend: ik_llama
  model_id: /home/your-user/models/qwen.gguf
  prompt_template: Qwen2-VL
  max_seq_len: 32768
  backend_extra_args:
    base_url: http://127.0.0.1:8080
    cache_prompt: true
    use_server_tokenizer: true
```

Notes for `ik_llama`:

- This backend inherits the `llama_cpp_server` integration and uses the same `/completion` and `/tokenize` flow.
- It automatically applies `backend_extra_args.multimodal_marker: "<__media__>"` for multimodal `/completion` requests unless you override it explicitly.
- Use `ik_llama` when the base `llama_cpp_server` backend works for text but `ik_llama.cpp` vision requests require the server-side MTMD marker format.

Notes:

- Use the YAML key itself as the API model name. For example, if the key is `qwen-2.5-32B`, then that is the model string to pass in the client request.
- `prompt_template` matters. If the wrong one is chosen, the model may still load but chat quality or tool use can break.
- `backend_extra_args` is the place for backend-specific tuning such as custom tokenizer/model/processor classes or TTS model paths.
- You can keep your Gallama config in another location by setting `GALLAMA_HOME_PATH`.
   
### Advanced Usage

Using `gallama run -id` followed by a string which is a dictionary of key-value pair will unlock additional option as following:

Customize the model launch using various parameters. Available parameters for the `-id` option include:

- `model_name`: API model name to expose from Gallama. Required when running without a matching `model_config.yaml` entry.
- `model_id`: Model path or Hugging Face repo ID. Required for YAML-free launch and optional when it already exists in `model_config.yaml`.
- `gpus`: VRAM usage for each GPU, comma-separated list of floats (optional)
- `cache_size`: Context length for cache text in integers (optional)
- `cache_quant`: Quantization to use for cache, options are "FP16", "Q4", "Q6", "Q8" (optional)
- `max_seq_len`: Maximum sequence length (optional)
- `backend`: Model engine backend. Options include `exllama`, `exllamav3`, `llama_cpp`, `llama_cpp_server`, `ik_llama`, `transformers`, `vllm`, `sglang`, `mlx_vlm`, `embedding`, `faster_whisper`, `mlx_whisper`, `kokoro`.
- `tp`: enable tensor parallel with exllama v2 (experimental). See further below

#### Run Without `model_config.yaml`

If you fully specify the model on the CLI, Gallama can run it without a matching entry in `~/gallama/model_config.yaml`.

Minimum required arguments for a YAML-free LLM launch:

- `model_name`
- `model_id`
- `backend`

Example:

```shell
gallama run -id "model_name=minimax model_id=/path/to/model backend=exllamav3"
```

To also write the same CLI logs to a file:

```shell
gallama run -id "model_name=minimax model_id=/path/to/model backend=exllamav3" `--log-file ./log/gallama.log`
```

To control log verbosity:

```shell
gallama run -id "model_name=minimax model_id=/path/to/model backend=exllamav3" -v
gallama run -id "model_name=minimax model_id=/path/to/model backend=exllamav3" -vv
```

Useful optional arguments:

- `max_seq_len=32768`
- `gpus=20,20` or leave it as automatic
- default ExLlamaV3 auto reserve is `0.8 GB` on logical GPU 0 and `0.4 GB` on other visible GPUs
- `reserve_vram=0.4` for ExLlamaV3 auto mode on all visible GPUs, or `reserve_vram=1.0,0.0` to reserve only logical GPU 0
- `cache_size=32768`
- `cache_quant=Q4`
- `prompt_template=<template-name>`
- `strict=True`
- `max_concurrent_requests=<n>`
- `--log-file ./log/gallama.log` to mirror CLI logs into a file
- `-v` to enable debug logging while still truncating large base64 image payloads in API request logs
- `-vv` to enable maximum verbosity, including full base64 image payloads in API request logs

Notes:

- If you omit `prompt_template`, Gallama will use the tokenizer's built-in Hugging Face chat template. That is usually fine for modern transformers models, but older or custom models may still need an explicit prompt template.
- `reserve_vram` is interpreted in GB against the final visible-device order after `CUDA_VISIBLE_DEVICES` is applied. For ExLlamaV3, it only applies when `gpus=auto`; explicit `gpus=...` and `reserve_vram` cannot be combined.
- Draft/speculative decoding still expects the draft model to exist in `model_config.yaml` unless you pass a full `draft_model_id` directly.
- ExLlamaV3 DFlash speculative decoding requires `exllamav3>=0.0.31`. Gallama detects DFlash from the draft model and defaults DFlash to `num_draft_tokens=15` unless you override it in `backend_extra_args`.
- This is mainly useful for multimodal requests with large message histories or `data:image/...;base64,...` inputs. At normal verbosity Gallama truncates those image payloads in logs to keep them readable.

#### Speculative Decoding Parameters
- `draft_model_id`: ID of the draft model (optional)
- `draft_model_name`: Name of the draft model (optional)
- `draft_gpus`: VRAM usage for each GPU for the draft model, comma-separated list of floats (optional)
- `draft_cache_size`: Context length for cache text in integers for the draft model (optional; ExLlamaV3 keeps the draft cache size matched to the main cache)
- `draft_cache_quant`: Quantization to use for cache for the draft model, options are `FP16`, `Q4`, `Q6`, `Q8`. Defaults to `FP16`
- `backend_extra_args.num_draft_tokens`: Number of draft tokens. For ExLlamaV3 DFlash, Gallama defaults this to `15`; explicit values override the default

### Examples

1. Launch two models simultaneously:
   ```shell
   gallama run -id "model_name=mistral model_id=/path/to/mistral backend=exllamav3" -id "model_name=llama3 model_id=/path/to/llama3 backend=exllamav3"
   ```

2. Launch a model with specific VRAM limits per GPU:
   ```shell
   gallama run -id "model_name=qwen2-72B model_id=/path/to/qwen2-72B backend=exllamav3 gpus=22,22,10,0"
   ```
   This limits memory usage to 22GB for GPU0 and GPU1, 10GB for GPU2, and 0GB for GPU3.

3. Launch a model with custom cache size and quantization:
   By default cache_size is initialized to max sequence length of the model.
   However, if there is VRAM to spare, increase cache_size will have model to perform better for concurrent and batched request.
   By default, cache_quant is `FP16`. You can set `Q4`, `Q6`, or `Q8` to reduce KV cache VRAM if the model tolerates quantized cache.
   ```shell
   gallama run -id "model_name=mistral model_id=/path/to/mistral backend=exllamav3 cache_size=102400 cache_quant=Q8"
   ```
   
4. Launch a model with reduced cache size and quantization:
   For model with high context, lower the sequence length can significantly reduce VRAM usage.
   e.g. Mistral Large 2 can handle 128K content, however, it will require significant VRAM for the cache
   ```shell
   gallama run -id "model_name=mistral_large model_id=/path/to/mistral_large backend=exllamav3 max_seq_len=32768"
   ```

5. Launch a model for embedding:
   ```shell
   gallama run -id "model_name=gte-large-en-v1.5 model_id=Alibaba-NLP/gte-large-en-v1.5 backend=embedding"
   ```

6. Launch a model with speculative decoding:
   Only use a draft model that is compatible with the target model. For normal draft models this usually means the same tokenizer/vocabulary. For DFlash, use a DFlash draft model built for that target model family.

   ExLlamaV3 supports two speculative decoding modes:
   - normal flash draft: a smaller draft model proposes tokens with regular flash attention
   - DFlash draft: a DFlash draft model proposes a block of tokens and syncs accepted target states back into the draft cache

   Recommended DFlash setup in `~/gallama/model_config.yaml`:
   ```yaml
   qwen3.6-27B:
     backend: exllamav3
     model_id: /path/to/Qwen3.6-27B-exl3
     gpus: auto
     max_seq_len: 128000
     draft_model_id: /path/to/Qwen3.6-27B-DFlash
     draft_cache_quant: FP16
     backend_extra_args:
       num_draft_tokens: 15
   ```

   Then launch it by model name:
   ```shell
   gallama run qwen3.6-27B
   ```

   You can also pass a direct draft path from the CLI. Dotted keys can be used for `backend_extra_args`:
   ```shell
   gallama run -id "model_name=qwen3.6-27B model_id=/path/to/Qwen3.6-27B-exl3 backend=exllamav3 draft_model_id=/path/to/Qwen3.6-27B-DFlash draft_cache_quant=FP16 backend_extra_args.num_draft_tokens=15"
   ```

   For normal flash speculative decoding with ExLlamaV3, use a standard compatible draft model instead:
   ```shell
   gallama run -id "model_name=qwen2-72B model_id=/path/to/qwen2-72B backend=exllamav3 draft_model_id=/path/to/qwen2-draft"
   ```

   Ensure your GPU settings can accommodate both the target and draft model. Trial and adjust `gpus`, `draft_gpus`, and `cache_quant` for your hardware.

7. Tensor Parallel (TP)
   Exllama V2 Tensor Parallel support Tensor Parallel from v0.1.9.
   - Update your python>=3.11
   - Install ExllamaV2>=0.1.9
   - Only support Qwen2-72B, Llama3.1-70B and Mistral Large at the moment
   - Do run a draft model to help further with speed (Qwen2-1.5B, Llama3.1-8B, Mistral v0.3 respectively)
   To enable tensor parallel, simply add `tp=True`
   Exllama tensor parallel support parallelism on odd number of GPUs. Also exact matching of GPU is not requirement
   The speed boost for TP for dense model is huge (close to X1.5-X2).
   ```shell
   gallama run -id "model_name=qwen-2-72B model_id=/path/to/qwen-2-72B backend=exllama draft_model_id=/path/to/qwen-2-1.5B tp=True"
   ```
8. Others
   If you keep gallama config folder in another location instead of `~home/gallama` then you can set env parameter `GALLAMA_HOME_PATH` when running. 


# OpenAI realtime websocket (Experimental)
From version 0.0.9, gallama does provide a OpenAI Realtime websocket by wrapping Websocker over a TTS + LLM + TTS setup.
While this is not true Sound to Sound set up, it does provide a mock-up of OpenAI realtime websocket for testing.
The setup also provide integration with Video from Livekit for video voice chat app.

The Realtime Websocket API is tested working with follow:
- https://github.com/livekit-examples/realtime-playground.git
- https://github.com/openai/openai-realtime-console/tree/websockets

API Spec:
- https://platform.openai.com/docs/guides/realtime

To Use Video Chat feature
Please refer to the PAI app here:
- https://github.com/remichu-ai/pai.git
- https://github.com/remichu-ai/pai-agent.git

Do note that there are some package at Linux level that you will need to install. Refer to installation portion below.
While it does mimic openai realtime, there could be bug due to it not using native audio to audio model



## Legacy Model Downloader

The built-in model downloader is now considered outdated.

The preferred workflow is:

1. Download or prepare your model manually using your normal tooling.
2. Put it wherever you want on disk.
3. Add or update the corresponding entry in `~/gallama/model_config.yaml`.

The legacy downloader commands may still exist in parts of the codebase, but they are no longer the recommended way to manage models.
