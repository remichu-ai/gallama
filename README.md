# gallama - Guided Agentic Llama

__gallama__ is an opinionated Python library that provides a LLM inference API service backend optimized for local agentic tasks.
It focuses on model serving, realtime, multimodal, and local inference integrations rather than multi-agent orchestration.

Gallama is predominantly tested with the Exllama V3 workflow at this point. Other backends are still available, but they may have bugs or rough edges depending on the model and feature path.

Currently, the backend is mainly using Exllama-family backends. Llama.cpp support is under experiment. 

Do checkout [TabbyAPI](https://github.com/theroyallab/tabbyAPI) if you want a reliable and pure ExllamaV2 API backend.


# Quick Start
   Head down to the installation guide at the bottom of this page.
   Then check out the [Examples_Notebook.ipynb](https://github.com/remichu-ai/gallama/blob/main/examples/Examples_Notebook.ipynb) in the examples folder
   A simple python streamlit frontend chat UI code is included in the examples folder [streamlit](https://github.com/remichu-ai/gallama/blob/main/examples/streamlit_chatbot.py)
   Or checkout [GallamaUI](https://github.com/remichu-ai/gallamaUI.git) 

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

See [`src/tests/test_openai.py`](./src/tests/test_openai.py) and [`src/tests/test_openai_server.py`](./src/tests/test_openai_server.py) for more complete examples.

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

See [`src/tests/test_anthropic.py`](./src/tests/test_anthropic.py) for a more complete Anthropic client example suite.

### Claude Code
You can also point Claude Code at a local Gallama server by overriding the Anthropic base URL and auth token:

```shell
ANTHROPIC_BASE_URL="http://127.0.0.1:8000/" ANTHROPIC_AUTH_TOKEN="local" claude --model minimax
```

This lets Claude Code talk to your local model through Gallama's Anthropic-compatible API.

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
   - Exllama V2, llama.cpp, transformers, vLLM, sglang, and other backends are still available, but expect some backend-specific rough edges.

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

### model_config.yaml

Each top-level key is the model name that Gallama will expose through the API. The value under that key is the configuration used to load the backend.

Minimal Exllama example:

```yaml
mistral:
  backend: exllama
  model_id: /home/your-user/gallama/models/Mistral-7B-instruct-v0.3-4.5bpw-exl2
  prompt_template: Mistral_large
  gpus: auto
```

Typical keys:

- `backend`: backend name such as `exllama`, `llama_cpp`, `transformers`, `embedding`, `kokoro`, or `gpt_sovits`
- `model_id`: local path to the model or model directory
- `prompt_template`: prompt formatter to use for the model family
- `gpus`: usually `auto`, but can also be a per-GPU split
- `max_seq_len`: override context length if needed
- `cache_quant`: KV cache quantization such as `FP16`, `Q4`, `Q6`, or `Q8`
- `quant`: optional metadata for the model quantization you downloaded
- `eos_token_list`: optional extra EOS tokens for models that need them
- `backend_extra_args`: backend-specific options, commonly used for `transformers`, `sglang`, `gpt_sovits`, and similar backends

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

Example with TTS voice presets:

```yaml
gpt_sovits:
  backend: gpt_sovits
  model_id: /home/your-user/gallama/models/gpt_sovits
  backend_extra_args:
    device: cuda
    is_half: false
    version: v2
    chunk_size_in_s: 0.1
    bert_base_path: /home/your-user/gallama/models/gpt_sovits/pretrained_models/chinese-roberta-wwm-ext-large
    cnhuhbert_base_path: /home/your-user/gallama/models/gpt_sovits/pretrained_models/chinese-hubert-base
    t2s_weights_path: /home/your-user/gallama/models/gpt_sovits/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
    vits_weights_path: /home/your-user/gallama/models/gpt_sovits/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
  voice:
    shenhe:
      language: en
      ref_audio_path: /path/to/reference.wav
      ref_audio_transcription: I'm not trying to save the world...
      speed_factor: 1.1
```

Notes:

- Use the YAML key itself as the API model name. For example, if the key is `qwen-2.5-32B`, then that is the model string to pass in the client request.
- `prompt_template` matters. If the wrong one is chosen, the model may still load but chat quality or tool use can break.
- `backend_extra_args` is the place for backend-specific tuning such as custom tokenizer/model/processor classes or TTS model paths.
- You can keep your Gallama config in another location by setting `GALLAMA_HOME_PATH`.
   
### Advanced Usage

Using `gallama run -id` followed by a string which is a dictionary of key-value pair will unlock additional option as following:

Customize the model launch using various parameters. Available parameters for the `-id` option include:

- `model_id`: ID of the model from the yml file (required)
- `model_name`: Name of the model (optional, defaults to the last part of `model_id`)
- `gpus`: VRAM usage for each GPU, comma-separated list of floats (optional)
- `cache_size`: Context length for cache text in integers (optional)
- `cache_quant`: Quantization to use for cache, options are "FP16", "Q4", "Q6", "Q8" (optional, defaults to Q4)
- `max_seq_len`: Maximum sequence length (optional)
- `backend`: Model engine backend, options are "exLlama", "Llama_cpp", "embedding" (optional, defaults to "exLlama")
- `tp`: enable tensor parallel with exllama v2 (experimental). See further below

#### Speculative Decoding Parameters
- `draft_model_id`: ID of the draft model (optional)
- `draft_model_name`: Name of the draft model (optional)
- `draft_gpus`: VRAM usage for each GPU for the draft model, comma-separated list of floats (optional)
- `draft_cache_size`: Context length for cache text in integers for the draft model (optional)
- `draft_cache_quant`: Quantization to use for cache for the draft model, options are "FP16", "Q4", "Q6", "Q8" (optional)

### Examples

1. Launch two models simultaneously:
   ```shell
   gallama run -id "model_id=mistral" -id "model_id=llama3"
   ```

2. Launch a model with specific VRAM limits per GPU:
   ```shell
   gallama run -id "model_id=qwen2-72B gpus=22,22,10,0"
   ```
   This limits memory usage to 22GB for GPU0 and GPU1, 10GB for GPU2, and 0GB for GPU3.

3. Launch a model with custom cache size and quantization:
   By default cache_size is initialized to max sequence length of the model.
   However, if there is VRAM to spare, increase cache_size will have model to perform better for concurrent and batched request.
   By default, cache_quant=Q4 will be used. However, do adjust it if required e.g. Qwen2 1.5B doesn't work well with Q4 cache, please use Q6 or Q8.
   ```shell
   gallama run -id "model_id=mistral cache_size=102400 cache_quant=Q8"
   ```
   
4. Launch a model with reduced cache size and quantization:
   For model with high context, lower the sequence length can significantly reduce VRAM usage.
   e.g. Mistral Large 2 can handle 128K content, however, it will require significant VRAM for the cache
   ```shell
   gallama run -id "model_id=mistral_large max_seq_len=32768"
   ```

5. Launch a model for embedding:
   ```shell
   gallama run -id "model_id=Alibaba-NLP/gte-large-en-v1.5 backend=embedding"
   ```

6. Launch a model with speculative decoding:
   Only model with same vocabulary should be used for speculative decoding.
   For reference, by enabling speculative decoding, qwen2-72B generation speed improve from 20tok/s to 25-35tok/s on my 4090s.
   Highly recommend speculative decoding if you have VRAM to spare.
   ```shell
   gallama run -id "model_id=qwen2-72B draft_model_id=qwen2-1.5B"
   ```
   Ensure your GPU settings can accommodate the model requirements. Trial and adjust parameters as needed for your specific use case.
   Note: The backend is assumed to be the same for both the main model and the draft model in speculative decoding.

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
   gallama run -id "model_id=qwen-2-72B draft_model_id=qwen-2-1.5B tp=True"
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
