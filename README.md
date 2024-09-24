# gallama - Guided Agentic Llama

__gallama__ is an opinionated Python library that provides a LLM inference API service backend optimized for local agentic tasks.

It tries to close the gap between pure inference engine (such as ExLlamaV2 and Llama.cpp) and additional needs for agentic work (e.g., function calling, formatting constraints).

It provides some experimental features such as Artifact (like Claude's) and Thinking template (to guide chain of though).

As there is no standard approach for these experimental feature current, the library does implement it via some boilerplate code (you will find some fixed prompt inside the code) and thus is an opinionated API engine. 

This library is marked with rolling update so please be patient hi-cup and bugs :) (feel free to open an issue if you come across any).

Currently, the backend is mainly using ExllamaV2. Llama.cpp support is under experiment. 

Do checkout [TabbyAPI](https://github.com/theroyallab/tabbyAPI) if you want a reliable and pure ExllamaV2 API backend.

# Quick Start
   Head down to the installation guide at the bottom of this page.
   Then check out the [Examples_Notebook.ipynb](https://github.com/remichu-ai/gallama/blob/main/examples/Examples_Notebook.ipynb) in the examples folder
   A simple python streamlit frontend chat UI code is included in the examples folder [streamlit](https://github.com/remichu-ai/gallama/blob/main/examples/streamlit_chatbot.py)

# Features

## Integrated Model downloader

Ability to download exl2 model from Hugging Face via CLI for popular models.

Example as following to download llama-3.1 8B at 4.0bpw (4 bit per weight quantization)

```shell
gallama download qwen-2.5-32B:4.0
```

By default, it will download `exllama` version. For model with `gguf` available, you can download it by set the `backend`
```shell
gallama download qwen-2.5-32B:4.0 --backend=llama_cpp
```

After download, you can run the model with:
```shell
gallama run qwen-2.5-32B
```

Here is the list of currently supported models for downloader:

You can view it in CLI using following command:
```shell
gallama list available
```

**LLM Models**

| Model           | Backend    | Available Quantizations (bpw)                                                 |
|-----------------|------------|-------------------------------------------------------------------------------|
| llama-3.1-8B    | exllama    | `3.0`, `3.5`, `4.0`, `4.5`, `5.0`, `6.0`, `8.0`                               |
|                 | llama_cpp  | `2.0`, `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                      |
| llama-3.1-70B   | exllama    | `2.5`, `3.0`, `3.5`, `4.0`, `4.5`, `5.0`, `6.0`                               |
|                 | llama_cpp  | `3.0`, `4.0`, `5.0`                                                           |
| mistral         | exllama    | `2.8`, `3.0`, `4.0`, `4.5`, `5.0`, `6.0`                                      |
|                 | llama_cpp  | `2.0`, `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                      |
| mistral-large   | exllama    | `2.3`, `2.5`, `2.75`, `3.0`, `3.5`, `3.75`, `4.0`, `4.25`, `4.5`, `4.75`, `5.0`, `6.0` |
| mistral-nemo    | exllama    | `2.5`, `3.0`, `3.5`, `4.0`, `4.5`, `5.0`, `6.0`, `8.0`                        |
|                 | llama_cpp  | `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                             |
| codestral       | exllama    | `3.0`, `3.5`, `4.25`, `5.0`, `6.5`, `8.0`                                     |
|                 | llama_cpp  | `2.0`, `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                      |
| gemma-2-9B      | exllama    | `2.5`, `3.0`, `3.5`, `4.0`, `4.5`, `5.0`, `5.5`, `6.0`, `8.0`                 |
| gemma-2-27B     | exllama    | `3.0`, `3.5`, `4.0`, `4.5`, `5.0`, `6.0`, `8.0`                               |
| qwen-2-1.5B     | exllama    | `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                             |
|                 | llama_cpp  | `4.0`, `5.0`, `6.0`, `8.0`                                                    |
| qwen-2-7B       | exllama    | `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                             |
| qwen-2-72B      | exllama    | `3.0`, `3.5`, `4.0`, `4.25`, `4.65`, `5.0`, `6.0`, `8.0`                      |
|                 | llama_cpp  | `2.0`, `3.0`, `4.0`                                                           |
| yi-1.5-34B      | exllama    | `3.0`, `3.5`, `3.75`, `4.0`, `4.25`, `5.0`, `6.0`, `6.5`, `8.0`               |
| qwen-2.5-72B    | exllama    | `2.2`, `3.0`, `3.5`, `4.25`, `4.65`, `5.0`, `6.5`, `8.0`                      |
|                 | llama_cpp  | `2.0`, `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                      |
| qwen-2.5-32B    | exllama    | `3.0`, `4.0`, `4.65`, `5.0`, `6.0`, `8.0`                                     |
|                 | llama_cpp  | `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                             |
| qwen-2.5-14B    | exllama    | `3.0`, `3.5`, `4.25`, `5.0`, `6.5`, `8.0`                                     |
|                 | llama_cpp  | `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                             |
| qwen-2.5-7B     | exllama    | `3.5`, `4.25`, `5.0`, `6.5`, `8.0`                                            |
|                 | llama_cpp  | `3.0`, `4.0`, `5.0`, `6.0`, `8.0`                                             |


**Embedding Models:**

| Model | Backend | Available Quantizations (bpw) |
|-------|---------|-------------------------------|
| multilingual-e5-large-instruct | embedding | `16.0` |
| gte-large-en-v1.5 | embedding | `16.0` |

The syntax to specify the model is model name follow by `:` then follow by quantization float number e.g. `qwen-2-72B:4.0`

For model not listed here, you can refer to `examples/Model_Downloader.ipynb` for code to download from huggingface.
And then you will need to manually update the config in `~/gallama/model_config.yaml`. Instruction will be updated

Sometime, the download speed might get slowed down. Just cancel the download by Ctrl+C and run the same download command again. The download will resume with higher speed.

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

## Artifact System (Experimental)
The engine support artifact mode (similar to Claude's Artifact).
While Claude probably train their model with custom dataset, we neither have this with open source nor I think there will be a standard for it (much like every new model use a different special tokens :D)

Current the library implement it via a combination of prompting, XML tag and format enforcement.

To be able to use Artifact mode effectively, you will need a good enough model, from our testing you will need minimally codestral 22B.
70B+ models such as Qwen2 or Llama 3.1 will give the best result.

As Artifact feature is most useful for interaction instead of via API, we do have a lightweight WebUI here which work with our Artifact System.

If using python, you can interact with the UI as following.

__Non-Streaming__
```python
completion = client.chat.completions.create(
    model="codestral",
    messages=[{"role": "user",
               "content": "Explain in one liner what quicksort is and write me quicksort in python and Java"}],
    extra_body={
        "artifact": "Fast",
    }
)

for choice in completion.choices:
    print("-------------------------------\n")
    print(choice.message.artifact_meta)
    print(choice.message.content)
```

__Streaming__
```python
completion = client.chat.completions.create(
  model="codestral",
    messages=[{"role": "user",
               "content": "Explain in one liner what quicksort is and write me quicksort in python and Java"}],
  stream=True,
  extra_body= {
      "artifact": "Fast",
  }
)

current_content_type = None
for chunk in completion:
  if current_content_type != chunk.choices[0].delta.artifact_meta["tag_type"]:
      print("\n------------------------------")
      print(chunk.choices[0].delta.artifact_meta)
      print("\n------------------------------")
      current_content_type = chunk.choices[0].delta.artifact_meta["tag_type"]
  print(chunk.choices[0].delta.content, end='')
```


## Thinking Method
A novel approach to guide LLM's thinking with XML templates (e.g., chain of thought) without modifying the prompt.
The benefit of this versus traditional CoT prompting is you can customize the XML template to be used depending on model and situation without the need to fix change the prompt,
As following example, you can influence the model to provide the answer with CoT while not showing the CoT in the answer.

```python
thinking_template = """
<chain_of_thought>
  <problem>{problem_statement}</problem>
  <initial_state>{initial_state}</initial_state>
  <steps>
    <step>{action1}</step>
    <step>{action2}</step>
    <!-- Add more steps as needed -->
  </steps>
  <answer>Provide the answer</answer>
  <final_answer>Only the final answer, no need to provide the step by step problem solving</final_answer>
</chain_of_thought>
"""

messages = [
    {"role": "user", "content": "I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?"}
]

completion = client.chat.completions.create(
    model="mistral",
    messages=messages,
    temperature=0.1,
    max_tokens=200,
    extra_body={
        "thinking_template": thinking_template,
    },
)

print(completion.choices[0].message.content)
# 10 apples
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

## Mixture of Experts
Query and combine response from multiple model via OpenAI client



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
Streaming is fully supported even in artifact mode.

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

Ensure that you have either ExllamaV2 (recommended) or Llama CPP Python (in development) installed. You can have both installed and use both with gallama as well,
If you already have either ExLlamaV2 (and optionally Flash Attention) running, you can install gallama by:

```shell
pip install gallama
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

2. Install and verify ExLlamaV2 (Recommended):
   - Follow instructions at [ExLlamaV2 GitHub](https://github.com/turboderp/exLlamav2)
   - Test with examples from [ExLlamaV2 Examples](https://github.com/turboderp/exLlamav2/tree/master/examples)

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
   Or, install from source:
   ```shell
   git clone https://github.com/remichu-ai/gallama.git
   cd gallama
   pip install .
   ```

## Usage

Follow these steps to use the model. We aim to make this process more convenient in future releases.

### Setup

1. Download the ExLlama model. A Jupyter notebook to assist with the download is included in `examples/Model_download.ipynb`.

2. Initialize gallama:
   ```shell
   gallama run
   ```
   This creates a `model_config.yaml` file in `~/gallama`.

3. Update `~/gallama/model_config.yaml` with your model configurations.

4. Launch the model:
   Simple method
      ```shell
      gallama run mistral
      ```
   Advanced method
      ```shell
      gallama run -id "model_id=mistral"
      ```
   
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
   The speed boost for TP is huge (close to X1.5-X2).
   ```shell
   gallama run -id "model_id=qwen-2-72B draft_model_id=qwen-2-1.5B tp=True"
   ```
8. Others
   If you keep gallama config folder in another location instead of `~home/gallama` then you can set env parameter `GALLAMA_HOME_PATH` when running. 
