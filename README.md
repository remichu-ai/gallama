# gallama - Guided Agentic Llama

gallama is a Python library that provides a Llama backend optimized for local agentic tasks. It serves as a middleware layer, bridging the gap between base inference engines (such as ExLlamaV2 and Llama.cpp) and agentic work (e.g., function calling, formatting constraints).

## Features

### OpenAI Compatible Server
Fully compatible with the OpenAI client, extensively tested for seamless integration.

```python
import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = 'test'
client = OpenAI(base_url='http://127.0.0.1:8000/v1')

messages = [{"role": "user", "content": "Which is faster in terms of reaction speed: a cat or a dog?"}]

completion = client.chat.completions.create(
    model="mistral",
    messages=messages,
    tool_choice="auto"
)

print(completion)
```

### Function Calling
Supports function calling for all models, mimicking OpenAI's behavior.

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

### Thinking Method
A novel approach to guide LLM's thinking with XML templates (e.g., chain of thought) without modifying the prompt.

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
```

### Multiple Concurrent Models
Run multiple models (different or same) with automatic load balancing and request routing.

```shell
gallama serve -id "model_id=mistral gpus=20,0,0,0" -id "model_id=Llama3.1-8B gpus=0,20,0,0"
```

### OpenAI Embedding Endpoint
Utilize Infinity Embedding for both inference and embedding via a single URL.

```python
response = client.embeddings.create(
    input="Your text string for embedding goes here",
    model="Alibaba-NLP/gte-large-en-v1.5"
)

print(response.data[0].embedding)
```

### Legacy OpenAI Completion Endpoint
Support for the Completion Endpoint is maintained.

```python
client.completions.create(
    model="mistral",
    prompt="Tell me a story about a Llama in 200 words",
    max_tokens=1000,
    temperature=0
)
```

### Remote Model Management
Load and unload models via API calls.

```python
import requests

api_url = "http://127.0.0.1:8000/add_model"

payload = [
    {
        "model_id": "qwen2-72B",
        "gpus": [22,22,4,0],
        "cache_size": 32768,
    },
    {
        "model_id": "gemma2-9b",
        "gpus": [0,0,0,20],
        "cache_size": 32768,
    },
    {
        "model_id": "/home/remichu/work/ML/model/multilingual-e5-large-instruct",
        "gpus": [0,0,0,1],
        "model_type": "embedding",
    },
]

response = requests.post(api_url, json=payload)
```

### Regex Enforcement
Ensure output conforms to specified patterns.

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

### Streaming
Support for streaming responses using the OpenAI client.

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

## Installation

gallama requires certain components to be installed and functioning. 

If you already have ExLlamaV2 (and optionally Flash Attention) running, you can install gallama by:

```shell
pip install gallama
```

Or, install from source:

```shell
git clone https://github.com/waterangel91/gallama.git
cd gallama
pip install .
```

If you're starting from scratch and don't have these dependencies yet, follow these steps:

1. Create a virtual environment (recommended):
   ```shell
   conda create --name genv python=3.11
   conda activate genv
   ```

2. Install and verify ExLlamaV2:
   - Follow instructions at [ExLlamaV2 GitHub](https://github.com/turboderp/exLlamav2)
   - Test with examples from [ExLlamaV2 Examples](https://github.com/turboderp/exLlamav2/tree/master/examples)

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
   gallama serve
   ```
   This creates a `model_config.yaml` file in `~/.gallama`.

3. Update `~/.gallama/model_config.yaml` with your model configurations.

4. Launch the model:
   ```shell
   gallama serve -id "model_id=mistral"
   ```

### Advanced Usage

Customize the model launch using various parameters. Available parameters for the `-id` option include:

- `model_id`: ID of the model from the yml file (required)
- `model_name`: Name of the model (optional, defaults to the last part of `model_id`)
- `gpus`: VRAM usage for each GPU, comma-separated list of floats (optional)
- `cache_size`: Context length for cache text in integers (optional)
- `cache_quant`: Quantization to use for cache, options are "FP16", "Q4", "Q6", "Q8" (optional, defaults to Q4)
- `max_seq_len`: Maximum sequence length (optional)
- `backend`: Model engine backend, options are "exLlama", "Llama_cpp", "embedding" (optional, defaults to "exLlama")

#### Speculative Decoding Parameters
- `draft_model_id`: ID of the draft model (optional)
- `draft_model_name`: Name of the draft model (optional)
- `draft_gpus`: VRAM usage for each GPU for the draft model, comma-separated list of floats (optional)
- `draft_cache_size`: Context length for cache text in integers for the draft model (optional)
- `draft_cache_quant`: Quantization to use for cache for the draft model, options are "FP16", "Q4", "Q6", "Q8" (optional)

### Examples

1. Launch two models simultaneously:
   ```shell
   gallama serve -id "model_id=mistral" -id "model_id=Llama3"
   ```

2. Launch a model with specific VRAM limits per GPU:
   ```shell
   gallama serve -id "model_id=qwen2-72B gpus=22,22,10,0"
   ```
   This limits memory usage to 22GB for GPU0 and GPU1, 10GB for GPU2, and 0GB for GPU3.

3. Launch a model with custom cache size and quantization:
   ```shell
   gallama serve -id "model_id=mistral cache_size=4096 cache_quant=Q8"
   ```

4. Launch a model with a specific backend:
   ```shell
   gallama serve -id "model_id=mistral backend=Llama_cpp"
   ```

5. Launch a model with speculative decoding:
   ```shell
   gallama serve -id "model_id=mistral draft_model_id=mistral-small draft_gpus=10,10"
   ```

Ensure your GPU settings can accommodate the model requirements. Adjust parameters as needed for your specific use case.

Note: The backend is assumed to be the same for both the main model and the draft model in speculative decoding.