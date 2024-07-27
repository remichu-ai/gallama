# gallama - Guided Agentic Llama

Gallama is a Python library that provides a Llama backend optimized for local agentic tasks. It serves as a middleware layer, bridging the gap between base inference engines (such as ExllamaV2 and LLama CPP) and agentic work (e.g., function calling, formatting constraints).

## Features

- **OpenAI Compatible Server**: Fully compatible with OpenAI client, tested extensively with the OpenAI client.
- **Function Calling**: Supported for all models. Tool option mimics OpenAI behavior, returning tool calling or normal response as needed.
- **Multiple Concurrent Models**: Run multiple models (different or same) with automatic load balancing and request routing.
- **OpenAI Embedding Endpoint**: Utilize Infinity Embedding for both inference and embedding via a single URL.
- **Legacy OpenAI Completion Endpoint**: Supported for backward compatibility.
- **Remote Model Management**: Load and unload models via API calls.
- **Regex Enforcement**: Ensure output conforms to specified patterns.
- **Streaming**: Support for streaming responses.
- **Thinking Method**: A novel approach to guide LLM's thinking (e.g., chain of thought) without modifying the prompt.

## Installation

Gallama requires certain components to be installed and functioning. 

If you already have ExllamaV2 (and optionally Flash Attention) running, then you can install gallama by:
```
pip install gallama
```
Or, install from source:
```
git clone https://github.com/waterangel91/gallama.git cd gallama pip install .
```

If you start from scratch and doesnt have these dependency yet, follow these steps:

1. Create a virtual environment (recommended):
```
conda create --name genv python=3.11 conda activate genv
```
2. Install and verify ExllamaV2:
- Follow instructions at [ExllamaV2 GitHub](https://github.com/turboderp/exllamav2)
- Test with examples from [ExllamaV2 Examples](https://github.com/turboderp/exllamav2/tree/master/examples)

3. (Optional) Install Flash Attention for improved performance:
- Follow instructions at [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)

4. (Optional) Install Llama CPP:
- Note: ExllamaV2 is currently recommended. Llama CPP support is under development.

5. Install gallama:
```
pip install gallama
```
Or, install from source:
```
git clone https://github.com/waterangel91/gallama.git cd gallama pip install .
```
## Usage

[Add basic usage examples here]

## Documentation

[Add link to full documentation or additional usage information]

## Contributing

[Add information about how to contribute to the project]

## License

[Specify the license under which gallama is distributed]

## Support

[Provide information on how users can get support or report issues]