# gallama - Guided Agentic Llama

Gallama is a Python library that provides a Llama backend optimized for local agentic tasks. It serves as a middleware layer, bridging the gap between base inference engines (such as ExllamaV2 and LLama CPP) and agentic work (e.g., function calling, formatting constraints).

## Features

- **OpenAI Compatible Server**: Fully compatible with OpenAI client, tested extensively with the OpenAI client.
```python
import os
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = 'test'
client = OpenAI(base_url = 'http://127.0.0.1:8000/v1')

messages = [{"role": "user", "content": "Cat or Dog is faster, in term of reaction speed?"}]

completion = client.chat.completions.create(
  model="mistral",
  messages=messages,
  tool_choice="auto"
)

print(completion)
```

- **Function Calling**: Supported function calling for all models.
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
Result:
```
Function(arguments='{"location": "Boston, MA"}', name='get_current_weather')
```

Tool option mimics OpenAI behavior, returning tool calling or normal response as needed.
```python
messages = [{"role": "user", "content": "Cat or Dog is faster, in term of reaction speed?"}]

completion = client.chat.completions.create(
  #model="llama3.1-8B",
  model="mistral",
  messages=messages,
  tools=tools,
  tool_choice="auto"      # Mimic OpenAI "auto" mode
)

print(completion.choices[0].message.content)
```
Response is normal generation and do not use tool:
```
In terms of reaction speed, a cat is generally faster than a dog. Cats have a faster visual processing system and can react more quickly to stimuli.
This is due to the structure of their eyes and brain, which are optimized for quick movements and changes in light.
However, it's important to note that individual animals may vary in their reactions based on factors such as age, health, and training.
```

- **Thinking Method**: A novel approach to guide LLM's thinking with XML template (e.g., chain of thought) without modifying the prompt.
```python
Following chat was using Mistral v0.3 at 4.0bpw
messages = [
    {"role": "user","content": """
I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?
"""}                    
]


completion = client.chat.completions.create(
  model="mistral",
  messages=messages,
  temperature=0.1,
  max_tokens= 200,
)
print(completion.choices[0].message.content)
```
Answer (wrong, correct answer is 10 apples)
```
You started with 10 apples. You gave away 4 (2 to the neighbor and 2 to the repairman), so you had 6 left. 
Then, you bought 5 more apples, making a total of 11 apples. 
After eating 1 apple, you were left with 10 - 6 (the initial apples) + 5 (the new apples) - 1 (the eaten apple) = 9 apples. 
So, you remained with 9 apples.
```

Now, using thinking for chain of thought, the correct answer is obtained without all the thinink step:
```python
thinking_template ="""
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
    {"role": "user","content": """
I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?
"""}                    
]

completion = client.chat.completions.create(
  model="mistral",
  messages=messages,
  temperature=0.1,
  max_tokens= 200,
  extra_body={
    "thinking_template": thinking_template,
  },
)

print(completion.choices[0].message.content)
```
Answer:
```
The user has 10 APPLES LEFT.
```
Thinking that the model generated behind the scene:
```xml
Now, the thinking template i need to apply to answer this question is:                                                                                                                                                                                       
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
My thinking using the XML template as follow:                                                                                                                                                                                                                
```xml                                                                                                                                                                                                                                                       
<chain_of_thought>                                                                                                                                                                                                                                           
  <problem>How many apples does the user have left after giving some away and buying more?</problem>                                                                                                                                                         
  <initial_state>The user started with 10 apples.</initial_state>                                                                                                                                                                                            
  <steps>                                                                                                                                                                                                                                                    
    <step>The user gave away 2 apples to the neighbor and 2 to the repairman (total = 4).</step>                                                                                                                                                             
    <step>The user bought 5 more apples, making a total of 10 + 5 - 4 = 11 apples.</step>                                                                                                                                                                    
    <step>The user ate 1 apple, leaving him with 11 - 1 = 10 apples.</step>                                                                                                                                                                                  
  </steps>                                                                                                                                                                                                                                                   
  <answer>The user has 10 apples left.</answer>                                                                                                                                                                                                              
  <final_answer>The user has 10 APPLES LEFT.</final_answer>                                                                                                                                                                                                  
</chain_of_thought>  
```

- **Multiple Concurrent Models**: Run multiple models (different or same) with automatic load balancing and request routing.
Load multiple models and customize which gpu it use. In following example, i specify which out of my 4 GPUs it uses, and cap by what VRAM per gpu 
```shell
gallama serve -id "model_id=mistral gpus=20,0,0,0" -id "model_id=llama3.1-8B gpus=0,20,0,0"
```

```
INFO     | ```python                                                                                                                                                                                                                                                  | server.py
           +------------------------------------------------------------------------------+                                                                                                                                                             
           | Current Status: 2 model(s) loaded with 2 total instance(s)                                                                                                                                 
           +------------------------------------------------------------------------------+                                                                                                                                                             
           | Model Name           | # | Ports                                                                                                                                                                                           
           +----------------------+---+---------------------------------------------------+                                                                                                                                                             
           | mistral              |  1 | 8001                                                                                                                                                                           
           | llama3.1-8B          |  1 | 8002                                                                                                                                           
           +------------------------------------------------------------------------------+                                                                                                                                                             
           | GPU Memory Information                                                                                                                                                                                                                     
           +-------+-------------+-------------+------------------------------------------+                                                                                                                                                             
           | GPU   | Used        | Free        | Total                                                                                                                                                                                                  
           +-------+-------------+-------------+------------------------------------------+                                                                                                                                                             
           | GPU 0: Used:  13.7GB, Free:  10.0GB, Total:  24.0GB                                                                                                                        
           | GPU 1: Used:  5.9GB, Free:  17.8GB, Total:  24.0GB                                                                                                                         
           | GPU 2: Used:  0.0GB, Free:  11.8GB, Total:  12.0GB                                                                                                                         
           | GPU 3: Used:  0.0GB, Free:  23.7GB, Total:  24.0GB                                                                                                                         
           | ------+-------------+-------------+------------------------------------------+                                                                                                                                                             
           | Total: Used: 19.6GB, Free: 63.1GB, Total: 84.0GB                                                                                                                                           
           +-------+-------------+-------------+------------------------------------------+                                                                                                                                                             
           ```                                                                         
```

- **OpenAI Embedding Endpoint**: Utilize Infinity Embedding for both inference and embedding via a single URL.
```python
response = client.embeddings.create(
    input="Your text string for embedding goes here",
    model="Alibaba-NLP/gte-large-en-v1.5"
)

print(response.data[0].embedding)
```

- **Legacy OpenAI Completion Endpoint**: Supported for Completion Endpoint.
```python
client.completions.create(
  model="mistral",
  prompt="Tell me a story about Llama in 200 words",
  max_tokens=1000,
  temperature=0
)
```

- **Remote Model Management**: Load and unload models via API calls.
```
api_url = "http://127.0.0.1:8000/add_model"

# Define the payload
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

# Make the POST request
response = requests.post(api_url, json=payload)
```
- **Regex Enforcement**: Ensure output conforms to specified patterns.
```python
completion = client.chat.completions.create(
  model="mistral",
  messages="Smoking is bad for health? Answer with Yes or No",
  temperature=0.1,
  max_tokens= 200,
  extra_body={
    # "leading_prompt": leading_prompt,         # prefix the generation with some string
    # "regex_pattern": regex_pattern,           # define the regex for the whole generation
    # "regex_prefix_pattern": '(Yes|No)\.',     # define the regex to match the starting words
    # "stop_words": stop_words,                 # define the word to stop generation
  },
)
```

- **Streaming**: Support for streaming responses using OpenAI client.
```ipython
messages = [{"role": "user","content": """tell me a 200 words story on llama"""}]

completion = client.chat.completions.create(
  model="mistral",
  messages=messages,
  stream=True,
  temperature=0.1,
)

for chunk in completion:
  #print(chunk)
  print(chunk.choices[0].delta.content, end='')
```


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
- Follow instructions at [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
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

```shell
gallama serve -id "model_id=mistral"
```

## Documentation

[Add link to full documentation or additional usage information]

## Contributing

[Add information about how to contribute to the project]

## License

[Specify the license under which gallama is distributed]

## Support

[Provide information on how users can get support or report issues]