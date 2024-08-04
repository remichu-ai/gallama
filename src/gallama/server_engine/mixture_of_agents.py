from fastapi import FastAPI, HTTPException, Request
import json
import textwrap
from copy import deepcopy
from gallama.data_classes import AgentWithThinking, MixtureOfAgents
from typing import List, Dict, Union
from fastapi.responses import Response, StreamingResponse
from .request_handler import forward_request
import asyncio


async def modify_request(request: Request, changes: Dict[str, any]):
    body = await request.json()
    body_copy = deepcopy(body)

    # Apply the changes to the body
    for key, value in changes.items():
        if key in body_copy:
            body_copy[key] = value

    modified_body = json.dumps(body_copy).encode()

    # Create a copy of the headers and update the Content-Length
    modified_headers = dict(request.headers)
    modified_headers["content-length"] = str(len(modified_body))

    return modified_body, modified_headers


async def consolidate_responses(original_request: Request, responses: List[Response]):
    # This is a placeholder implementation. Replace with your actual consolidation logic.
    consolidation_prompt = "\n---\n" + textwrap.dedent("""
    Please synthesize the following reference answer into a single, high-quality response.
    It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
    """).strip() + "\n---\n"

    for index, response in enumerate(responses):
        response_body = json.loads(response.body)
        response_msg = response_body["choices"][0].get("message")

        # add the answer to the prompt
        consolidation_prompt += f"Reference answer {str(index+1)}\n"
        if response_msg and response_msg.get("content"):
            consolidation_prompt += response_msg["content"] + "\n"

        if response_msg and response_msg.get("tool_calls"):
            for tool_call in response_msg.get("tool_calls"):
                consolidation_prompt += str(tool_call["function"]) + "\n"

        consolidation_prompt += "---\n"

    consolidation_prompt += "Now provide the final synthesized response:\n"

    # Create a modified version of request with the answers from respective agents
    original_body = await original_request.json()

    modified_messages = original_body.get("messages", [])

    # append the final synthesis prompt to the message list
    modified_messages.append({
        "role": "user",
        "content": consolidation_prompt
    })

    modified_body, modified_headers = await modify_request(
        original_request,
        changes={"messages": modified_messages}
    )

    return modified_body, modified_headers


async def forward_to_multiple_agents(request: Request, agent_list: List[Union[str, AgentWithThinking]], modified_body: str, modified_headers: str, models: Dict, active_requests: Dict):
    tasks = []
    for agent in agent_list:
        if isinstance(agent, str):
            instance = await get_instance_for_model(agent, models, active_requests)
            tasks.append(forward_request(request, instance, modified_body, modified_headers))
        elif isinstance(agent, AgentWithThinking):
            instance = await get_instance_for_model(agent.model, models, active_requests)
            if agent.thinking_template:
                # Modify the request to include the thinking in the thinking_template field
                agent_body = json.loads(modified_body)
                agent_body["thinking_template"] = agent.thinking_template
                agent_modified_body = json.dumps(agent_body).encode()
                agent_modified_headers = dict(modified_headers)
                agent_modified_headers["content-length"] = str(len(agent_modified_body))
                tasks.append(forward_request(request, instance, agent_modified_body, agent_modified_headers))
            else:
                tasks.append(forward_request(request, instance, modified_body, modified_headers))
    return await asyncio.gather(*tasks)


async def get_instance_for_model(model: str, models: Dict, active_requests: Dict):
    running_instances = [inst for inst in models[model].instances if inst.status == "running"]
    if not running_instances:
        raise HTTPException(status_code=503, detail=f"No running instances available for model: {model}")

    return min(running_instances, key=lambda inst: active_requests[inst.port])


async def handle_mixture_of_agent_request(request: Request, body_json: dict, models: Dict, active_requests: Dict):
    moa_config = MixtureOfAgents(**body_json.get("mixture_of_agents"))
    agent_list = moa_config.agent_list
    master_agent = body_json.get("model")

    if not agent_list or not master_agent:
        raise HTTPException(status_code=400, detail="Invalid request: missing agent_list or master_agent")

    # Validate all models upfront
    all_models = [agent.model if isinstance(agent, AgentWithThinking) else agent for agent in agent_list] + [master_agent]

    # Create a modified request where streaming is turned off
    changes = {"stream": False}
    modified_body, modified_headers = await modify_request(request, changes)

    # Forward the request to all models in the agent_list
    responses = await forward_to_multiple_agents(request, agent_list, modified_body, modified_headers, models, active_requests)

    # Consolidate responses
    consol_modified_body, consol_modified_headers = await consolidate_responses(request, responses)

    # Forward consolidated response to master_agent
    instance = await get_instance_for_model(master_agent, models, active_requests)
    final_response = await forward_request(request, instance, consol_modified_body, consol_modified_headers)
    return final_response

