# llm_server.py
from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from typing import Dict, List, Optional, Literal, Union, AsyncGenerator, TypeVar
from pydantic import BaseModel
from collections import OrderedDict
from gallama.logger.logger import logger
from gallama.data_classes.realtime_data_classes import *
from ..routes.chat import validate_api_request
from ..data_classes.data_class import (
    BaseMessage,
    ChatMLQuery,
    ChatCompletionResponse,
    ChatMessage,
    OneTool,
    ToolCallResponse,
    StreamChoice,
    MultiModalTextContent,
    MultiModalImageContent,
    UsageResponse,
)
from ..data_classes.generation_data_class import (
    GenerationStats,
    GenQueue,
    GenText,
    GenEnd,
    GenStart,
)
import time
from ..utils.utils import get_response_uid, get_response_tool_uid

import json
import asyncio

from ..dependencies import get_model_manager


router = APIRouter(prefix="", tags=["llm"])


class LLMGenerateParams(BaseModel):
    temperature: float = 0.8
    max_tokens: Optional[int] = None


class GenerateEvent(BaseModel):
    type: Literal["generate", "generate_cache"]
    params: Optional[LLMGenerateParams] = None





T = TypeVar("T", bound=ConversationItemServer)


class LLMWebSocketServer:
    def __init__(self):
        # Dictionary to store session-specific OrderedDicts
        self.sessions: Dict[str, OrderedDict[str, T]] = {}

    def convert_conversation_to_chatml(
        self,
        conversation_item_od: OrderedDict,
        session_config: SessionConfig
    ) -> ChatMLQuery:
        """Convert conversation history to ChatML format"""
        messages = []

        # Iterate through items in order they were added to OrderedDict
        for item_id, item in conversation_item_od.items():
            role = item.role.value if hasattr(item, 'role') and item.role else "user"

            # Handle different item types
            if item.type == "message" and item.content:
                message_content: List[Union[MultiModalTextContent, MultiModalImageContent]] = []

                for content in item.content:
                    if content.type in [ContentType.INPUT_TEXT, ContentType.TEXT]:
                        message_content.append(MultiModalTextContent(
                            type="text",
                            text=content.text
                        ))

                messages.append(BaseMessage(
                    role=role,
                    content=message_content
                ))

        # Create ChatMLQuery
        query = ChatMLQuery(
            messages=messages,
            model=session_config.model,
            temperature=session_config.temperature if session_config else 0.8,
            stream=True,  # Always stream for WebSocket
            max_tokens=session_config.max_response_output_tokens if session_config else None,
            artifact="No"  # Not using artifact handling
        )

        return validate_api_request(query)

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        session_id = str(id(websocket))

        # Initialize session-specific OrderedDict
        self.sessions[session_id] = OrderedDict()

        try:
            while True:
                message = await websocket.receive_json()
                logger.info(f"Received message: {message}")

                if message.get("object") == "realtime.item":
                    item = parse_conversation_item(message)
                    # Store item in session-specific OrderedDict
                    self.sessions[session_id][item.id] = item

                    # Acknowledge update
                    await websocket.send_json({
                        "type": "conversation.update.ack",
                        "item_id": item.id
                    })

                elif message.get("type") in ["response.create", "generate_cache"]:
                    # Create generation queue
                    gen_queue = GenQueue()

                    # Get session-specific conversation history
                    session_history = self.sessions[session_id]
                    # check if there is specific generation parameters
                    response_config = SessionConfig(**message.get("response"))

                    query = self.convert_conversation_to_chatml(session_history, response_config)

                    # Start generation task
                    model_manager = get_model_manager()
                    llm = model_manager.llm_dict.get(query.model)

                    if not llm:
                        # just take the model available  # TODO to make the code more proper
                        llm = model_manager.llm_dict.get(list(model_manager.llm_dict.keys())[0])

                    if llm is None:
                        error_response = {
                            "type": "error",
                            "error": {
                                "code": "model_not_found",
                                "message": f"Model '{query.model}' not found or not available"
                            }
                        }
                        await websocket.send_json(error_response)
                        continue

                    asyncio.create_task(
                        llm.chat(
                            query=query,
                            prompt_eng=llm.prompt_eng,
                            gen_queue=gen_queue,
                            request=None  # No HTTP request in WebSocket context
                        )
                    )

                    if message["type"] == "response.create":
                        # Process and send response chunks
                        async for chunk in self.process_generation_stream(gen_queue, llm.model_name):
                            await websocket.send_json(chunk)

                        # send completion event
                        await websocket.send_json({
                            "type": "generation.complete",
                        })

                    else:  # generate_cache
                        # Just process the generation without sending response
                        async for _ in self.process_generation_stream(gen_queue, llm.model_name):
                            pass
                        # Send acknowledgment
                        await websocket.send_json({
                            "type": "pre_cache.complete",
                        })

        except WebSocketDisconnect:
            self.cleanup_session(session_id)
        except Exception as e:
            logger.error(f"Error in LLM websocket: {str(e)}")
            self.cleanup_session(session_id)

    def cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]


    async def process_generation_stream(self, gen_queue: GenQueue, model_name: str) -> AsyncGenerator[dict, None]:
        """Process generation queue and yield response chunks"""
        unique_id = get_response_uid()
        created = int(time.time())
        full_response = ""
        eos = False
        gen_type = "text"
        gen_stats = None

        while not eos:
            accumulated_text = ""

            try:
                while True:
                    item = gen_queue.get_nowait()
                    if isinstance(item, GenText) and item.text_type == "text":
                        accumulated_text += item.content
                    elif isinstance(item, GenText) and item.text_type == "tool":
                        accumulated_text += item.content
                    elif isinstance(item, GenEnd):
                        eos = True
                        break
                    elif isinstance(item, GenStart):
                        gen_type = item.gen_type
                    elif isinstance(item, GenerationStats):
                        gen_stats = item
            except asyncio.QueueEmpty:
                pass

            if accumulated_text:
                full_response += accumulated_text

                if gen_type == "text":
                    chunk_data = ChatCompletionResponse(
                        unique_id=unique_id,
                        model=model_name,
                        object="chat.completion.chunk",
                        created=created,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChatMessage(
                                    role="assistant",
                                    content=accumulated_text
                                ),
                            )
                        ],
                    )
                    yield chunk_data.model_dump(exclude_unset=True)

                elif gen_type == "tool":
                    # Handle tool response
                    tool_response = json.loads(accumulated_text)
                    tools_list = []
                    for index, tool in enumerate(tool_response.get('functions_calling', [])):
                        tool_id = get_response_tool_uid()
                        tools_list.append(
                            ToolCallResponse(
                                id=tool_id,
                                index=index,
                                function=OneTool(
                                    name=tool['name'],
                                    arguments=json.dumps(tool['arguments']),
                                )
                            )
                        )
                    chunk_data = ChatCompletionResponse(
                        unique_id=unique_id,
                        model=model_name,
                        object="chat.completion.chunk",
                        created=created,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChatMessage(
                                    role="assistant",
                                    tool_calls=tools_list,
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                    )
                    yield chunk_data.model_dump(exclude_unset=True)

            if eos:
                # Log the full response at the end
                logger.info(f"----------------------LLM Response---------------\n{full_response.strip()}")

                # Include generation stats if available
                if gen_stats:
                    usage_data = ChatCompletionResponse(
                        unique_id=unique_id,
                        model=model_name,
                        object="chat.completion.chunk",
                        choices=[],
                        usage=UsageResponse(
                            prompt_tokens=gen_stats.input_tokens_count,
                            completion_tokens=gen_stats.output_tokens_count,
                            total_tokens=gen_stats.total_tokens_count,
                        ),
                    )
                    yield {
                        "type": "usage.update",
                        "usage": json.dumps(usage_data.model_dump(exclude_unset=True))
                    }

                    logger.info(f"{model_name} | LLM speed {gen_stats.generation_speed:.1f}/s tokens")


                # yield {"data": "[DONE]"}
            else:
                await asyncio.sleep(0.1)


llm_server = LLMWebSocketServer()

@router.websocket("/llm")
async def websocket_endpoint(websocket: WebSocket):
    await llm_server.handle_websocket(websocket)