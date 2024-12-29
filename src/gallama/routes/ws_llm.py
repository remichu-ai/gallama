# llm_server.py
from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from typing import Dict, List, Optional, Literal, Union, AsyncGenerator, TypeVar
from pydantic import BaseModel
from collections import OrderedDict
from ..logger import logger
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
from ..data_classes.realtime_data_classes import (
    SessionConfig
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


class WebSocketSession:
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.conversation_history = OrderedDict()
        self.session_config = SessionConfig()

        # job cancelling
        self.stop_event = asyncio.Event()

    def update_session_config(self, new_session_config: SessionConfig):
        """Update session config"""
        self.session_config = SessionConfig(**{
            **self.session_config.model_dump(),
            **new_session_config
        })

    def convert_conversation_to_chatml(self, session_config: SessionConfig) -> ChatMLQuery:
        """Convert conversation history to ChatML format"""
        messages = []

        if session_config.instructions != "":
            messages.append(BaseMessage(
                role="system",
                content=session_config.instructions
            ))

        for item_id, item in self.conversation_history.items():
            role = item.role.value if hasattr(item, 'role') and item.role else "user"

            if item.type == "message" and item.content:
                message_content: List[Union[MultiModalTextContent, MultiModalImageContent]] = []

                for content in item.content:
                    if content.type in [ContentTypeServer.INPUT_TEXT, ContentTypeServer.TEXT, ContentTypeServer.INPUT_AUDIO, ContentTypeServer.AUDIO]:
                        message_content.append(MultiModalTextContent(
                            type="text",
                            text=content.text or content.transcript
                        ))

                messages.append(BaseMessage(
                    role=role,
                    content=message_content
                ))

        # logger.info(f"messages: {self.conversation_history.items()}")
        max_token_to_use = session_config.max_response_output_tokens if session_config and session_config.max_response_output_tokens!="inf" else None
        query = ChatMLQuery(
            messages=messages,
            model=session_config.model,
            temperature=session_config.temperature if session_config else 0.8,
            stream=True,
            max_tokens=max_token_to_use,
            artifact="No"
        )

        return validate_api_request(query)

    async def process_generation_stream(self, gen_queue: GenQueue, model_name: str) -> AsyncGenerator[dict, None]:
        """Process generation queue and yield response chunks"""
        unique_id = get_response_uid()
        created = int(time.time())
        full_response = ""
        eos = False
        gen_type = "text"
        gen_stats = None

        while not eos:
            # Check if generation should be stopped
            if self.stop_event.is_set():
                break

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
                logger.info(f"----------------------LLM Response---------------\n{full_response.strip()}")

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
            else:
                await asyncio.sleep(0.1)

    @staticmethod
    def update_ordered_dict(od: OrderedDict, target_key, new_value) -> OrderedDict:
        return OrderedDict((k, new_value if k == target_key else v) for k, v in od.items())


    async def handle_message(self, message: dict):
        """Handle incoming WebSocket messages for this session"""
        logger.info(f"Received message: {message}")

        if message.get("object") == "realtime.item":
            item = parse_conversation_item(message)

            if item.id not in self.conversation_history.keys():
                # new item
                self.conversation_history[item.id] = item
            else:
                # update existing item
                self.conversation_history = self.update_ordered_dict(
                    od=self.conversation_history,
                    target_key=item.id,
                    new_value=item,
                )

            await self.websocket.send_json({
                "type": "conversation.update.ack",
                "item_id": item.id
            })
        elif message.get("type") == "conversation.item.deleted":
            # Handle deletion event
            item_id = message.get("item_id")
            if item_id in self.conversation_history:
                del self.conversation_history[item_id]

            await self.websocket.send_json({
                "type": "conversation.update.ack",
                "item_id": item_id
            })


        elif message.get("type") in ["response.create", "generate_cache"]:
            # Reset stop event before starting new generation
            self.stop_event.clear()

            gen_queue = GenQueue()
            response_config = self.session_config.merge(message.get("response"))
            query = self.convert_conversation_to_chatml(response_config)

            model_manager = get_model_manager()
            llm = model_manager.llm_dict.get(query.model)

            if not llm:
                llm = model_manager.llm_dict.get(list(model_manager.llm_dict.keys())[0])

            if llm is None:
                error_response = {
                    "type": "error",
                    "error": {
                        "code": "model_not_found",
                        "message": f"Model '{query.model}' not found or not available"
                    }
                }
                await self.websocket.send_json(error_response)
                return

            asyncio.create_task(
                llm.chat(
                    query=query,
                    prompt_eng=llm.prompt_eng,
                    gen_queue=gen_queue,
                    request=None,
                    stop_event=self.stop_event,
                )
            )

            if message["type"] == "response.create":
                async for chunk in self.process_generation_stream(gen_queue, llm.model_name):
                    await self.websocket.send_json(chunk)

                await self.websocket.send_json({
                    "type": "generation.complete",
                })
            else:  # generate_cache
                async for _ in self.process_generation_stream(gen_queue, llm.model_name):
                    pass
                await self.websocket.send_json({
                    "type": "pre_cache.complete",
                })
        elif message.get("type") == "common.cancel":
            # abort generation
            self.stop_event.set()


class LLMWebSocketServer:
    def __init__(self):
        self.sessions: Dict[str, WebSocketSession] = {}

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        session_id = str(id(websocket))

        session = WebSocketSession(websocket, session_id)
        self.sessions[session_id] = session

        try:
            while True:
                message = await websocket.receive_json()
                await session.handle_message(message)
        except WebSocketDisconnect:
            self.cleanup_session(session_id)
        except Exception as e:
            logger.error(f"Error in LLM websocket: {str(e)}")
            self.cleanup_session(session_id)

    def cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]


llm_server = LLMWebSocketServer()


@router.websocket("/llm")
async def websocket_endpoint(websocket: WebSocket):
    await llm_server.handle_websocket(websocket)