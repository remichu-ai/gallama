# llm_server.py
from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from typing import AsyncGenerator, TypeVar
from collections import OrderedDict

from pydantic.json import pydantic_encoder
from ..api_response.chat_response import chat_completion_response_stream
from ..data_classes.realtime_server_proto import ContentTypeServer, ConversationItemServer, parse_conversation_item, MessageContentServer
from ..logger import logger
from ..data_classes.realtime_client_proto import *
from ..routes.chat import validate_api_request
from ..data_classes.data_class import (
    BaseMessage,
    ChatMLQuery,
    ToolCall,
    FunctionCall,
    ChatCompletionResponse,
    ChatMessage,
    OneTool,
    ToolCallResponse,
    StreamChoice,
    MultiModalTextContent,
    MultiModalImageContent,
    UsageResponse,
    ToolSpec,
    FunctionSpec,
    ParameterSpec,
)

from ..data_classes.internal_ws import WSInterLLM, WSInterConfigUpdate

from ..data_classes.generation_data_class import (
    GenQueueDynamic,
)
from ..data_classes.realtime_client_proto import (
    SessionConfig
)
from ..data_classes.video import VideoFrameCollection


import json
import asyncio

from ..dependencies import get_model_manager, get_video_collection, get_session_config, dependency_update_session_config


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

    def reset(self):
        self.conversation_history = OrderedDict()

    async def update_session_config(self, new_session_config: SessionConfig):
        """Update session config"""
        self.session_config = self.session_config.merge(new_session_config)

        # update session config in dependencies for video to refer to
        await dependency_update_session_config(self.session_config)

    def convert_conversation_to_chatml(
            self,
            session_config: SessionConfig,
            video_start_time: float = None,
            video_end_time: float = None,
    ) -> ChatMLQuery:
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
                    if content.type in [ContentTypeServer.INPUT_TEXT, ContentTypeServer.TEXT,
                                        ContentTypeServer.INPUT_AUDIO, ContentTypeServer.AUDIO]:
                        message_content.append(MultiModalTextContent(
                            type="text",
                            text=content.text or content.transcript
                        ))
                    elif content.type == "input_image":
                        logger.info("-----------------------------------------append images-------------------------------")
                        logger.info(content.image[:50])
                        message_content.append(MultiModalImageContent(
                            type="image_url",
                            image_url=MultiModalImageContent.ImageDetail(
                                url=content.image,
                                detail="high"
                            )
                        ))

                messages.append(BaseMessage(
                    role=role,
                    content=message_content
                ))
            elif item.type == "function_call":
                logger.info(f"Function call: {item}")
                messages.append(BaseMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=item.id,
                            type="function",
                            function=FunctionCall(
                                name=item.name,
                                arguments=item.arguments
                            )
                        )
                    ]
                ))
            elif item.type == "function_call_output":
                logger.info(f"Function call output: {item}")
                messages.append(BaseMessage(
                    role="tool",
                    content=item.output,
                    tool_call_id=item.call_id
                ))

        # Convert tools from SessionConfig to ToolSpec format
        tools = None
        if session_config.tools and len(session_config.tools) > 0:
            tools = [
                ToolSpec(
                    type="function",
                    function=FunctionSpec(
                        name=tool.name,
                        description=tool.description,
                        parameters=ParameterSpec(
                            type="object",
                            properties=tool.parameters.properties,
                            required=tool.parameters.required
                        )
                    )
                ) for tool in session_config.tools
            ]

        # Handling video
        video_for_llm = None
        if session_config.video.video_stream:
            video_collection = get_video_collection()
            if video_start_time and video_end_time:
                video_for_llm = video_collection.get_frames_between_timestamps(video_start_time, video_end_time)
            else:
                video_for_llm = video_collection.get_all_frames()
                video_collection.clear()

            logger.info(f"Number of video frames used for LLM: {len(video_for_llm)}")

            # Add special token to the messages if there is video
            if len(video_for_llm) > 0:
                messages.append(BaseMessage(
                    role="user",
                    content="{{VIDEO-PlaceHolderTokenHere}}"
                ))

                # Get retained video frames
                retained_video_frames = VideoFrameCollection.get_retained_frames_from_list(
                    frames=video_for_llm,
                    retained_video_frames_per_message=session_config.retained_video_frames_per_message,
                    return_base64=True
                )

                # Update the last conversation item with the retained frames if possible
                if retained_video_frames:
                    last_item = next(reversed(self.conversation_history.values()), None)
                    if last_item is not None and last_item.type == "message" and last_item.content:
                        for base64_frame in retained_video_frames:
                            last_item.content.append(
                                MessageContentServer(
                                    type="input_image",
                                    image=base64_frame,
                                )
                            )

        max_token_to_use = session_config.max_response_output_tokens if session_config and session_config.max_response_output_tokens != "inf" else None
        query = ChatMLQuery(
            messages=messages,
            model=session_config.model,
            temperature=session_config.temperature if session_config else 0.8,
            stream=True,
            max_tokens=max_token_to_use,
            artifact="No",
            tools=tools if tools else None,
            # tool extra settings
            tool_call_thinking=session_config.tool_call_thinking,
            tool_call_thinking_token=session_config.tool_call_thinking_token,
            tool_instruction_position=session_config.tool_instruction_position,
            tool_schema_position=session_config.tool_schema_position,
            video=video_for_llm if video_for_llm and len(video_for_llm) > 0 else None,
        )

        return validate_api_request(query)

    async def process_generation_stream(self, gen_queue: GenQueueDynamic, model_name: str, query: ChatMLQuery
        ) -> AsyncGenerator[dict, None]:
        """Process generation queue and yield response chunks by wrapping chat_completion_response_stream"""
        # Create a mock request object since chat_completion_response_stream expects one
        mock_request = type('MockRequest', (), {'is_disconnected': lambda: False})()

        async for chunk in chat_completion_response_stream(
            query=query,
            gen_queue=gen_queue,
            model_name=model_name,
            request=mock_request
        ):
            # Check if generation should be stopped
            if self.stop_event.is_set():
                break

            # Handle the [DONE] message
            if chunk.get("data") == "[DONE]":
                continue

            # Parse the chunk data
            data = json.loads(chunk["data"])

            # Handle usage statistics differently for websocket
            if "usage" in data:
                yield {
                    "type": "usage.update",
                    "usage": json.dumps(data)
                }
            # For normal message chunks, pass through the data
            else:
                yield data

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

        elif message.get("type") in ["response.create"]:
            message_pydantic = ResponseCreate(**message)

            # Reset stop event before starting new generation
            self.stop_event.clear()

            gen_queue = GenQueueDynamic()
            response_config = self.session_config.merge(message.get("response", {}))
            query = self.convert_conversation_to_chatml(
                response_config,
                video_start_time = message_pydantic.speech_start_time,
                video_end_time = message_pydantic.speech_end_time
            )

            # Ensure stream is set to True for streaming responses
            query.stream = True

            model_manager = get_model_manager()
            llm = model_manager.get_model(query.model, _type="llm")
            # llm = model_manager.llm_dict.get(query.model)
            #
            # if not llm:
            #     llm = model_manager.llm_dict.get(list(model_manager.llm_dict.keys())[0])

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
                async for chunk in self.process_generation_stream(gen_queue, llm.model_name, query):
                    await self.websocket.send_json(chunk)

                await self.websocket.send_json({
                    "type": "generation.complete",
                })
            # elif essage["type"] == "response.create_cache":  # generate_cache
            #     async for _ in self.process_generation_stream(gen_queue, llm.model_name, query):
            #         pass
            #     await self.websocket.send_json({
            #         "type": "pre_cache.complete",
            #     })
            else:
                raise Exception(f"Unknown message type: {message['type']}")
        elif message.get("type") == "common.cancel":
            # abort generation
            self.stop_event.set()
        elif message.get("type") == "common.cleanup":
            # abort generation
            self.reset()
        elif message.get("type") == "common.config_update":
            message_pydantic = WSInterConfigUpdate(**message)
            await self.update_session_config(message_pydantic.config)


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