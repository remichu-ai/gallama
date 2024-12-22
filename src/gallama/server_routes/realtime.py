from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Query
import asyncio
import uuid
from typing import Dict, Optional
import logging
from typing import List
from gallama.data_classes.realtime_data_classes import (
    SessionConfig
)
from .realtime_backend import MessageQueues, WebSocketMessageHandler, WebSocketManager, WebSocketSession, SessionManager

# Create router
router = APIRouter(prefix="", tags=["realtime"])


# Initialize managers
session_manager = SessionManager()
message_handler = WebSocketMessageHandler(
    stt_url="ws://localhost:8001/speech-to-text",
    llm_url="ws://localhost:8002/llm",
    tts_url="ws://localhost:8003/ws/speech"
)
websocket_manager = WebSocketManager(session_manager, message_handler)


@router.websocket("/{path:path}")
async def websocket_endpoint(
    websocket: WebSocket,
    path: str,
    model: str = Query(..., description="Realtime model ID to connect to"),
):
    """
    WebSocket endpoint matching OpenAI's Realtime API specifications

    Connection URL: wss://api.openai.com/v1/realtime
    Required query parameters:
    - model: Realtime model ID (e.g., gpt-4o-realtime-preview-2024-12-17)
    Required headers:
    - Authorization: Bearer YOUR_API_KEY
    - OpenAI-Beta: realtime=v1
    """
    try:
        # Get headers
        headers = dict(websocket.headers)
        authorization = headers.get("authorization", "")
        openai_beta = headers.get("openai-beta")

        # Accept WebSocket connection with OpenAI protocol
        # protocols = websocket.headers.get("sec-websocket-protocol", "").split(", ")
        # if "openai-beta.realtime-v1" in protocols:
        #     await websocket.accept(subprotocol="openai-beta.realtime-v1")
        # else:
        #     await websocket.accept()

        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization.replace("Bearer ", "")

        # Create session
        session = await websocket_manager.initialize_session(
            websocket,
            model=model,
            api_key=api_key
        )

        try:
            await websocket_manager.start_background_tasks(session, websocket)

            while True:
                message = await websocket.receive_json()
                await message_handler.handle_message(websocket, session, message)

        except WebSocketDisconnect:
            await session_manager.delete_session(session.id)
            logging.info(f"WebSocket disconnected for session {session.id}")

        except Exception as e:
            logging.error(f"Error in websocket connection: {str(e)}")
            await session_manager.delete_session(session.id)
            await websocket.close(code=4003, reason="Internal server error")

    except Exception as e:
        logging.error(f"Error establishing websocket connection: {str(e)}")
        await websocket.close(code=4003, reason="Connection error")


# class MessageQueues:
#     """Manages different message queues for the websocket system"""
#
#     def __init__(self):
#         self.unprocessed = asyncio.Queue()  # Queue for items not yet processed
#         self.history: List[ConversationItem] = []  # Global history
#         self.latest_item: Optional[ConversationItem] = None  # Currently processing item
#         self.response_queue = asyncio.Queue()  # Queue for LLM responses
#         self.audio_to_client = asyncio.Queue()  # Queue for audio to be sent to client
#         self.lock = asyncio.Lock()  # Lock for thread-safe operations
#
#     async def add_to_history(self, item: ConversationItem):
#         """Add an item to global history"""
#         async with self.lock:
#             self.history.append(item)
#
#     def get_snapshot(self) -> List[ConversationItem]:
#         """Get current conversation snapshot including history and latest item"""
#         snapshot = self.history.copy()
#         if self.latest_item:
#             snapshot.append(self.latest_item)
#         return snapshot
#
#
# class WebSocketProcessor:
#     def __init__(self, stt_url: str, llm_url: str, tts_url: str):
#         self.queues = MessageQueues()
#         self.stt_url = stt_url
#         self.llm_url = llm_url
#         self.tts_url = tts_url
#         self.active_tasks: Dict[str, List[asyncio.Task]] = {}
#
#     async def process_unprocessed_queue(self, session_id: str):
#         """Process items in unprocessed queue one at a time"""
#         while True:
#             item: ConversationItem = await self.queues.unprocessed.get()
#
#             try:
#                 if item.content[0].type == ContentType.INPUT_TEXT:
#                     # Text items go straight to history
#                     await self.queues.add_to_history(item)
#
#                 elif item.content[0].type == ContentType.INPUT_AUDIO:
#                     # Process audio through STT
#                     self.queues.latest_item = item
#                     audio_data = base64.b64decode(item.content[0].audio)
#
#                     async with websockets.connect(self.stt_url) as stt_ws:
#                         transcription = await self.process_stt(stt_ws, audio_data)
#
#                         # Update item with transcription
#                         item.content.append(MessageContent(
#                             type=ContentType.INPUT_TEXT,
#                             text=transcription
#                         ))
#
#                         # Move to history and clear latest_item
#                         await self.queues.add_to_history(item)
#                         self.queues.latest_item = None
#
#             except Exception as e:
#                 logging.error(f"Error processing queue item: {str(e)}")
#             finally:
#                 self.queues.unprocessed.task_done()
#
#     async def process_llm_queue(self, session_id: str, websocket: WebSocket):
#         """Monitor response queue and process LLM requests"""
#         while True:
#             event = await self.queues.response_queue.get()
#
#             try:
#                 if event.get("type") == "response.create":
#                     # Wait for unprocessed queue to be empty
#                     await self.queues.unprocessed.join()
#
#                     # Get conversation snapshot
#                     snapshot = self.queues.get_snapshot()
#
#                     # Process through LLM
#                     async with websockets.connect(self.llm_url) as llm_ws:
#                         response = await self.process_llm(llm_ws, snapshot)
#
#                         # If audio output requested, send to TTS
#                         if "audio" in event.get("modalities", []):
#                             await self.queues.audio_to_client.put(response)
#                         else:
#                             # Send text response directly to client
#                             await websocket.send_json({
#                                 "type": "response.message",
#                                 "text": response
#                             })
#
#             except Exception as e:
#                 logging.error(f"Error processing LLM queue: {str(e)}")
#             finally:
#                 self.queues.response_queue.task_done()
#
#     async def process_audio_to_client(self, session_id: str, websocket: WebSocket):
#         """Monitor audio queue and send to client"""
#         while True:
#             text = await self.queues.audio_to_client.get()
#
#             try:
#                 async with websockets.connect(self.tts_url) as tts_ws:
#                     audio_data = await self.process_tts(tts_ws, text)
#                     await websocket.send_bytes(audio_data)
#
#             except Exception as e:
#                 logging.error(f"Error processing audio to client: {str(e)}")
#             finally:
#                 self.queues.audio_to_client.task_done()
#
#     async def handle_websocket(self, websocket: WebSocket, session_id: str):
#         """Main websocket handler"""
#
#         # Create processing tasks
#         tasks = [
#             asyncio.create_task(self.process_unprocessed_queue(session_id)),
#             asyncio.create_task(self.process_llm_queue(session_id, websocket)),
#             asyncio.create_task(self.process_audio_to_client(session_id, websocket))
#         ]
#         self.active_tasks[session_id] = tasks
#
#         try:
#             while True:
#                 message = await websocket.receive()
#
#                 if isinstance(message, dict):
#                     data = json.loads(message.get("text", "{}"))
#
#                     if data["type"] == "conversation.item.create":
#                         item = ConversationItem(**data["item"])
#                         await self.queues.unprocessed.put(item)
#
#                     elif data["type"] == "response.create":
#                         await self.queues.response_queue.put(data)
#
#         except WebSocketDisconnect:
#             # Cancel all tasks
#             for task in tasks:
#                 task.cancel()
#
#             # Clean up
#             if session_id in self.active_tasks:
#                 del self.active_tasks[session_id]
#
#     async def process_stt(self, ws, audio_data: bytes) -> str:
#         """Process audio through STT service"""
#         await ws.send(audio_data)
#         return await ws.recv()
#
#     async def process_llm(self, ws, snapshot: List[ConversationItem]) -> str:
#         """Process conversation through LLM service"""
#         await ws.send(json.dumps({
#             "messages": [item.dict() for item in snapshot]
#         }))
#         return await ws.recv()
#
#     async def process_tts(self, ws, text: str) -> bytes:
#         """Process text through TTS service"""
#         await ws.send(json.dumps({"text": text}))
#         return await ws.recv()
#
#
# # Initialize processor
# processor = WebSocketProcessor(
#     stt_url="ws://localhost:8001/speech-to-text",
#     llm_url="ws://llm-service/ws",
#     tts_url="ws://tts-service/ws"
# )
#
#
# class SessionManager:
#     def __init__(self):
#         self.sessions: Dict[str, SessionConfig] = {}
#         self.voice_used: Dict[str, bool] = {}  # Track if voice has been used in session
#
#     def create_session(self, session_id: str, config: Optional[SessionConfig] = None) -> SessionConfig:
#         """Create a new session with default or provided config"""
#         if config is None:
#             config = SessionConfig(modalities=["text"])
#         self.sessions[session_id] = config
#         self.voice_used[session_id] = False
#         return config
#
#     def get_session(self, session_id: str) -> Optional[SessionConfig]:
#         """Get session config by ID"""
#         return self.sessions.get(session_id)
#
#     def update_session(self, session_id: str, update_config: SessionConfig) -> Optional[SessionConfig]:
#         """Update session configuration"""
#         current_config = self.get_session(session_id)
#         if not current_config:
#             return None
#
#         # Create dictionary of current config
#         current_dict = current_config.model_dump()
#
#         # Create dictionary of update config, excluding None values
#         update_dict = {k: v for k, v in update_config.model_dump().items() if v is not None}
#
#         # Check voice update restriction
#         if 'voice' in update_dict and self.voice_used[session_id]:
#             # Remove voice from update if it's been used
#             del update_dict['voice']
#
#         # Update the configuration
#         current_dict.update(update_dict)
#
#         # Create new config with updated values
#         updated_config = SessionConfig(**current_dict)
#         self.sessions[session_id] = updated_config
#
#         return updated_config
#
#     def mark_voice_used(self, session_id: str):
#         """Mark that voice has been used in this session"""
#         self.voice_used[session_id] = True
#
#     def delete_session(self, session_id: str):
#         """Delete a session"""
#         if session_id in self.sessions:
#             del self.sessions[session_id]
#         if session_id in self.voice_used:
#             del self.voice_used[session_id]
#
#
# # Initialize session manager
# session_manager = SessionManager()
#
#
# @router.websocket("/{path:path}")
# async def websocket_endpoint(websocket: WebSocket, path: str):
#     # TODO this function is only for openAI protocol
#     # Accept with OpenAI protocol if requested
#     protocols = websocket.headers.get("sec-websocket-protocol", "").split(", ")
#     if "openai-beta.realtime-v1" in protocols:
#         await websocket.accept(subprotocol="openai-beta.realtime-v1")
#     else:
#         await websocket.accept()
#
#     session_id = str(id(websocket))
#
#     # for new connection:
#     # Create default session configuration
#     default_session = session_manager.create_session(session_id)
#     #session_manager.sessions[session_id] = default_session
#
#     # Send session.created event when connection first establish
#     await websocket.send_json({
#         "event_id": generate_event_id_uuid(),
#         "type": "session.created",
#         "session": default_session.model_dump()
#     })
#
#     # await processor.handle_websocket(websocket, session_id)
#
#     try:
#         while True:
#             message = await websocket.receive_json()
#
#             if message["type"] == "session.update":
#                 update_config = SessionConfig(**message["session"])
#                 updated_config = session_manager.update_session(session_id, update_config)
#
#                 if updated_config:
#                     # Send session.updated event
#                     await websocket.send_json({
#                         "event_id": generate_event_id_uuid(),
#                         "type": "session.updated",
#                         "session": {
#                             "id": session_id,
#                             "object": "realtime.session",
#                             "model": "gpt-4o-realtime-preview-2024-10-01",
#                             **updated_config.model_dump()
#                         }
#                     })
#
#             # Handle other message types...
#             elif message["type"] == "conversation.item.create":
#                 await processor.handle_conversation_item(websocket, session_id, message)
#
#             # Add other message type handlers as needed
#
#     except WebSocketDisconnect:
#         session_manager.delete_session(session_id)
#
#     except Exception as e:
#         print(f"Error in websocket connection: {str(e)}")
#         session_manager.delete_session(session_id)