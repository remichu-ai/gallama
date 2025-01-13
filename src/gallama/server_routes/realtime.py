from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Query
import logging
from gallama.realtime.websocket_handler import WebSocketMessageHandler
from ..data_classes import ModelInstanceInfo
from ..realtime.websocket_manager import WebSocketManager
from ..realtime.session_manager import SessionManager
from ..dependencies_server import get_server_manager

# Create router
router = APIRouter(prefix="", tags=["realtime"])

server_manager = get_server_manager()

# Initialize managers
session_manager = SessionManager()
message_handler = WebSocketMessageHandler()
#     stt_url="ws://localhost:8001/speech-to-text",
#     llm_url="ws://localhost:8002/llm",
#     tts_url="ws://localhost:8003/ws/speech"
# )
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

        stt_instance = server_manager.get_instance(model_type="stt")
        llm_instance = server_manager.get_instance(model_type="llm")
        tts_instance = server_manager.get_instance(model_type="tts")

        # Create session
        session = await websocket_manager.initialize_session(
            websocket,
            model=model,
            api_key=api_key,
            stt_url=f"ws://localhost:{stt_instance.port}/speech-to-text",
            llm_url=f"ws://localhost:{llm_instance.port}/llm",
            tts_url=f"ws://localhost:{tts_instance.port}/ws/speech",
        )

        try:
            await websocket_manager.start_background_tasks(session, websocket)

            while True:
                message = await websocket.receive_json()
                await message_handler.handle_message(websocket, session, message)

        except WebSocketDisconnect:
            await websocket_manager.stop_background_tasks(session)
            await message_handler.cleanup()
            await session_manager.delete_session(session.id)
            logging.info(f"WebSocket disconnected for session {session.id}")

        except Exception as e:
            logging.error(f"Error in websocket connection: {str(e)}")
            await websocket.close(code=4003, reason="Internal server error")

    except Exception as e:
        logging.error(f"Error establishing websocket connection: {str(e)}")
        await websocket.close(code=4003, reason="Connection error")
