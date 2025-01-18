from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Query
import logging
from gallama.realtime.websocket_handler import WebSocketMessageHandler
from ..data_classes import ModelInstanceInfo
from ..realtime.websocket_manager import WebSocketManager
from ..realtime.session_manager import SessionManager
from ..dependencies_server import get_server_manager, get_server_logger
import websockets

# Create router
router = APIRouter(prefix="", tags=["realtime"])

server_manager = get_server_manager()

logger = get_server_logger()

# Initialize managers
session_manager = SessionManager()
message_handler = WebSocketMessageHandler()
#     stt_url="ws://localhost:8001/speech-to-text",
#     llm_url="ws://localhost:8002/llm",
#     tts_url="ws://localhost:8003/ws/speech"
# )
websocket_manager = WebSocketManager(session_manager, message_handler)

@router.websocket("/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    llm_websocket = None  # Initialize LLM WebSocket connection
    try:
        # Get the LLM instance and its WebSocket URL
        llm_instance = server_manager.get_instance(model_type="llm")
        llm_websocket_url = f"ws://localhost:{llm_instance.port}/video"

        # Connect directly to the LLM WebSocket
        llm_websocket = await websockets.connect(llm_websocket_url)
        logger.info(f"Connected to LLM WebSocket at {llm_websocket_url}")

        while True:
            # Receive raw data from the client
            data = await websocket.receive_bytes()
            logger.info(f"Received {len(data)} bytes from /video WebSocket")

            # Forward the raw data directly to the LLM WebSocket
            await llm_websocket.send(data)

            # Optional: Send an acknowledgment back to the client
            await websocket.send_text("Data forwarded to LLM WebSocket")

    except WebSocketDisconnect:
        logger.info("Client disconnected from /video WebSocket")
    except Exception as e:
        logger.error(f"Error in /video WebSocket: {str(e)}")
    finally:
        # Close the LLM WebSocket connection if it exists
        if llm_websocket:
            await llm_websocket.close()
        # Close the /video WebSocket connection gracefully
        try:
            await websocket.close()
        except RuntimeError as e:
            logger.debug(f"Websocket already closed: {str(e)}")


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


