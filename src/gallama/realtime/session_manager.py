from typing import Dict, Optional

from gallama.data_classes.realtime_client_proto import SessionConfig
from gallama.realtime.websocket_session import WebSocketSession


class SessionManager:
    """Handles session lifecycle"""

    def __init__(self):
        self.sessions: Dict[str, WebSocketSession] = {}

    def create_session(self, session_id: str, config: Optional[SessionConfig] = None) -> WebSocketSession:
        if config is None:
            config = SessionConfig(modalities=["text"])
        session = WebSocketSession(session_id, config)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[WebSocketSession]:
        return self.sessions.get(session_id)

    async def delete_session(self, session_id: str):
        if session_id in self.sessions:
            await self.sessions[session_id].cleanup()
            del self.sessions[session_id]
