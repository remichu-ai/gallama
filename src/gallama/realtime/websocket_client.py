import asyncio
import json
from typing import Optional, Dict, Any

import websockets
from pydantic import BaseModel
from websockets import WebSocketException
from websockets.protocol import State

from gallama.logger import logger


class WebSocketClient:
    def __init__(
            self,
            uri: str,
            reconnect_interval: float = 5.0,
            max_retries: int = 5,
            auto_reconnect: bool = True,
            max_receive_retries: int = 3  # Added parameter for receive_message retries
    ):
        self.uri = uri
        self.connection: Optional[websockets.WebSocketClientProtocol] = None
        self.reconnect_interval = reconnect_interval
        self.max_retries = max_retries
        self.auto_reconnect = auto_reconnect
        self.max_receive_retries = max_receive_retries  # Store the max receive retries
        self.retry_count = 0
        self.is_connecting = False

        self.websocket_options = {
            "ping_interval": 20,  # Enable periodic ping to detect connection issues
            "ping_timeout": 20,
            "close_timeout": 300,
            "max_size": None,
            "open_timeout": 60
        }

    async def _wait_for_connection(self, timeout: float = 30.0):
        """Wait for an ongoing connection attempt to complete."""
        start_time = asyncio.get_event_loop().time()
        while self.is_connecting:
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning("Timeout waiting for connection attempt")
                return False
            await asyncio.sleep(0.1)

        # After waiting, check if we actually have a valid connection
        return self.connection is not None and self.connection.state == State.OPEN

    async def ensure_connection(self) -> bool:
        """Ensures that the connection is active and attempts to reconnect if necessary."""
        try:
            # First check if we have an active connection
            if self.connection and self.connection.state == State.OPEN:
                try:
                    # Verify connection is truly alive with ping
                    pong_waiter = await self.connection.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    return True
                except Exception as e:
                    logger.debug(f"Ping check failed: {str(e)}")
                    # Connection is not truly alive
                    self.connection = None

            # If another task is already trying to connect, wait for it
            if self.is_connecting:
                logger.debug("Another task is connecting, waiting...")
                return await self._wait_for_connection()

            # Set connecting flag before attempting connection
            self.is_connecting = True
            try:
                return await self.connect()
            finally:
                self.is_connecting = False

        except Exception as e:
            logger.debug(f"Connection check failed: {str(e)}")
            return False

    async def connect(self) -> bool:
        """Establishes a WebSocket connection with retry logic."""
        if self.is_connecting:
            return await self._wait_for_connection()

        self.is_connecting = True
        try:
            while self.retry_count < self.max_retries:
                try:
                    self.connection = await websockets.connect(self.uri, **self.websocket_options)
                    self.retry_count = 0
                    logger.info("Successfully connected to WebSocket server")
                    return True
                except WebSocketException as e:
                    self.retry_count += 1
                    logger.warning(f"Connection attempt {self.retry_count} failed: {str(e)}")

                    if self.retry_count >= self.max_retries:
                        logger.error("Max retry attempts reached")
                        return False

                    await asyncio.sleep(self.reconnect_interval)
            return False
        finally:
            self.is_connecting = False

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Sends a dictionary message, handling reconnection if necessary."""
        try:
            if not await self.ensure_connection():
                return False

            await self.connection.send(json.dumps(message))
            return True
        except WebSocketException as e:
            logger.error(f"Error sending message: {str(e)}")
            if self.auto_reconnect:
                self.connection = None
                return await self.send_message(message)
            return False

    async def send_pydantic_message(self, message: BaseModel) -> bool:
        """Sends a Pydantic model message, handling reconnection if necessary."""
        try:
            if not await self.ensure_connection():
                return False

            await self.connection.send(message.model_dump_json())
            return True
        except WebSocketException as e:
            logger.error(f"Error sending Pydantic message: {str(e)}")
            if self.auto_reconnect:
                self.connection = None
                return await self.send_pydantic_message(message)
            return False

    async def receive_message(self, timeout: int = 30, max_retries: Optional[int] = None) -> Optional[str]:
        """
        Receives a message from the WebSocket connection with retry logic.

        Args:
            timeout: Time in seconds to wait for each receive attempt
            max_retries: Maximum number of retry attempts (defaults to self.max_receive_retries)

        Returns:
            The received message as a string, or None if receiving failed
        """
        retries = 0
        max_attempts = max_retries if max_retries is not None else self.max_receive_retries

        while retries <= max_attempts:
            try:
                if not await self.ensure_connection():
                    logger.warning(f"Connection failed on receive attempt {retries + 1}")
                    retries += 1
                    if retries > max_attempts:
                        return None
                    continue

                return await asyncio.wait_for(self.connection.recv(), timeout=timeout)

            except asyncio.TimeoutError:
                logger.warning(f"Timeout on receive attempt {retries + 1}")
                retries += 1
                if retries > max_attempts:
                    logger.error("Max receive retry attempts reached")
                    return None

            except WebSocketException as e:
                logger.error(f"Error receiving message on attempt {retries + 1}: {str(e)}")
                if self.auto_reconnect:
                    self.connection = None
                    retries += 1
                    if retries > max_attempts:
                        logger.error("Max receive retry attempts reached")
                        return None
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    return None

            except Exception as e:
                logger.error(f"Unexpected error on receive attempt {retries + 1}: {str(e)}")
                retries += 1
                if retries > max_attempts:
                    return None

    async def close(self):
        """Closes the WebSocket connection."""
        if self.connection:
            try:
                await self.connection.close()
            except WebSocketException as e:
                logger.error(f"Error closing connection: {str(e)}")
            finally:
                self.connection = None
                self.retry_count = 0

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()