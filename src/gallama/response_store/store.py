import asyncio
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..data_classes import BaseMessage, ResponsesCreateResponse


class ResponseStoreRecord(BaseModel):
    response_id: str
    model: str
    request: Dict[str, Any]
    response: ResponsesCreateResponse
    conversation_messages: List[BaseMessage] = Field(default_factory=list)
    previous_response_id: Optional[str] = None
    created_at: int = Field(default_factory=lambda: int(time.time()))
    store: bool = True


class ResponseStore:
    def __init__(self, max_items: int = 1000, ttl_seconds: Optional[int] = None):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._records: "OrderedDict[str, ResponseStoreRecord]" = OrderedDict()
        self._lock = asyncio.Lock()

    def _is_expired(self, record: ResponseStoreRecord) -> bool:
        return self.ttl_seconds is not None and record.created_at + self.ttl_seconds < int(time.time())

    def _prune_locked(self):
        expired_keys = [key for key, record in self._records.items() if self._is_expired(record)]
        for key in expired_keys:
            self._records.pop(key, None)

        while len(self._records) > self.max_items:
            self._records.popitem(last=False)

    async def put(self, record: ResponseStoreRecord):
        async with self._lock:
            self._records[record.response_id] = record
            self._records.move_to_end(record.response_id)
            self._prune_locked()

    async def get(self, response_id: str) -> Optional[ResponseStoreRecord]:
        async with self._lock:
            self._prune_locked()
            record = self._records.get(response_id)
            if record is None:
                return None
            self._records.move_to_end(response_id)
            return record

    async def delete(self, response_id: str) -> Optional[ResponseStoreRecord]:
        async with self._lock:
            self._prune_locked()
            return self._records.pop(response_id, None)
