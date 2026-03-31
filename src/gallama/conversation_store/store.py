import asyncio
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..data_classes import BaseMessage, ConversationResource


def _make_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex}"


class ConversationStoreRecord(BaseModel):
    id: str = Field(default_factory=_make_conversation_id)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    messages: List[BaseMessage] = Field(default_factory=list)

    def to_resource(self) -> ConversationResource:
        return ConversationResource(
            id=self.id,
            created_at=self.created_at,
            metadata=dict(self.metadata),
        )


class ConversationStore:
    def __init__(self, max_items: int = 1000, ttl_seconds: Optional[int] = None):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._records: "OrderedDict[str, ConversationStoreRecord]" = OrderedDict()
        self._lock = asyncio.Lock()

    def _is_expired(self, record: ConversationStoreRecord) -> bool:
        return self.ttl_seconds is not None and record.created_at + self.ttl_seconds < int(time.time())

    def _prune_locked(self):
        expired_keys = [key for key, record in self._records.items() if self._is_expired(record)]
        for key in expired_keys:
            self._records.pop(key, None)

        while len(self._records) > self.max_items:
            self._records.popitem(last=False)

    async def put(self, record: ConversationStoreRecord):
        async with self._lock:
            self._records[record.id] = record
            self._records.move_to_end(record.id)
            self._prune_locked()

    async def create(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        messages: Optional[List[BaseMessage]] = None,
    ) -> ConversationStoreRecord:
        record = ConversationStoreRecord(
            metadata=dict(metadata or {}),
            messages=[message.model_copy(deep=True) for message in messages or []],
        )
        await self.put(record)
        return record

    async def get(self, conversation_id: str) -> Optional[ConversationStoreRecord]:
        async with self._lock:
            self._prune_locked()
            record = self._records.get(conversation_id)
            if record is None:
                return None
            self._records.move_to_end(conversation_id)
            return record

    async def update_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> Optional[ConversationStoreRecord]:
        async with self._lock:
            self._prune_locked()
            record = self._records.get(conversation_id)
            if record is None:
                return None
            record.metadata = dict(metadata)
            self._records.move_to_end(conversation_id)
            return record

    async def append_messages(
        self,
        conversation_id: str,
        messages: List[BaseMessage],
    ) -> Optional[ConversationStoreRecord]:
        async with self._lock:
            self._prune_locked()
            record = self._records.get(conversation_id)
            if record is None:
                return None
            record.messages.extend(message.model_copy(deep=True) for message in messages)
            self._records.move_to_end(conversation_id)
            return record

    async def delete(self, conversation_id: str) -> Optional[ConversationStoreRecord]:
        async with self._lock:
            self._prune_locked()
            return self._records.pop(conversation_id, None)
