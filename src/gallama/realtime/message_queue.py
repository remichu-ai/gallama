import asyncio
from collections import OrderedDict
from typing import TypeVar, Optional, Union, Tuple, List

from starlette.websockets import WebSocket

from gallama.data_classes.realtime_data_classes import ConversationItemServer, ConversationItem, \
    ConversationItemCreated, parse_conversation_item
from gallama.realtime.websocket_client import WebSocketClient

T = TypeVar("T", bound=ConversationItemServer)


class MessageQueues:
    """Manages different message queues for the websocket system"""
    def __init__(self):
        self.unprocessed = asyncio.Queue()

        # ordered dict of conversation item
        self.conversation_item_od: OrderedDict[str, T] = OrderedDict()

        self.response_queue = asyncio.Queue()
        self.audio_to_client = asyncio.Queue()
        self.response_counter = 0
        self.event_counter = 0
        self.item_counter = 0

        self.lock_conversation_item = asyncio.Lock()
        self.lock_response_counter = asyncio.Lock()
        self.lock_event_counter = asyncio.Lock()
        self.lock_item_counter = asyncio.Lock()

        self.latest_item: Optional[ConversationItem] = None




        # self.uncommitted_audio_data: Optional[bytes] = None
        # self.uncommitted_text: Optional[str] = None

    async def next_event(self) -> str:
        """ return the next counter for event"""
        async with self.lock_event_counter:
            self.event_counter += 1
            return f"event_{self.event_counter}"

    async def next_resp(self) -> str:
        """ return the next counter for response"""
        async with self.lock_response_counter:
            self.response_counter += 1
            return f"resp_{self.response_counter}"

    async def next_item(self, return_current=False) -> Union[str,Tuple[str,Union[str, None]]]:
        """ return the next counter for response"""
        async with self.lock_item_counter:
            self.item_counter += 1
            if return_current:
                if not self.conversation_item_od:
                    return f"item_{self.item_counter}", None
                else:
                    return f"item_{self.item_counter}", next(reversed(self.conversation_item_od.keys()))
            else:
                return f"item_{self.item_counter}"

    async def get_previous_item_id(self, message_id: str) -> Optional[str]:
        """Get the ID of the message that comes before the given message_id"""
        async with self.lock_conversation_item:
            previous_id = None
            for id in self.conversation_item_od.keys():
                if id == message_id:
                    return previous_id
                previous_id = id
            return None  # Message ID not found


    async def update_conversation_item_ordered_dict_client(
        self,
        item: ConversationItemServer        # item to create
    ):
        """
        Add an item to the conversation item list.
        If it is a new item, return a ConversationItemCreated object for onward return to front end
        """
        async with self.lock_conversation_item:
            # update to history
            if isinstance(item, ConversationItem):
                self.conversation_item_od[item.id] = item


    async def update_conversation_item_ordered_dict(
        self,
        ws_client: WebSocket,               # web socket for client
        ws_llm: WebSocketClient,            # web socket for llm
        item: ConversationItemServer        # item to create
    ):
        """
        Add an item to the conversation item list.
        If it is a new item, return a ConversationItemCreated object for onward return to front end
        """
        new_event_id = await self.next_event()
        previous_item_id = await self.get_previous_item_id(item.id)

        # async with self.lock_conversation_item:
        await ws_llm.send_pydantic_message(item)

        # send client update about new item created
        async with self.lock_conversation_item:
            if not item.id:
                item.id = await self.next_item()

            if item.id not in self.conversation_item_od.keys():
                item_to_send = ConversationItemCreated(**{
                    "event_id": new_event_id,
                    "type": "conversation.item.created",
                    "previous_item_id": previous_item_id,
                    "item": item.model_dump()
                }).model_dump()

            # update to history with server type item
            item_server = parse_conversation_item(item.model_dump())
            self.conversation_item_od[item.id] = item_server

        # send to client
        await ws_client.send_json(item_to_send)


    def get_snapshot(self) -> List[ConversationItem]:
        snapshot = self.conversation_item_od.copy()
        # if self.latest_item:
        #     snapshot.append(self.latest_item)
        return snapshot
