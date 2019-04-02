import collections
import logging
from typing import Callable, DefaultDict, Type, Set

from packet import Packet, Address

logger = logging.getLogger(__name__)


PacketHandler = Callable[[Address, Packet], None]


class PacketDispatcher:
    handlers: DefaultDict[Type[Packet], Set[PacketHandler]]

    def __init__(self) -> None:
        super().__init__()

        self.handlers = collections.defaultdict(set)

    def add_handler(self, type: Type[Packet], handler: PacketHandler):
        self.handlers[type].add(handler)

    def dispatch(self, address: Address, packet: Packet):
        packet_type = type(packet)
        handlers = self.handlers[packet_type]

        if handlers:
            for handler in handlers.copy():
                handler(address, packet)
        else:
            logger.error(f"Unhandled packet type {packet_type}")
