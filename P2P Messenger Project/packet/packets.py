from datetime import datetime
from typing import List

from .packet import Packet, register
from .types import Type, MessageID, ClientID, Address, PublicKey, TrackerQueryReplyEntry


@register(Type.LOCAL_QUERY)
class LocalQueryPacket(Packet):
    pass


@register(Type.LOCAL_ANNOUNCE)
class LocalAnnouncePacket(Packet):
    client_id: ClientID
    name: str
    public_key: PublicKey

    def __init__(self, client_id: ClientID, name: str, public_key: PublicKey) -> None:
        super().__init__()

        self.client_id = client_id
        self.name = name
        self.public_key = public_key


@register(Type.TRACKER_QUERY)
class TrackerQueryPacket(Packet):
    pass


@register(Type.TRACKER_QUERY_REPLY)
class TrackerQueryReplyPacket(Packet):
    clients: List[TrackerQueryReplyEntry]

    def __init__(self, clients: List[TrackerQueryReplyEntry]) -> None:
        super().__init__()

        self.clients = clients


@register(Type.TRACKER_ANNOUNCE)
class TrackerAnnouncePacket(Packet):
    client_id: ClientID
    name: str
    public_key: PublicKey

    def __init__(self, client_id: ClientID, name: str, public_key: PublicKey) -> None:
        super().__init__()

        self.client_id = client_id
        self.name = name
        self.public_key = public_key


@register(Type.TRACKER_ANNOUNCE_OK)
class TrackerAnnounceOkPacket(Packet):
    timestamp: datetime

    def __init__(self, timestamp: datetime) -> None:
        super().__init__()

        self.timestamp = timestamp


@register(Type.TRACKER_PUNCH_TO)
class TrackerPunchToPacket(Packet):
    client_id: ClientID

    def __init__(self, client_id: ClientID) -> None:
        super().__init__()

        self.client_id = client_id


@register(Type.TRACKER_PUNCH_FROM)
class TrackerPunchFromPacket(Packet):
    address: Address
    client_id: ClientID
    name: str

    def __init__(self, address: Address, client_id: ClientID, name: str) -> None:
        super().__init__()

        self.address = address
        self.client_id = client_id
        self.name = name


@register(Type.PEER_INIT)
class PeerInitPacket(Packet):
    client_id: ClientID
    name: str
    public_key: PublicKey

    def __init__(self, client_id: ClientID, name: str, public_key: PublicKey) -> None:
        super().__init__()

        self.client_id = client_id
        self.name = name
        self.public_key = public_key


@register(Type.PEER_MESSAGE)
class PeerMessagePacket(Packet):
    message_id: MessageID
    timestamp: datetime
    iv: bytes
    message: bytes
    aes_key: bytes

    def __init__(self, message_id: MessageID, timestamp: datetime, iv: bytes, message: bytes, aes_key: bytes) -> None:
        super().__init__()

        self.message_id = message_id
        self.timestamp = timestamp
        self.iv = iv
        self.message = message
        self.aes_key = aes_key


@register(Type.PEER_MESSAGE_OK)
class PeerMessageOkPacket(Packet):
    message_id: MessageID
    timestamp: datetime

    def __init__(self, message_id: MessageID, timestamp: datetime) -> None:
        super().__init__()

        self.message_id = message_id
        self.timestamp = timestamp
