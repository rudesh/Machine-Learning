from typing import NewType, NamedTuple, Tuple
from datetime import datetime
import enum
import uuid


@enum.unique
class Type(enum.IntEnum):
    LOCAL_QUERY = 1
    LOCAL_ANNOUNCE = 2

    TRACKER_QUERY = 3
    TRACKER_QUERY_REPLY = 4
    TRACKER_ANNOUNCE = 5
    TRACKER_ANNOUNCE_OK = 6

    PEER_MESSAGE = 7
    PEER_MESSAGE_OK = 8

    TRACKER_PUNCH_TO = 9
    TRACKER_PUNCH_FROM = 10

    PEER_INIT = 11


# MessageID = typing.NewType("MessageID", uuid.UUID)
MessageID = uuid.UUID

ClientID = uuid.UUID

IP = str
Port = NewType("Port", int)
Address = Tuple[IP, Port]
PublicKey = bytes


class TrackerQueryReplyEntry(NamedTuple):
    address: Address
    client_id: ClientID
    name: str
    timestamp: datetime
    public_key: PublicKey

