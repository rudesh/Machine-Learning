import typing

from .types import Type
from .field import get_handler
from . import fields
from collections import OrderedDict


PACKETS = {}


def register(type):
    def decorator(cls):
        PACKETS[type] = cls
        cls.type = type
        return cls
    return decorator


class Packet:
    type: Type

    def to_bytes(self) -> bytes:
        return self.type.value.to_bytes(1, "big") + self.to_bytes_payload()

    def to_bytes_payload(self) -> bytes:
        payload = b""

        for field, type_hint in Packet.get_type_hints(self.__class__).items():
            field_handler = get_handler(type_hint)
            value = getattr(self, field)
            payload += field_handler.write(value)

        return payload

    @classmethod
    def from_bytes(cls, bytes: bytes) -> "Packet":
        type = Type(bytes[0])
        payload = bytes[1:]
        return PACKETS.get(type, Packet).from_bytes_payload(payload)

    @classmethod
    def from_bytes_payload(cls, payload: bytes) -> "Packet":
        packet = cls.__new__(cls)

        for field, type_hint in Packet.get_type_hints(cls).items():
            field_handler = get_handler(type_hint)
            value, payload = field_handler.read(payload)
            setattr(packet, field, value)

        return packet

    def __str__(self) -> str:
        fields = ",".join(f"{field}={getattr(self, field)!r}" for field, type_hint in Packet.get_type_hints(self.__class__).items())
        return f"{self.type.name}({fields})"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def get_type_hints(cls):
        return OrderedDict((field, type_hint) for field, type_hint in typing.get_type_hints(cls).items() if field not in Packet.__annotations__)