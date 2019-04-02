import struct
import time
from datetime import datetime
from typing import Tuple, List, Any, TypeVar
from uuid import UUID

from .field import register, register_generic, FieldHandler, GenericFieldHandler, get_handler, register_subclass
from .types import Port

T = TypeVar("T")


@register(UUID)
class UUIDFieldHandler(FieldHandler[UUID]):
    def write(self, value: UUID) -> bytes:
        return value.bytes

    def read(self, bytes: bytes) -> Tuple[UUID, bytes]:
        return (UUID(bytes=bytes[0:16]), bytes[16:])


@register(datetime)
class TimestampFieldHandler(FieldHandler[datetime]):
    FORMAT = "!L"

    def write(self, value: datetime) -> bytes:
        return struct.pack(self.FORMAT, int(time.mktime(value.timetuple())))

    def read(self, bytes: bytes) -> Tuple[datetime, bytes]:
        size = struct.calcsize(self.FORMAT)
        return (datetime.fromtimestamp(struct.unpack(self.FORMAT, bytes[0:size])[0]), bytes[size:])


@register(str)
class StrFieldHandler(FieldHandler[str]):
    def write(self, value: str) -> bytes:
        bytes = value.encode("utf-8")
        return len(bytes).to_bytes(4, "big") + bytes

    def read(self, bytes: bytes) -> Tuple[str, bytes]:
        size = int.from_bytes(bytes[0:4], "big")
        return (bytes[4:4 + size].decode("utf-8"), bytes[4 + size:])


@register(bytes)
class BytesFieldHandler(FieldHandler[bytes]):
    def write(self, value: bytes) -> bytes:
        return len(value).to_bytes(4, "big") + value

    def read(self, bytes: bytes) -> Tuple[bytes, bytes]:
        size = int.from_bytes(bytes[0:4], "big")
        return bytes[4:4 + size], bytes[4 + size:]


@register(Port)
class PortFieldHandler(FieldHandler[Port]):
    def write(self, value: Port) -> bytes:
        return value.to_bytes(2, "big")

    def read(self, bytes: bytes) -> Tuple[Port, bytes]:
        return (Port(int.from_bytes(bytes[0:2], "big")), bytes[2:])


@register_generic(List[T])
class ListFieldHandler(GenericFieldHandler[List[T]]):
    def __init__(self, typ) -> None:
        super().__init__(typ)

        self.elem_type = typ.__args__[0]
        self.elem_handler = get_handler(self.elem_type)

    def write(self, value: List[T]) -> bytes:
        return len(value).to_bytes(4, "big") + b"".join(map(self.elem_handler.write, value))

    def read(self, bytes: bytes) -> Tuple[List[T], bytes]:
        size = int.from_bytes(bytes[0:4], "big")
        bytes = bytes[4:]

        elems = []
        for i in range(size):
            elem, bytes = self.elem_handler.read(bytes)
            elems.append(elem)

        return (elems, bytes)


@register_generic(Tuple[Any, ...])
@register_subclass(tuple)
class TupleFieldHandler(GenericFieldHandler[Tuple[Any, ...]]):
    def __init__(self, typ) -> None:
        super().__init__(typ)

        elem_types = None
        if hasattr(typ, "__args__"):  # Tuple[Any, ...]
            elem_types = typ.__args__
            self.constructor = tuple
        elif hasattr(typ, "_field_types"):  # NamedTuple
            elem_types = typ._field_types.values()
            self.constructor = lambda elems: self.typ(*elems)

        self.elem_handlers = list(map(get_handler, elem_types))

    def write(self, value: Tuple[Any, ...]) -> bytes:
        return b"".join(elem_handler.write(elem) for elem_handler, elem in zip(self.elem_handlers, value))

    def read(self, bytes: bytes) -> Tuple[Tuple[Any, ...], bytes]:
        elems = []

        for elem_handler in self.elem_handlers:
            elem, bytes = elem_handler.read(bytes)
            elems.append(elem)

        return (self.constructor(elems), bytes)
