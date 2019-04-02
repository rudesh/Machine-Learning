from typing import Tuple, TypeVar, Generic


HANDLERS = {}
GENERIC_HANDLERS = {}
SUBCLASS_HANDLERS = {}


def register(typ):
    def decorator(cls):
        field_handler = cls()
        HANDLERS[typ] = field_handler
        return cls
    return decorator


def register_generic(typ):
    assert hasattr(typ, "__origin__")

    def decorator(cls):
        GENERIC_HANDLERS[typ.__origin__] = cls
        return cls
    return decorator


def register_subclass(typ):
    def decorator(cls):
        SUBCLASS_HANDLERS[typ] = cls
        return cls
    return decorator


def get_handler(typ):
    if typ in HANDLERS:
        return HANDLERS[typ]
    elif hasattr(typ, "__origin__") and typ.__origin__ in GENERIC_HANDLERS:
        generic_handler = GENERIC_HANDLERS[typ.__origin__]
        field_handler = generic_handler(typ)
        HANDLERS[typ] = field_handler
        return field_handler
    else:
        for subtyp, subclass_handler in SUBCLASS_HANDLERS.items():
            if issubclass(typ, subtyp):
                field_handler = subclass_handler(typ)
                HANDLERS[typ] = field_handler
                return field_handler

        return None


T = TypeVar("T")


class FieldHandler(Generic[T]):
    def write(self, value: T) -> bytes:
        raise NotImplementedError()

    def read(self, bytes: bytes) -> Tuple[T, bytes]:
        raise NotImplementedError


class GenericFieldHandler(FieldHandler[T]):
    typ = None

    def __init__(self, typ) -> None:
        super().__init__()

        self.typ = typ
