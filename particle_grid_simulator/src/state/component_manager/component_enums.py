from enum import Enum, auto

class CommandType(Enum):
    """
    Defines the types of domain mutations that can be queued
    in the CommandBuffer for the State Module.
    """
    ADD_BATCH = auto()
    DELETE_BATCH = auto()