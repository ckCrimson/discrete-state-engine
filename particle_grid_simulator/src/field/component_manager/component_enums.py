# Enums for Field Component Manager
from enum import Enum

class FieldCommandType(Enum):
    """
    ENUM: Operations allowed during an incremental hardware update.
    Passed as string values into the generic CommandBuffer.
    """
    SET = "SET"           # F_new = V
    ADD = "ADD"           # F_new = F_old + V
    MULTIPLY = "MULTIPLY" # F_new = F_old * V
    CLEAR = "CLEAR"       # F_new = Null Vector