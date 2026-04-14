from typing import Any, List, Tuple

class CommandBuffer:
    def __init__(self) -> None:
        # Tuple of (command_id: int, command_type: str, data: Any)
        self._commands: List[Tuple[int, str, Any]] = []

    def add_command(self, command_id: int, command_type: str, data: Any) -> None:
        self._commands.append((command_id, command_type, data))

    def remove_command(self, command_id: int) -> None:
        self._commands = [cmd for cmd in self._commands if cmd[0] != command_id]

    def clear(self) -> None:
        self._commands.clear()
        
    @property
    def queue(self) -> List[Tuple[int, str, Any]]:
        return self._commands
