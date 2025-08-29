from dataclasses import dataclass, field

@dataclass
class FrameworkConfig:
  # Framework settings
  pedantic_input_checking: bool = True
  execution_settings: dict = field(default_factory=lambda: {"counter_width": 0})