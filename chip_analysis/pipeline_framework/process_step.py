from abc import ABC, abstractmethod

from chip_analysis.pipeline_framework.data_manager import DataManager

process_steps = {}

class AbstractProcessStep(ABC):
  inputs: dict[str, type] = {}
  deliverables: dict[str, type] = {}

  options: dict[str, tuple[type, any]] = {}

  def __init__(self,
               data_manager: DataManager,
               inputs: dict = None,
               options: dict = None):
    self.data_manager = data_manager
    self.options = options
    self._verify_and_set_inputs(inputs or {})
    self._configurate()
  
  def _verify_and_set_inputs(self, provided_inputs: dict):
    """Check provided inputs against required schema and set them as attributes."""
    required_keys = set(self.inputs.keys())
    provided_keys = set(provided_inputs.keys())

    # Check exact match
    if required_keys != provided_keys:
      missing = required_keys - provided_keys
      extra = provided_keys - required_keys
      msg = []
      if missing:
        msg.append(f"Missing inputs: {', '.join(missing)}")
      if extra:
        msg.append(f"Unexpected inputs: {', '.join(extra)}")
      raise ValueError("Input validation failed. " + "; ".join(msg))

    # Check types and assign attributes
    for key, expected_type in self.inputs.items():
      obj = self.data_manager.get(provided_inputs[key])
      if not isinstance(obj, expected_type):
        raise TypeError(
          f"Input '{key}' must be of type {expected_type.__name__}, "
          f"got {type(obj).__name__}"
        )
      setattr(self, key, obj)
  
  def _configurate(self):
    pass

  def execute(self):
    self._execute()
    self._validate_deliverables()
    self.data_manager.register({d: getattr(self, d) for d in self.deliverables})
  
  @abstractmethod
  def _execute(self):
    raise NotImplementedError("Subclasses must implement _execute method")
  
  def _validate_deliverables(self):
    """Check that deliverables exist as attributes and match expected types."""
    for key, expected_type in self.deliverables.items():
      if not hasattr(self, key):
        raise AttributeError(f"Deliverable '{key}' is missing as an attribute.")
      val = getattr(self, key)
      if not isinstance(val, expected_type):
        raise TypeError(
          f"Deliverable '{key}' must be of type {expected_type.__name__}, "
          f"got {type(val).__name__}"
      )