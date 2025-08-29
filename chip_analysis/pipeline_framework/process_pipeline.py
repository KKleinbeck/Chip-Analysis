import copy, yaml
from pathlib import Path

from chip_analysis.pipeline_framework.config import FrameworkConfig
from chip_analysis.pipeline_framework.serilisable_inputs import SerialisableInputs
from chip_analysis.pipeline_framework.data_manager import data_managers
from chip_analysis.pipeline_framework.process_step import process_steps

class ProcessPipeline(SerialisableInputs):
  required_inputs = {
    "config_path": Path,
    "output_dir": Path,
    "inputs": dict,
  }

  optional_inputs = {
    "data_manager_type": (str, "native"),
    "framework_config": (dict, FrameworkConfig()),
  }

  def on_init(self):
    # Validate paths
    if not self.config_path.exists():
      raise FileNotFoundError(f"Config file not found: {self.config_path}")
    if not self.output_dir.exists():
      self.output_dir.mkdir(parents=True, exist_ok=True)

    # Create data manager
    if self.data_manager_type not in data_managers:
      raise ValueError(
        f"Unknown data manager '{self.data_manager_type}'. "
        f"Available: {', '.join(data_managers.keys())}"
      )
    self.data_manager = data_managers[self.data_manager_type](
      # output_dir=self.output_dir, input_file=self.input_file
      # TODO: define interface
      # Idea: output_dir and compiled hdf5 file
    )
    self.data_manager.register(self.inputs)

    # Load and validate config
    self.config = self._load_config()
    self._validate_inputs()
    self.pipeline_steps = self._validate_pipeline_steps()

  def _load_config(self) -> dict:
    """Load YAML configuration file."""
    with open(self.config_path, "r", encoding="utf-8") as f:
      try:
        config = yaml.safe_load(f)
      except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {self.config_path}: {e}")
    
    required_keys = {"Inputs", "PipelineSteps"}
    missing = required_keys - config.keys()
    if missing:
      raise ValueError(
        f"Config file {self.config_path} is missing required keys: {', '.join(missing)}"
      )
    return config

  def _validate_inputs(self):
    """Check consistency between declared config inputs and data manager contents."""
    declared_inputs = self.config.get("Inputs", [])
    if not isinstance(declared_inputs, list):
      raise ValueError("Config field 'Inputs' must be a list of strings.")
    # 1. Ensure every declared input is present
    for inp in declared_inputs:
      if not self.data_manager.contains(inp):
        raise ValueError(
          f"Config requires input '{inp}' which is not registered in data manager."
        )
            
    # 2. Ensure no extra registered inputs exist
    registered = set(self.data_manager.registered_results())
    declared = set(declared_inputs)
    extra = registered - declared
    if extra:
      msg = f"Data manager has inputs not declared in config: {', '.join(extra)}"
      if self.framework_config.pedantic_input_checking:
        raise ValueError(msg)
      else:
        import warnings
        warnings.warn(msg, UserWarning)

  def _validate_pipeline_steps(self) -> list[dict]:
    """Validate the structure of the PipelineSteps entry."""
    if "PipelineSteps" not in self.config:
      raise ValueError("Config must contain a 'PipelineSteps' entry.")

    steps = self.config["PipelineSteps"]
    if not isinstance(steps, list):
      raise ValueError("'PipelineSteps' must be a list.")

    required_keys = {"DisplayId", "ProcessStep", "Deliverables"}

    # Work on a copy of the data manager for validation purposes
    dm_copy = copy.deepcopy(self.data_manager)

    for i, step in enumerate(steps, start=1):
      if not isinstance(step, dict):
        raise ValueError(f"Step {i} is not a dictionary.")
      missing = required_keys - step.keys()
      if missing:
        raise ValueError(
          f"Step {i} is missing required keys: {', '.join(missing)}"
        )

      display_id = step["DisplayId"]

      # Validate Inputs
      if "Inputs" in step:
        inputs = step["Inputs"]
        if isinstance(inputs, dict) is False:
          raise ValueError("Step '{display_id}' (#{i}) has invalid 'Inputs' format. Must be a list.")
        
        for input in inputs.values():
          if not dm_copy.contains(input):
            raise ValueError(
              f"Step '{display_id}' (#{i}) requires input '{input}', "
              f"which is not available in data manager."
            )
            
      # Register Deliverables
      deliverables = step["Deliverables"]
      if isinstance(deliverables, dict) is False:
        raise ValueError("Step '{display_id}' (#{i}) has invalid 'Deliverables' format. Must be a dict.")
      
      try:
        dm_copy.register({k: None for k in deliverables.values()})
      except Exception as e:
        raise ValueError(
          f"Step '{display_id}' (#{i}) tried to register a deliverable "
          f"that was already defined earlier. Details: {e}"
        )

    return steps
  
  def run(self):
    total_steps = len(self.pipeline_steps)
    width = self.framework_config.execution_settings.get("counter_width", 3)

    for idx, step_config in enumerate(self.pipeline_steps, start=1):
      display_id = step_config["DisplayId"]
      process_name = step_config["ProcessStep"]

      # Print formatted step info
      print(f"[{idx:>{width}}/{total_steps:{width}}] Executing: {display_id}")

      if process_name not in process_steps:
        raise ValueError(f"Unknown ProcessStep '{process_name}' in step {idx}")

      # Prepare kwargs for instantiation
      kwargs = {"data_manager": self.data_manager}
      if "Inputs" in step_config:
        kwargs["inputs"] = step_config["Inputs"]
      if "Options" in step_config:
        kwargs["options"] = step_config["Options"]

      # Instantiate and execute
      process_class = process_steps[process_name]
      current_process = process_class(**kwargs)
      current_process.execute()
  
  def serialise(self, path: Path):
    pass