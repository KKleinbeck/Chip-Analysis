import git, h5py

from abc import ABC, abstractmethod
from pathlib import Path


class SerialisableInputs(ABC):
  required_inputs: dict[str, type] = {}
  optional_inputs: dict[str, tuple[type, object]] = {}

  def __init__(self, **kwargs):
    # Validate required inputs
    for key, expected_type in self.required_inputs.items():
      if key not in kwargs:
        raise ValueError(f"Missing required input: '{key}'")
      if not isinstance(kwargs[key], expected_type):
        raise TypeError(
          f"Input '{key}' must be of type {expected_type.__name__}, "
          f"got {type(kwargs[key]).__name__}"
        )
    
    # Handle optional inputs
    for key, (expected_type, default_value) in self.optional_inputs.items():
      if key in kwargs:
        if not isinstance(kwargs[key], expected_type):
          raise TypeError(
            f"Optional input '{key}' must be of type {expected_type.__name__}, "
            f"got {type(kwargs[key]).__name__}"
          )
      else:
        setattr(self, key, default_value)

    # Store kwargs as attributes
    for key, value in kwargs.items():
      setattr(self, key, value)
    
    self.on_init()
    # self.serialise()

  @classmethod
  def reload(cls, path: Path, permit_version_changes: bool = False):
    if not path.exists():
      raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
      if "git_version" not in f.attrs:
        raise ValueError("HDF5 file missing required 'git_version' attribute.")

      file_git_hash = f.attrs["git_version"]

      # Get local git hash
      repo = git.Repo(search_parent_directories=True)
      local_git_hash = repo.head.commit.hexsha

      if file_git_hash != local_git_hash:
        msg = (
          f"Git version mismatch: file has {file_git_hash}, "
          f"local repo is {local_git_hash}"
        )
        if permit_version_changes:
          import warnings
          warnings.warn(msg, UserWarning)
        else:
          raise ValueError(msg)

      # Collect all non-version attributes
      init_kwargs = {}
      for key, val in f.attrs.items():
        if key == "git_version":
          continue
        init_kwargs[key] = val

    return cls(**init_kwargs)

  @abstractmethod
  def serialise(self, path: Path):
    """Serialise the instance into a HDF5 file."""
    raise NotImplementedError
