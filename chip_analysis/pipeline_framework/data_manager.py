import copy

from typing import Union

data_managers = {}

class DataManager:
  def __init__(self):
    self._results = {}
  
  def add(self, id, data):
    self._results[id] = data
  
  def contains(self, id):
    return id in self._results
  
  def get(self, id):
    if not self.contains(id):
      raise KeyError(f"Data with id {id} not found.")
    return copy.deepcopy(self._results[id])
  
  def registered_results(self):
    return list(self._results.keys())
  
  def register(self, id: Union[str, dict], data = None):
    if isinstance(id, str): self._register_individual(id, data)
    elif isinstance(id, dict): self._register_bulk(id)
    else: raise TypeError(f"id must be a string or a dict. Instead got {type(id)}.")
  
  def _register_bulk(self, data_dict: dict):
    for id in data_dict.keys():
      if isinstance(id, str) is False:
        raise TypeError("All keys in data_dict must be strings.")
    for id, data in data_dict.items():
      self._register_individual(id, data)
  
  def _register_individual(self, id: str, data):
    if self.contains(id):
      raise KeyError(f"Data with id {id} already exists.")
    self._results[id] = data

  def serialize(self):
    pass

data_managers["native"] = DataManager