"""
Config parser module. Contains config parser class as well as utility functions
and helpful type aliases. Meant to read in config from a `.yml` file.
"""
from __future__ import annotations ## DO NOT REMOVE, allows for cross-types annotations within the same class

import yaml
import glob

from typing import Dict, Union, Tuple, List, Optional


def find_cfg_file_path() -> Union[str, None]:
	"""Automatically find config file."""
	yaml_files = glob.glob('*/*.yml', recursive=True) + glob.glob('*/*.yaml', recursive=True)
	if len(yaml_files) == 0:
		return None
	return yaml_files[0]



class Config(dict):
	"""
	Config class that is internally a dict but allows accessing items by
	using `Config.ITEM` (as well as `Config['ITEM']`).
	"""
	def __init__(self, cfg_path: str = None, *args, **kwargs) -> None:
		super(Config, self).__init__()

		## Remember the common config dict to add to direct subconfigs
		common = None

		for key, value in kwargs.items():
			if key == 'COMMON':
				common = value

			self[key] = value

		if cfg_path is not None:
			self._path = cfg_path

		self._add_common_to_subdicts(common)


	def _add_common_to_subdicts(self, common: Optional[Config]) -> None:
		"""
		Add the `COMMON` config subdict to all direct subdicts of the current subdict.
		This should be done after key, value loading in `__init__()`, the `COMMON` config would otherwise
		be propagated to all branches of the config tree, not just the ones directly at the root.
		"""
		if common is not None:
			for value in self.values():
				if isinstance(value, Config):
					value.COMMON = common


	def __getattr__(self, key: str) -> ConfigItem:
		"""
		Get a config item by `.` access.
		"""
		retval, ok = self._recursive_search(key)
		if ok:
			return retval
		raise KeyError(f'Tried to access {key}, which is not in config.')


	def __setattr__(self, key: str, value: ConfigItem) -> None:
		"""Dunder overload for `.` dict indexing."""
		self[key] = value


	def __setitem__(self, key: str, value: ConfigItem) -> None:
		"""Dunder overload to ensure `.` indexing within subdicts."""
		if isinstance(value, dict):
			value = Config(**value)

		super(Config, self).__setitem__(key, value)


	def _recursive_search(self, key: str) -> Tuple[Union[None, ConfigItem], bool]:
		"""
		Search recursively for a key if it was not found as an immediate member variable of this object.
		This is for convenience.

		Args:
		- `key`: Attribute name to find

		Returns:
		- `None` if the attribute was never find
		- an instance of `Config`
		"""
		## Return key if it's directly in this config
		if key in self.keys():
			return self[key], True

		## Find any sub-configs in this config
		subcfgs = []
		for value in self.values():
			if isinstance(value, Config):
				subcfgs.append(value)

		## Check if any of the subconfigs have the key directly of indirectly in them
		for subcfg in subcfgs:
			retval, ok = subcfg._recursive_search(key)
			if ok:
				return retval, True

		return None, False


	def dict(self) -> Dict[str, YAMLCompatible]:
		"""Convert self to dict safely."""
		retdict = {}

		for key, value in self.items():
			if key.startswith('_'): ## Ignore private keys. Currently not implemented but leaving this in for future compatibility
				continue
			elif isinstance(value, Config):
				retdict[key] = value.dict()
			else:
				retdict[key] = value

		return retdict


	@classmethod
	def from_file(cls, path: str = None) -> Config:
		"""
		Read in config from the specified file.
		"""
		if path is None:
			return cls.auto()

		with open(path, 'r') as cf:
			cfg_dict = yaml.safe_load(cf)

		return cls(cfg_path=path, **cfg_dict)


	@classmethod
	def auto(cls) -> Config:
		"""
		Find config file automatically and create the config object from it.
		"""
		cfg_file = find_cfg_file_path()
		if cfg_file is None:
			raise TypeError('Couldn\'t find config file. Make sure there is a config.yml in the working tree.')

		return cls.from_file(cfg_file)



## Type alias for any config item, including subconfigs
ConfigItem = Union[int, float, bool, str, None, List, Tuple, Config]
## Type alias for any type that can be read in from YAML
YAMLCompatible = Union[int, float, bool, str, None, List, Tuple, Dict]