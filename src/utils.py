import numpy as np

from dataclasses import dataclass, field

from typing import Iterable, List, Tuple, Union



@dataclass(init=True, eq=False, repr=False)
class ImagesToSave():
	"""
	Simple dataclass with info about images to be saved, along with the processing step index
	and name (if given) that produced that image
	"""
	images: List[np.ndarray] = field(default_factory=list)
	steps: List[int] = field(default_factory=list)
	step_names: List[str] = field(default_factory=list)


	def add(self, img: np.ndarray, step: int, step_name: Union[str, None] = None) -> None:
		"""
		Remember and image and step (with its name) to save.
		"""
		if step not in self.steps:
			self.images.append(img)
			self.steps.append(step)
			self.step_names.append(step_name)


	def __bool__(self) -> bool:
		"""
		Boolean overload for convenience.
		"""
		return len(self.steps) != 0


	def __iter__(self) -> Iterable[Tuple[np.ndarray, int, Union[str, None]]]:
		"""
		Iteration overload for convenience.
		"""
		for i in range(len(self.steps)):
			yield self.images[i], self.steps[i], self.step_names[i]



def pad(num: int, width: int = 4) -> str:
	"""Pad an int with zeros to ensure constant filename length."""
	snum = str(num)
	zeros = '0'*(width - len(snum))
	return zeros + snum


def to_kernel(ksize: Union[int, List[int], Tuple[int, int]]) -> Tuple[int, int]:
	"""
	Convert an arbitrary type convolution kernel representation into 
	a tuple of integers (an OpenCV-compatible representation).
	"""
	if isinstance(ksize, int):
		return (ksize, ksize)
	elif isinstance(ksize, list):
		return tuple(ksize)
	elif isinstance(ksize, tuple):
		return ksize
	else:
		raise NotImplementedError('Utility function to_kernel() supports only ints, lists and tuples!')