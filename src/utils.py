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



@dataclass
class PixelCoordinate():
	"""
	A 2D coordinate index class for indexing images.
	`x` and `y` values correspond to OpenCV axes.
	"""
	x: int = -1
	y: int = -1

	def __bool__(self) -> bool:
		"""Boolean check if the class has been initialized."""
		return self.x != -1 and self.y != -1



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
	

def to_comma_separated_string(arr: Union[np.ndarray, int]) -> str:
	"""
	Convert a pixel value to a comma-separated string.
	"""
	if isinstance(arr, np.ndarray):
		if len(arr) == 0:
			return ''

	elems = str(arr).split()
	
	## Safeguard against numpy array string representation inserting spaces before first element
	if elems[0] in ['[', '(']:
		paren = elems.pop(0)
		elems[0] = paren + elems[0]
	
	## Same as above, just in case for the last element
	if elems[-1] in [']', ')']:
		paren = elems.pop()
		elems[-1] = elems[-1] + paren
	
	return ','.join(elems)


def is_binary_image(img: np.ndarray) -> bool:
	"""
	Check whether an image is binary (contains only fully white and fully black pixels).
	Shorthand function.
	"""
	un = np.unique(img)
	return np.array_equal(un, [0, 255])



