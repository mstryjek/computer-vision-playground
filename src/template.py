import numpy as np
import cv2
from enum import Enum
from dataclasses import dataclass

class CardType(Enum):
	_6     = 'SIX'
	_8     = 'EIGHT'
	_9     = 'NINE'
	PLUS_2 = 'PLUS_2'
	BLOCK  = 'BLOCK'
	WRONG  = 'WRONG'


class Color(Enum):
	RED    = [0, 0, 255]
	BLUE   = [255, 0, 0]
	GREEN  = [0, 255, 0]
	YELLOW = [0, 255, 255]



@dataclass(eq=False, repr=True, init=False, order=False)
class CardTemplate():
	def __init__(self, kernel_path: str, card_type: CardType) -> None:
		self._type = card_type
		self.kernel = cv2.imread(kernel_path, cv2.IMREAD_GRAYSCALE)


	def get_type(self) -> CardType:
		return self._type


	def get_name(self) -> str:
		return self._type.value


	def get_kernel(self) -> np.ndarray:
		return self.kernel

