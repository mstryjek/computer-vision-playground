import numpy as np
import cv2
from enum import Enum


class CardType(Enum):
	_6     = 'SIX'
	_8     = 'EIGHT'
	_9     = 'NINE'
	PLUS_2 = 'PLUS_2'
	BLOCK  = 'BLOCK'



class CardTemplate():
	def __init__(self, kernel_path: str, card_type: CardType) -> None:
		self._type = card_type
		self.kernel = cv2.imread(kernel_path, cv2.IMREAD_GRAYSCALE)


	def get_name(self) -> str:
		return self._name


	def get_kernel(self) -> np.ndarray:
		return self.kernel

