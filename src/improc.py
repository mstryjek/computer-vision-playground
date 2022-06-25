import numpy as np
import cv2

from config import Config

from typing import Any


class ImageProcessor():
	"""
	Comprehensive image processing class, containing all logic & vision algorithms.
	"""
	def __init__(self, cfg: Config) -> None:
		self.CFG = cfg



	def __enter__(self):
		"""
		Enter context.
		"""
		return self
	

	def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
		pass


	def equalize(self, img: np.ndarray) -> np.ndarray:
		"""
		Equalize a BGR image along the value channel.
		"""
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
		return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


	def smooth(self, img: np.ndarray) -> np.ndarray:
		"""
		Blur an image to reduce noise.		
		"""
		kernel = (self.CFG.BLUR_SIZE,)*2 if isinstance(self.CFG.BLUR_SIZE, int) else self.CFG.BLUR_SIZE
		return cv2.GaussianBlur(img, kernel, 0)


	def remove_small_blobs(self, mask: np.ndarray) -> np.ndarray:
		"""
		Remove small blobs.
		"""
		ret = np.zeros(mask.shape, dtype=np.uint8)
		cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		cnts_valid = [cnt for cnt in cnts if cv2.contourArea(cnt) >= self.CFG.BLOB_AREA_THRESH]
		return cv2.drawContours(ret, cnts_valid, -1, (255, 255, 255), -1)

