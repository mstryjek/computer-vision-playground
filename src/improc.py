import numpy as np
import cv2

from config import Config
from utils import to_kernel

from typing import Any, List


class ImageProcessor():
	"""
	Comprehensive image processing class, containing all logic & vision algorithms.
	You can implement your own methods for image processing here quite easily.
	You can use the `self.CFG` variable already existing here, as well as some simpler methods
	provided by this class.
	You can also build on the existing methods to create your own image processing algorithm.
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


	def to_grayscale(self, img: np.ndarray) -> np.ndarray:
		"""
		Convert a BGR image to grayscale.
		"""
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	def equalize(self, img: np.ndarray) -> np.ndarray:
		"""
		Equalize a a grayscale image, or a BGR image along the value channel.
		"""
		if len(img.shape) == 2:
			return cv2.equalizeHist(img)
		elif len(img.shape) == 3:
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
			return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


	def smooth(self, img: np.ndarray) -> np.ndarray:
		"""
		Blur an image to reduce noise.
		"""
		kernel = to_kernel(self.CFG.BLUR_SIZE)
		return cv2.GaussianBlur(img, kernel, 0)


	def erode(self, img: np.ndarray) -> np.ndarray:
		"""
		Perform an erosion operation on the image.
		The image must already be grayscale.
		Parameters for all morhpology operations are stored in config.
		Note that this operation uses the default OpenCV kernel shape, if you need a different one
		you might need to create it, for example in `__init__()`.
		"""
		assert len(img.shape) == 2, 'Morphology can only be performed on grascale images!'

		ksize = to_kernel(self.CFG.EROSION.KERNEL_SIZE)
		return cv2.morphologyEx(img, cv2.MORPH_ERODE, ksize, iterations=self.CFG.EROSION.ITERATIONS)


	def dilate(self, img: np.ndarray) -> np.ndarray:
		"""
		Perform a dilation operation on the image.
		The image must already be grayscale.
		Parameters for all morhpology operations are stored in config.
		Note that this operation uses the default OpenCV kernel shape, if you need a different one
		you might need to create it, for example in `__init__()`.
		"""
		assert len(img.shape) == 2, 'Morphology can only be performed on grascale images!'

		ksize = to_kernel(self.CFG.DILATION.KERNEL_SIZE)
		return cv2.morphologyEx(img, cv2.MORPH_DILATE, ksize, iterations=self.CFG.DILATION.ITERATIONS)


	def open(self, img: np.ndarray) -> np.ndarray:
		"""
		Perform an opening operation on the image.
		The image must already be grayscale.
		Parameters for all morhpology operations are stored in config.
		Note that this operation uses the default OpenCV kernel shape, if you need a different one
		you might need to create it, for example in `__init__()`.
		"""
		assert len(img.shape) == 2, 'Morphology can only be performed on grascale images!'

		ksize = to_kernel(self.CFG.OPENING.KERNEL_SIZE)
		return cv2.morphologyEx(img, cv2.MORPH_OPEN, ksize, iterations=self.CFG.OPENING.ITERATIONS)


	def close(self, img: np.ndarray) -> np.ndarray:
		"""
		Perform a closing operation on the image.
		The image must already be grayscale.
		Parameters for all morhpology operations are stored in config.
		Note that this operation uses the default OpenCV kernel shape, if you need a different one
		you might need to create it, for example in `__init__()`.
		"""
		assert len(img.shape) == 2, 'Morphology can only be performed on grascale images!'

		ksize = to_kernel(self.CFG.CLOSING.KERNEL_SIZE)
		return cv2.morphologyEx(img, cv2.MORPH_CLOSE, ksize, iterations=self.CFG.CLOSING.ITERATIONS)


	def thresh(self, img: np.ndarray) -> np.ndarray:
		"""
		Perform a binary thresholding operation on a grayscale image.
		"""
		assert len(img.shape) == 2, 'Thresholding can only be performed on grayscale images!'

		return cv2.threshold(img, self.CFG.THRESHOLD, 255, cv2.THRESH_BINARY)[1]


	def otsu(self, img: np.ndarray) -> np.ndarray:
		"""
		Perform Otsu thresholding on a grayscale image.
		"""
		assert len(img.shape) == 2, 'Thresholding can only be performed on grayscale images!'

		return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


	def remove_small_blobs(self, img: np.ndarray) -> np.ndarray:
		"""
		Remove all blobs (contours) below a given surface area.
		Expects a binary image.
		Area threshold value can be found in config.
		Returns a binary image with only contours larger than the specified area on it.
		Note that the area threshold might not be equal to the actual number of pixels comprising
		the contour (see https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga2c759ed9f497d4a618048a2f56dc97f1).
		"""
		ret = np.zeros(img.shape, dtype=np.uint8)

		cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		cnts_valid = [cnt for cnt in cnts if cv2.contourArea(cnt) >= self.CFG.BLOB_AREA_THRESH]

		return cv2.drawContours(ret, cnts_valid, -1, (255, 255, 255), -1)


	def separate_largest_contour(self, mask: np.ndarray) -> np.ndarray:
		"""
		Remove all contours except the largest one from the image.
		Expects a binary image.
		"""
		ret = np.zeros(mask.shape, dtype=np.uint8)

		cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		cnt = max(cnts, key=cv2.contourArea)

		return cv2.drawContours(ret, [cnt], -1, (255, 255, 255), -1)


	def get_largest_contours(self, img: np.ndarray, n: int) -> List[np.ndarray]:
		"""
		Return `n` largest contours (by contour area) in an image.
		Expects a binary image.
		"""
		contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		contours = sorted(contours, key=cv2.contourArea)

		return contours[-n:]


	def contour_bounding_rects(self, cnts: List[np.ndarray]) -> List[np.ndarray]:
		"""
		Convert contours to their min area rects, in drawable format.
		"""
		boxes = [cv2.minAreaRect(cnt) for cnt in cnts]
		rects = [np.int0(cv2.boxPoints(box)) for box in boxes]

		return rects


	def gray_to_BGR(self, img: np.ndarray) -> np.ndarray:
		"""
		Convert single-channel image to triple-channel image.
		"""
		return np.stack([img, img, img], axis=-1)