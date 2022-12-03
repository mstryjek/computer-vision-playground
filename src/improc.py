import numpy as np
import cv2

from config import Config
from utils import to_kernel

from typing import Any, List, Tuple


from template import CardType, CardTemplate


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
		self.templates = [
			CardTemplate('./templates/6.png',  CardType._6),
			CardTemplate('./templates/8.png',  CardType._8),
			CardTemplate('./templates/9.png',  CardType._9),
			CardTemplate('./templates/b.png', CardType.BLOCK),
			CardTemplate('./templates/p2.png',  CardType.PLUS_2)
		]


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


	def _euclidian_distance(self, pt1: np.ndarray, pt2: np.ndarray) -> float:
		"""
		Calculate the euclidian distance between two points.
		"""
		return np.sqrt(
			np.sum(
				(pt1 - pt2)**2.
			)
		)


	def _assert_box_starts_topleft_clockwise_upright(self, box: np.ndarray) -> np.ndarray:
		"""
		Change box points order if it does not start at the top left point or the order of the points is not clockwise.
		Also make sure longer edge of the rectangle is between vertices 1 and 2, as well as 3 and 0.
		"""
		mean = np.mean(box, axis=0)

		out = [None]*4

		for pt in box:
			i = None
			
			if pt[0] <= mean[0] and pt[1] < mean[1]:
				i = 0
			elif pt[0] <= mean[0] and pt[1] >= mean[1]:
				i = 3
			elif pt[0] > mean[0] and pt[1] < mean[1]:
				i = 1
			elif pt[0] > mean[0] and pt[1] >= mean[1]:
				i = 2
			
			out[i] = pt

		out = np.array(out).astype(np.float32)
		return out


	def warp_contours(self, img: np.ndarray, boxes: List[np.ndarray]) -> List[np.ndarray]:
		"""
		Warp portions of the image given by bounding boxes. Return warped portions.
		"""
		ret = []

		target_rect = np.array([
			[0, 0],
			[self.CFG.WARP.TARGET_SHAPE[1], 0],
			self.CFG.WARP.TARGET_SHAPE[::-1],
			[0, self.CFG.WARP.TARGET_SHAPE[0]],
		]).astype(np.float32)

		for box in boxes:

			warp_mat = cv2.getPerspectiveTransform(
				self._assert_box_starts_topleft_clockwise_upright(box), 
				target_rect
			)

			warped = cv2.warpPerspective(img, warp_mat, tuple(self.CFG.WARP.TARGET_SHAPE)[::-1])

			ret.append(warped)

		return ret


	def match_templates_to_image(self, img: np.ndarray) -> Tuple[str, Tuple[int, int], bool]:
		"""
		Match templates to image by convolution.
		Returns matched card type name, (x, y) coordinates of matched template and whether the template was found upside down.
		"""
		inv = np.rot90(img, k=2)

		maxes = []
		locs  = []

		maxes_inv = []
		locs_inv  = []

		## TODO Value inversion
		## TODO Image flip if upside down

		for tmpl in self.templates:
			kernel = tmpl.get_kernel()
			
			match     = cv2.matchTemplate(img, kernel, cv2.TM_CCOEFF_NORMED)
			match_inv = cv2.matchTemplate(inv, kernel, cv2.TM_CCOEFF_NORMED)

			_, max_, _, (x, y)            = cv2.minMaxLoc(match)
			_, max_inv, _, (x_inv, y_inv) = cv2.minMaxLoc(match_inv)

			maxes.append(max_)
			locs.append((int(x+kernel.shape[1]/2), int(y+kernel.shape[0]/2)))
		
			maxes_inv.append(max_inv)
			locs_inv.append((int(x_inv+kernel.shape[1]/2), int(y_inv+kernel.shape[0]/2)))
		

		isinv = np.max(maxes) < np.max(maxes_inv)

		if isinv:
			idx = np.argmax(maxes)
			locs_ = locs
		else:
			idx = np.argmax(maxes_inv)
			locs_ = locs_inv

		return self.templates[idx].get_name(), locs_[idx], isinv





