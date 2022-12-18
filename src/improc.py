import numpy as np
import cv2

from config import Config
from utils import to_kernel

from typing import Any, List, Tuple


from template import CardType, CardTemplate, Color


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
			CardTemplate('./templates/6.jpg',  CardType._6),
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


	def adaptive_thresh(self, img: np.ndarray) -> np.ndarray:
		"""
		Perform adaptive thresholding.
		"""
		assert len(img.shape) == 2, 'Thresholding can only be performed on grayscale images!'

		return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


	def contrast(self, img: np.ndarray) -> np.ndarray:
		"""
		
		"""
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		cl, ca, cb = cv2.split(lab)

		clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(32, 32))
		cl = clahe.apply(cl)

		lab = cv2.merge((cl, ca, cb))

		return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


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


	def keep_largest_contours(self, img: np.ndarray, n: int) -> np.ndarray:
		"""
		Remove all contours except `n` largest ones (by area).
		Expects a binary image.
		"""
		cnts = self.get_largest_contours(img, n)

		ret = np.zeros_like(img)
		return cv2.drawContours(ret, cnts, -1, (255,), -1)


	def keep_largest_contours_with_holes(self, img: np.ndarray, n: int) -> np.ndarray:
		"""
		Remove all contours except `n` largest ones (by area), keeping their holes (enclosed contours).
		Expects a binary image.
		"""
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		areas = [cv2.contourArea(c) for c in contours]
		largest_cnt_indices = np.argsort(areas)[::-1]

		non_hole_cnts = [
			i for i, (_, _, _, parent)
			in enumerate(hierarchy[0, :, :])
			if parent == -1
		]

		largest_outlines = []

		cnt = 0
		i = 0
		while cnt < n:
			if i == len(largest_cnt_indices):
				break
			idx = largest_cnt_indices[i]
			if idx in non_hole_cnts:
				largest_outlines.append(idx)
				cnt += 1
			i += 1


		hole_indices = [
			i
			for i, (_, _ , _, parent)
			in enumerate(hierarchy[0, :, :])
			if parent in largest_outlines
		]

		contours_to_keep = [
			c for i, c in enumerate(contours) 
			if i in largest_outlines
			or i in hole_indices
		]

		res = np.zeros_like(img)
		return cv2.drawContours(res, contours_to_keep, -1, (255,), -1)




	def remove_contours_touching_borders(self, img: np.ndarray) -> np.ndarray:
		"""
		Remove any contours touching any of the window borders.
		"""
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		h, w = img.shape[:2]

		contours = ([
			c for c in contours
			if not
				(
					np.any(c == 0)
					or
					np.any(c[:, :, 1] == h-1)
					or
					np.any(c[:, :, 0] == w-1)
				)
		])

		return cv2.drawContours(np.zeros(img.shape, dtype=np.uint8), contours, -1, (255,), -1)


	def remove_contours_touching_borders_or_background(self, img: np.ndarray) -> np.ndarray:
		"""
		
		"""
		if self.has_white_bg(img):
			img = self.erode(img)
			return self.remove_white_background(img)
		else:
			return self.remove_contours_touching_borders(img)


	def remove_white_background(self, img: np.ndarray) -> np.ndarray:
		"""
		Remove white background from an image.
		"""
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		areas = [cv2.contourArea(c) for c in contours]
		largest_cnt_idx = np.argmax(areas)

		holes = [
			i
			for i, (_, _, _, parent)
			in enumerate(hierarchy[0, :, :])
			if parent == largest_cnt_idx
		]

		holes_areas = [areas[i] for i in holes]
		order  = np.argsort(holes_areas)
		largest_hole = holes[order[-1]]

		contours = [c for i, c in enumerate(contours) if i not in [largest_cnt_idx, largest_hole]]

		res = np.zeros_like(img)
		return cv2.drawContours(res, contours, -1, (255,), -1)



	def has_white_bg(self, img: np.ndarray) -> bool:
		"""
		Check whether the images has white background.
		"""
		size = np.prod(img.shape)

		nonzero = np.count_nonzero(img)

		return nonzero/size >= self.CFG.BG_PERC_THRESH



	def crop_image_center(self, img: np.ndarray) -> np.ndarray:
		"""
		Crop image to the center.
		"""
		h, w = img.shape[:2]
		fac_x = int(w * self.CFG.CROP.PERC_X)
		fac_y = int(h * self.CFG.CROP.PERC_Y)

		return img[fac_y:h-fac_y, fac_x:w-fac_x]




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
		maxes = []
		locs  = []

		maxes_inv = []
		locs_inv  = []

		## TODO Value inversion
		## TODO Image flip if upside down

		for tmpl in self.templates:
			kernel = tmpl.get_kernel()
			kernel_inv = cv2.rotate(kernel, cv2.ROTATE_180)

			match     = cv2.matchTemplate(img, kernel, 	   cv2.TM_CCOEFF)
			match_inv = cv2.matchTemplate(img, kernel_inv, cv2.TM_CCOEFF)

			_, max_, _, (x, y)            = cv2.minMaxLoc(match)
			_, max_inv, _, (x_inv, y_inv) = cv2.minMaxLoc(match_inv)

			maxes.append(max_)
			locs.append((int(x+kernel.shape[1]/2), int(y+kernel.shape[0]/2)))

			maxes_inv.append(max_inv)
			locs_inv.append((int(x_inv+kernel.shape[1]/2), int(y_inv+kernel.shape[0]/2)))


		isinv = np.max(maxes) <= np.max(maxes_inv)

		if isinv:
			idx = np.argmax(maxes)
			locs_ = locs
		else:
			idx = np.argmax(maxes_inv)
			locs_ = locs_inv

		return self.templates[idx].get_name(), locs_[idx], isinv


	def count_holes_in_largest_contour(self, img: np.ndarray) -> int:
		"""
		Count holes in largest contour in image.
		Expects a binary image.
		"""
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		areas = [cv2.contourArea(cnt) for cnt in contours]
		largest_cnt_index = np.argmax(areas)

		holes = [
			parent
			for next_, prev, child, parent
			in hierarchy[0, :, :]
			if parent == largest_cnt_index
		]

		return len(holes)

	
	def largest_contour_vh_aspect_ratio(self, img: np.ndarray) -> float:
		"""
		Calculate the vertical to horizontal aspect ratio of the largest contour (by area) in the image.
		Expects a binary image.
		"""
		contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		cnt = np.squeeze(sorted(contours, key=cv2.contourArea)[-1])

		v = np.max(cnt[:, 1]) - np.min(cnt[:, 1])
		h = np.max(cnt[:, 0]) - np.min(cnt[:, 0])

		return v/h


	def is_contour_wider_at_top(self, img: np.ndarray, margin: int = 4) -> bool:
		"""
		Check whether the wider horizontal edge of a contour is above (True) or below (False) its center.
		Consider `margin` rows to make allowances for any slant in contour edges.
		"""
		cnt ,= self.get_largest_contours(img, 1)

		cnt_img = np.zeros_like(img)

		cnt_img = cv2.drawContours(cnt_img, [cnt], -1, (255,), -1)

		cnt = np.squeeze(cnt)
		y_max = np.max(cnt[:, 1])
		y_min = np.min(cnt[:, 1])

		top = cnt_img[y_min:y_min+margin, :]
		bottom = cnt_img[y_max-margin:y_max, :]

		return np.count_nonzero(top) >= np.count_nonzero(bottom)


	def is_contour_hole_above_center(self, img: np.ndarray) -> bool:
		"""
		Check whether the only hole in the contour is above (True) or below (False) its center of gravity.
		Expects a binary image, where the largest contour has only one hole.
		"""
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		areas = [cv2.contourArea(cnt) for cnt in contours]
		largest_cnt_index = np.argmax(areas)

		hole_idx = [
			i
			for i, (next_, prev, child, parent)
			in enumerate(hierarchy[0, :, :])
			if parent == largest_cnt_index
		][0]

		cnt = contours[largest_cnt_index]
		hole_cnt = contours[hole_idx]
		
		_, y = self.get_contour_center(cnt)
		_, hole_y = self.get_contour_center(hole_cnt)

		return hole_y <= y
	

	@staticmethod
	def get_contour_center(cnt: np.ndarray) -> Tuple[int, int]:
		"""
		Get the center of gravity of a contour as (x, y).
		"""
		M = cv2.moments(cnt)

		try:
			x = int(M['m10'] / M['m00'])
			y = int(M['m01'] / M['m00'])
			return (x, y)
		except ZeroDivisionError:
			return None, None


	def remove_small_holes(self, img: np.ndarray) -> np.ndarray:
		"""
		Remove all holes in the largest contour that are smaller than the defined threshold.
		Expects a binary image.
		"""
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		areas = [cv2.contourArea(cnt) for cnt in contours]
		largest_cnt_index = np.argmax(areas)

		holes = [
			i
			for i, (next_, prev, child, parent)
			in enumerate(hierarchy[0, :, :])
			if parent == largest_cnt_index
		]

		holes_to_remove = [
			i for i in holes
			if cv2.contourArea(contours[i]) < self.CFG.HOLE_AREA_THRESH	
		]
		
		contours = [cnt for i, cnt in enumerate(contours) if i not in holes_to_remove]

		res = np.zeros_like(img)
		res = cv2.drawContours(res, contours, -1, (255,), -1)

		return res
	

	def has_n_large_contours(self, img: np.ndarray, n: int) -> bool:
		"""
		Check whether a binary image has at least n contours above the specified area.
		"""
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		areas = [cv2.contourArea(cnt) for cnt in contours]
		largest_cnt_index = np.argmax(areas)

		holes = [
			i
			for i, (next_, prev, child, parent)
			in enumerate(hierarchy[0, :, :])
			if parent == largest_cnt_index
		]

		contours = [c for i, c in enumerate(contours) if cv2.contourArea(c) > self.CFG.BLOB_AREA_THRESH and i not in holes]

		return len(contours) >= n


	def is_second_largest_contour_above_first(self, img: np.ndarray) -> bool:
		"""
		Check whether the second largest contour in a binary image is above (True) or below (False) the largest contour.
		"""
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


		areas = [cv2.contourArea(cnt) for cnt in contours]
		largest_cnt_index = np.argmax(areas)

		holes = [
			i
			for i, (next_, prev, child, parent)
			in enumerate(hierarchy[0, :, :])
			if parent == largest_cnt_index
		]

		contours  = [c for i, c in enumerate(contours) if i not in holes]

		contours = sorted(contours, key=cv2.contourArea)[::-1][:2]

		moments = [cv2.moments(c) for c in contours]

		y_values = [int(m['m01'] / m['m00']) for m in moments]

		return y_values[1] <= y_values[0]


	def classify_card(self, img: np.ndarray) -> CardType:
		"""
		Classify an UNO symbol.
		Expects a warped, binary image with most noise removed, and the largest contour being the symbol.
		"""
		holes = self.count_holes_in_largest_contour(img)

		if holes == 2:
			ar = self.largest_contour_vh_aspect_ratio(img)
			if ar >= self.CFG.ASPECT_RATIO_THRESH:
				return CardType._8
			else:
				return CardType.BLOCK
		elif holes == 1:
			hole_above = self.is_contour_hole_above_center(img)
			if self.has_n_large_contours(img, 2):
				line_above = self.is_second_largest_contour_above_first(img)
			else:
				line_above = self.is_contour_wider_at_top(img, 2)
			if line_above == hole_above:
				return CardType._6
			else:
				return CardType._9
		elif holes == 0:
			return CardType.PLUS_2
		else:
			raise ValueError(f'Found contour with invalid number of holes: {holes}')

	
	def get_mean_nonzero_hue(self, img: np.ndarray) -> int:
		"""
		Get mean hue of the image, ignoring black, gray and white values.
		"""
		hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
		h = hsv[:, :, 0]
		s = hsv[:, :, 1]

		return int(np.mean(h[np.logical_and(h != 0, s > 0.3)]))
		

	def get_color(self, img: np.ndarray) -> Color:
		"""
		Get card color.
		"""
		h = self.get_mean_nonzero_hue(img)

		# colors = [
		# 	Color.RED,
		# 	Color.GREEN,
		# 	Color.BLUE,
		# 	Color.YELLOW
		# ]

		if h > 224:
			return Color.RED
		elif h > 190:
			return Color.BLUE
		elif h > 160:
			return Color.GREEN
		else:
			return Color.YELLOW

