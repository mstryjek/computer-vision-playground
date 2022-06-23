"""
TODO Desc

KF
"""
import numpy as np
import cv2

from collections import deque

from config import Config

from typing import Tuple, Any


class ImageProcessor():
	"""
	Comprehensive image processing class, containing all logic & vision algorithms.
	"""
	def __init__(self, cfg: Config) -> None:
		self.CFG = cfg

		## Lower HSV threshold for skin area detection
		self.skin_bounds_low = np.array([
			self.CFG.PROCESSING.COLOR_BOUNDS.LOW.H,
			self.CFG.PROCESSING.COLOR_BOUNDS.LOW.S,
			self.CFG.PROCESSING.COLOR_BOUNDS.LOW.V
		], dtype=np.uint8)

		## Upper HSV threshold for skin area detection
		self.skin_bounds_high = np.array([
			self.CFG.PROCESSING.COLOR_BOUNDS.HIGH.H,
			self.CFG.PROCESSING.COLOR_BOUNDS.HIGH.S,
			self.CFG.PROCESSING.COLOR_BOUNDS.HIGH.V
		], dtype=np.uint8)

		self.deq = deque()



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
		return cv2.GaussianBlur(img, (self.CFG.PROCESSING.BLUR_SIZE,)*2, 0)



	def extract_color_mask(self, img: np.ndarray) -> np.ndarray:
		"""
		Extract a denoised, filtered version of the skin color mask.
		"""
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(hsv, self.skin_bounds_low, self.skin_bounds_high)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.CFG.PROCESSING.CLOSE_SIZE,)*2)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.CFG.PROCESSING.CLOSE_ITERS)

		return mask


	def extract_largest_contour(self, mask: np.ndarray) -> np.ndarray:
		"""
		Extract the contour with the largest area.
		"""
		contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		areas = [cv2.contourArea(cnt) for cnt in contours]
		filtered_mask = np.zeros(mask.shape, dtype=np.uint8)

		if areas == []:
			return filtered_mask

		cnt_max_area = contours[np.argmax(areas)]
		return cv2.drawContours(filtered_mask, [cnt_max_area], -1, (255, 255, 255), -1)


	def inscribe_circle(self, mask: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
		"""
		Remove center of the hand from the binary image
		"""
		distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
		# cv2.normalize(distance, distance, 0, 1.0, cv2.NORM_MINMAX)
		_, max_val, _, center = cv2.minMaxLoc(distance)
		radius = max_val*self.CFG.PROCESSING.CIRCLE_SCALE
		circle = cv2.circle(mask.copy(), center, int(radius), (0, 0, 0), -1)

		return circle, radius, center


	def remove_wrist(self, mask: np.ndarray, radius: float, center: Tuple[int, int]) -> np.ndarray:
		"""
		Remove contour of wrist from the binary image
		"""
		contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
			
		#Calculating the center of mass in Y coordinates for all contours
		mass_center = []
		for cnt in contours:
			M = cv2.moments(cnt)
			if M["m00"] != 0:
				cY = int(M["m01"] / M["m00"])
			else :
				cY = 0
			mass_center.append(cY)
		#Wrist contour is presumed to be the one below the center of the hand
		finger_cnts = [contours[i] for i in range(len(contours)) if mass_center[i] < center[1] + radius/2]
		return cv2.drawContours(filtered_mask, finger_cnts, -1, (255, 255, 255), -1)


	def remove_bent_fingers(self, mask: np.ndarray) -> np.ndarray :
		"""
		Sort out bent fingers from the binary image
		"""
		contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		areas = [cv2.contourArea(cnt) for cnt in contours]
		filtered_mask = np.zeros(mask.shape, dtype=np.uint8)

		if areas == []:
			return filtered_mask
	
		tresh = max(areas)*self.CFG.PROCESSING.SURF_RATIO

		# print("tresh: ", tresh)
		# print("max: ", max_area)
		# print(len(areas))
		# print ("areas ", areas)

		proper_finger_cnts = [contours[i] for i in range(len(contours)) if areas[i] > tresh]

		return cv2.drawContours(filtered_mask, proper_finger_cnts, -1, (255, 255, 255), -1)


	def remove_small_blobs(self, mask: np.ndarray) -> np.ndarray:
		"""
		Remove small blobs.
		"""
		ret = np.zeros(mask.shape, dtype=np.uint8)
		cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		cnts_valid = [cnt for cnt in cnts if cv2.contourArea(cnt) >= self.CFG.PROCESSING.BLOB_AREA_THRESH]
		return cv2.drawContours(ret, cnts_valid, -1, (255, 255, 255), -1)


	def count_fingers(self, mask: np.ndarray) -> int:
		"""
		Count valid blobs in a binary image.
		"""
		cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		self.deq.append(len(cnts))
		if len(self.deq) >= self.CFG.PROCESSING.NUM_FRAMES:
			self.deq.popleft()

		return np.round(np.mean(self.deq), 0)
