"""
Visualizer class for visualizing intermediate steps of image processing.
"""

import cv2
import numpy as np

from math import floor, ceil

from config import Config
from utils import ImagesToSave, PixelCoordinate, to_comma_separated_string, is_binary_image

from typing import Tuple, Optional, Union, Any


class ProcessingVisualizer():
	"""
	Class for visualizing images given to it. Has the capability of drawing text info on the images.
	"""
	def __init__(self, cfg: Config) -> None:
		self.CFG = cfg
		self.reset()
		self.inspect_mode = False
		self.zoom = 1
		self.mouse_position = PixelCoordinate()
		self.zoom_center = PixelCoordinate()
		self.display_pixel_label = False
		self.margin_pixels_left = 0
		self.margin_pixels_top = 0
		self.orig_pixels_left = 0
		self.orig_pixels_top = 0
		self.is_margin_left = False
		self.is_margin_top = False
		self.FONT = cv2.FONT_HERSHEY_SIMPLEX


	def __enter__(self):
		"""Enter context and create widnow."""
		cv2.namedWindow(self.CFG.WINDOW_NAME)
		cv2.setMouseCallback(self.CFG.WINDOW_NAME, self._mouse_callback)
		return self


	def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
		"""Safely exit."""
		cv2.destroyAllWindows()


	def reset(self) -> None:
		"""Reset image list."""
		self.images = []
		self.step_names = []


	def store(self, img: np.ndarray, step_name: Optional[str] = None) -> None:
		"""Store an image to be shown later."""
		self.images.append(img.copy())
		self.step_names.append(step_name)


	def draw_mask(self, img: np.ndarray, mask: np.ndarray, alpha: Optional[float] = None, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
		"""
		Utility function, not part of the system but you can use it if your code produced a single-channel
		mask that you'd like to visualize on an image.
		Draw a mask onto an image using the color and opactiy (alpha) given in config.
		Image and mask must have the same shape, excluding the number of channels.
		Image should have 3 channels and mask should have 1.
		You can add support for drawing on a grayscale image by changing `VISUALIZATION.COLOR` in config to
		be a single value from the 0-255 range.
		"""
		## Safety clauses
		assert img is not None, 'Image is empty!'
		assert mask is not None, 'Mask is empty!'
		assert img.shape[:2] == mask.shape[:2], f'Both images must have the same shape, but image\
			has shape {img.shape[:2]} and mask has shape {mask.shape[:2]}'

		alpha = alpha if alpha is not None else self.CFG.ALPHA
		color = color if color is not None else self.CFG.COLOR

		## Copy of the image to add with alpha weight later
		ret = img.copy()

		## Draw full opacity color on the copy
		ret[mask != 0] = color

		## Return the mask drawn on the image with the given alpha
		return cv2.addWeighted(ret, alpha, img, 1.-alpha, 0)


	def _draw_frame_info(self, frame_id: int, step: int, img: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Draw a label with information about the current frame number, processing step id and processing
		step name (if given). Uses the `ALPHA` value from config.
		"""
		## Copy image to allow for alpha-weighted composition
		draw_img = self.images[step].copy() if img is None else img.copy()
		orig_img = self.images[step].copy() if img is None else img.copy()

		## Bottom label text with current frame index, processing step number and name (if given), as well as zoom amount if more than 1
		text = f'FRAME: {frame_id-1} | STEP {step+1}'
		text = text + f' ({self.step_names[step].upper()})' if self.step_names[step] is not None else text
		text = text + f' [{self.zoom}x]' if self.zoom > 1 else text

		## Text size and total textbox size
		(tw, th), baseline = cv2.getTextSize(text, self.FONT, self.CFG.TEXT.SCALE, 1)
		textbox_size = (th+2*self.CFG.TEXT.MARGIN, tw+2*self.CFG.TEXT.MARGIN)

		## Rectangle top-left corner, text bottom-left corner and image corner (rectangle bottom-right corner)
		rect_org = (draw_img.shape[1] - textbox_size[1] - 1, draw_img.shape[0] - textbox_size[0] - 1)
		txt_org = (draw_img.shape[1] - textbox_size[1] - 1 + self.CFG.MARGIN, draw_img.shape[0] - 1 - self.CFG.MARGIN)
		corner = (draw_img.shape[1] - 1, draw_img.shape[0] - 1)

		## Draw rectangle and text on the image
		draw_img = cv2.rectangle(draw_img, rect_org, corner, self.CFG.TEXT.BG_COLOR, -1)
		draw_img = cv2.putText(draw_img, text, txt_org, self.FONT, self.CFG.TEXT.SCALE, self.CFG.TEXT.COLOR, 1, cv2.LINE_AA)

		## Return an alpha-weighted composition of the original image and the image with the label
		return cv2.addWeighted(draw_img, self.CFG.ALPHA, orig_img, 1.-self.CFG.ALPHA, 0)


	def _mouse_callback(self, event: int, x: int, y: int, flags: Any, params: Any) -> None:
		"""
		Mouse event callback for receiveing mouse move and right button click events.
		"""
		## Ignore events in stream mode
		if not self.inspect_mode:
			return

		## Update mouse position
		if event == cv2.EVENT_MOUSEMOVE:
			self.mouse_position = PixelCoordinate(x, y)
			if self.zoom == 1:
				self.zoom_center = PixelCoordinate(x, y)

		## Toggle pixel label on right button click
		if event == cv2.EVENT_LBUTTONDOWN:
			self.display_pixel_label = not self.display_pixel_label


	def _get_crop_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[bool, bool]]:
		"""
		Get crop bounds for the zoomed part of the image. Returns ((xmin, xmax), (ymin, ymax), (margin_left, margin_top))
		in accordance with OpenCV axes.
		"""
		## Zoomed image portion total height and width (will be clipped slightly to target shape)
		zoomed_h = ceil(self.CFG.COMMON.SHAPE[0] / self.zoom)
		zoomed_w = ceil(self.CFG.COMMON.SHAPE[1] / self.zoom)

		xmin = None
		xmax = None
		margin_left = None

		## Bounding the zoom box off of the left image edge
		if self.zoom_center.x < zoomed_w // 2:
			xmin = 0
			xmax = zoomed_w + 1
			margin_left = False
		## Bouncing bounding box off of the right image edge
		elif self.zoom_center.x > self.CFG.COMMON.SHAPE[1] - zoomed_w // 2:
			xmin = self.CFG.COMMON.SHAPE[1] - zoomed_w
			xmax = self.CFG.COMMON.SHAPE[1]
			margin_left	= True
		## Not close to left/right edge - crop as symetrically as possible
		else:
			margin_left = False
			if zoomed_w % 2 == 1:
				xmin = self.zoom_center.x - (zoomed_w - 1) // 2
				xmax = self.zoom_center.x + (zoomed_w - 1) // 2 + 1
			else:
				xmin = self.zoom_center.x - zoomed_w // 2
				xmax = self.zoom_center.x + zoomed_w // 2 + 1

		ymin = None
		ymax = None
		margin_top = None

		## Bouncing off of the top edge
		if self.zoom_center.y < zoomed_h // 2:
			ymin = 0
			ymax = zoomed_h
			margin_top = False
		## Bouncing the crop box off of the bottom edge
		elif self.zoom_center.y > self.CFG.COMMON.SHAPE[0] - zoomed_h // 2:
			ymin = self.CFG.COMMON.SHAPE[0] - zoomed_h
			ymax = self.CFG.COMMON.SHAPE[0]
		## Not close to top/bottom edge - crop symetrically
		else:
			margin_top = False
			if zoomed_h % 2 == 1:
				ymin = self.zoom_center.y - (zoomed_h - 1) // 2
				ymax = self.zoom_center.y + (zoomed_h - 1) // 2 + 1
			else:
				ymin = self.zoom_center.y - zoomed_h // 2
				ymax = self.zoom_center.y + zoomed_h // 2 + 1

		return ((xmin, xmax), (ymin, ymax), (margin_left, margin_top))


	def _zoom_image(self, img: np.ndarray) -> np.ndarray:
		"""
		Zoom an image with the current zoom factor.
		"""
		if self.zoom == 1:
			return img

		## Bound to crop from the original image
		(xmin, xmax), (ymin, ymax), (margin_left, margin_top) = self._get_crop_bounds()

		## Cropped image - still slighly too many pixels
		img = img[ymin:ymax, xmin:xmax]
		img = cv2.resize(img, (0, 0), fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_NEAREST)

		## Store values for inspected pixel recaulcation
		self.orig_pixels_left = xmin - 1
		self.orig_pixels_top = ymin - 1
		self.is_margin_left = margin_left
		self.is_margin_top = margin_top
		self.margin_pixels_left = img.shape[1] - self.CFG.COMMON.SHAPE[1]
		self.margin_pixels_top  = img.shape[0] - self.CFG.COMMON.SHAPE[0]

		## Crop to target size - horizontally
		if margin_left:
			img = img[:, self.margin_pixels_left:]
		else:
			img = img[:, :self.CFG.COMMON.SHAPE[1] ]

		## Crop to target size - vertically
		if margin_top:
			img = img[self.margin_pixels_top:, :]
		else:
			img = img[:self.CFG.COMMON.SHAPE[0] , :]

		return img


	def _get_inspected_pixel(self, step: int) -> Tuple[Union[np.ndarray, int], Tuple[int, int]]:
		"""
		Get the value and original image indices of the inspected pixel.
		Returns (pixel_value, (x, y)) in accordance with OpenCV axes.
		"""
		x, y = self.mouse_position.x, self.mouse_position.y

		## Remove non-full pixels when zoomed in
		x -= self.margin_pixels_left if self.is_margin_left else 0
		y -= self.margin_pixels_top if self.is_margin_top else 0

		## Remap to original coordinates
		x //= self.zoom
		y //= self.zoom
		x += self.orig_pixels_left
		y += self.orig_pixels_top

		## Add +1 to ensure the correct pixel is selected (operator //=) returns floor of value
		x += 1
		y += 1

		## HACK Clipping safeguard
		x = max(min(x, self.CFG.COMMON.SHAPE[1]-1), 0)
		y = max(min(y, self.CFG.COMMON.SHAPE[0]-1), 0)


		return self.images[step][y, x], (x, y)


	def _get_contour_info(self, step_idx: int, x: int, y: int) -> Union[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int], None]:
		"""
		Get information about the contour the mouse is current over.
		Returns ((cx, cy), (xmin, ymin), (xmax, ymax), area), where cx, cy are contour center of mass indices,
		xmin, xmax, ymin, ymax are two of its bounding box corners.
		Returns `None` if the pixel does not belong to any contour.
		"""
		contours, _ = cv2.findContours(self.images[step_idx], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		contour_index = None

		## Find contour
		for i in range(len(contours)):
			if cv2.pointPolygonTest(contours[i], (x, y), False) >= 0:
				contour_index = i
				break

		if contour_index is None:
			return None

		cnt = contours[contour_index]
		area = cv2.contourArea(cnt)

		## Calculate contour center of mass, with safeguards against very small contours
		moments = cv2.moments(cnt)
		cx = moments['m10'] / moments['m00'] if moments['m00'] != 0 else -1
		cy = moments['m01'] / moments['m00'] if moments['m00'] != 0 else -1

		## Get contour bounding coordinates
		xmin = np.min(cnt[:, :, 0])
		xmax = np.max(cnt[:, :, 0])
		ymin = np.min(cnt[:, :, 1])
		ymax = np.max(cnt[:, :, 1])

		return (int(cx), int(cy)), (xmin, ymin), (xmax, ymax), int(area)


	def _draw_contour_bbox(self, img: np.ndarray, xmin: int, xmax: int, ymin: int, ymax: int) -> np.ndarray:
		"""
		Draw the bounding box of a contour on the zoomed in image.
		"""
		## Store original coordinate values in order to draw text onto the image later
		xmin_orig = int(xmin)
		xmax_orig = int(xmax)
		ymin_orig = int(ymin)
		ymax_orig = int(ymax)

		## Transform variables into local coordinates (zoomed image)
		## Ensure values are as close to contour border as possible without overlap
		if self.zoom > 1:
			xmin = xmin - self.orig_pixels_left - 2
			xmax = xmax - self.orig_pixels_left
			ymin = ymin - self.orig_pixels_top - 2
			ymax = ymax - self.orig_pixels_top

			xmin *= self.zoom
			xmax *= self.zoom
			ymin *= self.zoom
			ymax *= self.zoom

			xmin += self.margin_pixels_left if self.is_margin_left else 0
			xmax += self.margin_pixels_left if self.is_margin_left else 0
			ymin += self.margin_pixels_top if self.is_margin_top else 0
			ymax += self.margin_pixels_top if self.is_margin_top else 0

			xmin += self.zoom - 1
			ymin += self.zoom - 1

		## Draw lines on the image
		clr = (self.CFG.BOUNDING_BOX_COLOR,) * 3
		img = cv2.line(img, (xmin, 0), (xmin, self.CFG.COMMON.SHAPE[0]-1), clr, thickness=1)
		img = cv2.line(img, (xmax, 0), (xmax, self.CFG.COMMON.SHAPE[0]-1), clr, thickness=1)
		img = cv2.line(img, (0, ymin), (self.CFG.COMMON.SHAPE[1]-1, ymin), clr, thickness=1)
		img = cv2.line(img, (0, ymax), (self.CFG.COMMON.SHAPE[1]-1, ymax), clr, thickness=1)

		## Draw x/y min/max int values on lines
		img = self._draw_centered_text(img, str(ymin_orig), (xmin+xmax)/2, ymin - self.CFG.TEXT.MARGIN, above=True,  color=clr, corr=2*self.CFG.TEXT.MARGIN)
		img = self._draw_centered_text(img, str(ymax_orig), (xmin+xmax)/2, ymax + self.CFG.TEXT.MARGIN, above=False, color=clr, corr=2*self.CFG.TEXT.MARGIN)
		img = self._draw_centered_text(img, str(xmin_orig), xmin - self.CFG.TEXT.MARGIN, (ymin+ymax)/2, left=True,   color=clr, corr=2*self.CFG.TEXT.MARGIN)
		img = self._draw_centered_text(img, str(xmax_orig), xmax + self.CFG.TEXT.MARGIN, (ymin+ymax)/2, left=False,  color=clr, corr=2*self.CFG.TEXT.MARGIN)

		return img


	def _draw_centered_text(self, img: np.ndarray, text: str, cx: int, cy: int, corr: int = 0, above: Optional[bool] = None, left: Optional[bool] = None, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
		"""
		Draw centered text on the image. `cx`, `cy` are coordinates of one of the text's bounding box's sides in accordance with OpenCV axes.
		If `above` is `True`, cy denotes coordinate of the text bottom, otherwise it is the text top.
		If `above` is `None`, this means that the text is to be horizontally centered, and cy is already the text's vertical center.
		If `left` is `True`, cx denotes the position of the text right, otherwise the text's left.
		If `left` is `None`, this means that the text is to be vertically centered, and cx is already the text's horizontal center.
		Note that `left` and `above` can neither both be boolean at the same time, nor both be `None` at the same time.
		`corr` is correction factor in case the `above` or `left` arguments have to be overloaded to prevent the text going out of the image.
		"""
		clr = color if color is not None else self.CFG.TEXT.COLOR

		if left is not None and above is not None:
			raise ValueError('Either left, above or both must be None!')

		(text_w, text_h), _ = cv2.getTextSize(text, self.FONT, self.CFG.TEXT.SCALE, thickness=1)

		## Text bottom left coordinates
		org_x, org_y = None, None

		## ================================ TODO Safeguards against text going out of bounds =================

		## `left` is bool - cy means text's vertical center
		if left is not None:
			## Ensure text is always drawn on the image
			if left and cx - text_w <= 0:
				left = False
				cx += corr
			elif not left and cx + text_w >= self.CFG.COMMON.SHAPE[1]:
				left = True
				cx -= corr

			if left:
				org_x = cx - text_w
			else:
				org_x = cx
			org_y = cy + text_h / 2

		## `above` is bool - cx mean text's horizontal center
		elif above is not None:
			## Guard against the text going out of the image
			if above and cy - text_h <= 0:
				above = False
				cy += corr
			elif not above and cy + text_h >= self.CFG.COMMON.SHAPE[0]:
				above = True
				cy -= corr

			if above:
				org_y = cy
			else:
				org_y = cy + text_h
			org_x = cx - text_w / 2

		return cv2.putText(img, text, (int(org_x), int(org_y)), self.FONT, self.CFG.TEXT.SCALE, clr, thickness=1, lineType=cv2.LINE_AA)


	def _draw_pixel_label(self, orig_step: int, img: np.ndarray) -> np.ndarray:
		"""
		Draw the pixel inspection label (if enabled) on the image.
		"""
		if not self.display_pixel_label:
			return img

		draw_img = img.copy()

		px, (x, y) = self._get_inspected_pixel(orig_step)
		px_str = to_comma_separated_string(px)

		## Construct label text
		## Displayed in two lines for better readability
		if len(img.shape) == 3:
			text_top = f'BGR={px_str}'
			text_bottom = f'X{x} Y{y}'
		else:
			if is_binary_image(self.images[orig_step]):
				ret = self._get_contour_info(orig_step, x=x, y=y)
				## Extra safeguard just in case, in order to avoid unpacking errors
				if ret is not None:
					(cx, cy), (xmin, ymin), (xmax, ymax), area = ret
					draw_img = self._draw_contour_bbox(draw_img, xmin, xmax, ymin, ymax)
					text_top = f'CX{cx} CY{cy} A{area}'
					text_bottom = f'V={px_str} X{x} Y{y}'
				else:
					text_top = f'V={px_str}'
					text_bottom = f'X{x} Y{y}'
			else:
				text_top = f'V={px_str}'
				text_bottom = f'X{x} Y{y}'


		## Bottom and top text line sizes
		(w_top, h_top), _ = cv2.getTextSize(text_top, self.FONT, self.CFG.TEXT.SCALE, thickness=1)
		(w_bottom, h_bottom), _ = cv2.getTextSize(text_bottom, self.FONT, self.CFG.TEXT.SCALE, thickness=1)

		## Total label height and width
		label_h = h_top + h_bottom + 2 * self.CFG.TEXT.MARGIN + 2 * self.CFG.PIXEL_LABEL_BORDER + self.CFG.TEXT.SPACING
		label_w = max(w_top, w_bottom) + 2 * self.CFG.TEXT.MARGIN + 2 * self.CFG.PIXEL_LABEL_BORDER

		## Draw text in a contrasting color
		text_clr = (255, 255, 255) if np.mean(px) < self.CFG.TEXT.CONTRAST_THRESH else (0, 0, 0)

		## Top left corner position of entire label (including margins and borders)

		## Check that drawing label to the right does not go out of image bounds, draw to the left if it does
		label_x_max_right = self.mouse_position.x + self.CFG.PIXEL_LABEL_OFFSET + label_w
		label_x = label_x_max_right - label_w if label_x_max_right < self.CFG.COMMON.SHAPE[1] else self.mouse_position.x - (self.CFG.PIXEL_LABEL_OFFSET + label_w)

		## Check that drawing label to the bottom does not go out of image bounds, draw to the top if it does
		label_y_max_bottom = self.mouse_position.y + self.CFG.PIXEL_LABEL_OFFSET + label_h
		label_y = label_y_max_bottom - label_h if label_y_max_bottom < self.CFG.COMMON.SHAPE[0] else self.mouse_position.y - (self.CFG.PIXEL_LABEL_OFFSET + label_h)

		## Add borders (underlying rectangle)
		draw_img = cv2.rectangle(draw_img, (label_x, label_y), (label_x + label_w, label_y + label_h), text_clr, thickness=-1)

		## Add text backgound
		bg_clr = (int(px),)*3 if isinstance(px, (int, np.uint8)) else tuple([int(v) for v in px])
		draw_img = cv2.rectangle(draw_img, (label_x + self.CFG.PIXEL_LABEL_BORDER, label_y + self.CFG.PIXEL_LABEL_BORDER), (label_x + label_w - self.CFG.PIXEL_LABEL_BORDER, label_y + label_h - self.CFG.PIXEL_LABEL_BORDER), bg_clr, thickness=-1)

		## Add top and bottom text
		draw_img = cv2.putText(draw_img, text_top, (label_x + self.CFG.PIXEL_LABEL_BORDER + self.CFG.TEXT.MARGIN, label_y + self.CFG.PIXEL_LABEL_BORDER + self.CFG.TEXT.MARGIN + h_top), self.FONT, self.CFG.TEXT.SCALE, text_clr, thickness=1, lineType=cv2.LINE_AA)
		draw_img = cv2.putText(draw_img, text_bottom, (label_x + self.CFG.PIXEL_LABEL_BORDER + self.CFG.TEXT.MARGIN, label_y + self.CFG.PIXEL_LABEL_BORDER + self.CFG.TEXT.MARGIN + h_top + h_bottom + self.CFG.TEXT.SPACING), self.FONT, self.CFG.TEXT.SCALE, text_clr, thickness=1, lineType=cv2.LINE_AA)

		## Alpha-add images
		return cv2.addWeighted(draw_img, self.CFG.ALPHA, img, 1.-self.CFG.ALPHA, 0)



	def _get_image(self, draw_label: bool, frame_id: int, step_idx: int) -> np.ndarray:
		"""
		Get a zoomed in version of the image.
		"""
		## Return drawn image if no zoom
		if self.zoom == 1:
			self.orig_pixels_left = 0
			self.orig_pixels_top = 0
			img = self._draw_frame_info(frame_id, step_idx) if draw_label else self.images[step_idx]
			return self._draw_pixel_label(step_idx, img) if self.display_pixel_label else img

		## Safegaurd against uninitialized mouse position - zoom into center
		if not self.zoom_center:
			self.zoom_center = PixelCoordinate(x=self.CFG.COMMON.SHAPE[1]//2, y=self.CFG.COMMON.SHAPE[0]//2)

		## Get image to zoom into
		img = self.images[step_idx].copy()

		## Get zoomed in image
		img = self._zoom_image(img)

		## Draw pixel inspection label
		img = self._draw_pixel_label(step_idx, img)

		return self._draw_frame_info(frame_id, step_idx, img) if draw_label else img


	def show(self, draw_label: bool = True, frame_id: int = 0) -> Tuple[bool, ImagesToSave]:
		"""
		Show images and return whether program should exit.
		Shows first image given if in stream mode or iterates through processing steps with keyboard
		interaction if in inspect mode.
		Returns whether program should exit and the list of images to save (screenshots taken) with
		frame and processing step info.

		Args:
		- `draw_label` - whether to draw a label in the bottom-right corner with information about the current frame and processing step
		- `frame_id` - index of the current frame being displayed. Ignored if `draw_label` is `False`
		"""
		## Dataclass containing images to save
		images_to_save = ImagesToSave()

		## Safeguard against uninitialized visualizer
		if not self.images:
			return False, images_to_save

		## Iterate through images via keyboard in inspect mode
		if self.inspect_mode:
			i = 0
			while True:
				## Get image with label (if specified) or raw image if label is turned off
				img = self._get_image(draw_label, frame_id, i)

				## Show the image on the window
				cv2.imshow(self.CFG.WINDOW_NAME, img)

				## Get user input to interact with the program
				key = cv2.waitKey(1)

				## Move the displayed processing forward one step
				if key == ord(self.CFG.KEYS.CONTINUE):
					i = min(i+1, len(self.images) - 1)
					self.zoom = 1

				## Move the displayed processing backwards one step
				elif key == ord(self.CFG.KEYS.BACK):
					i = max(i-1, 0)
					self.zoom = 1

				## Zoom in on current mouse position
				elif key == ord(self.CFG.KEYS.ZOOM_IN):
					self.zoom = min(self.zoom + 1, self.CFG.MAX_ZOOM)

				## Zoom out of current mouse position
				elif key == ord(self.CFG.KEYS.ZOOM_OUT):
					self.zoom = max(self.zoom - 1, 1)

				## Reset zoom to 1x
				elif key == ord(self.CFG.KEYS.RESTORE):
					self.zoom = 1

				## Exit out of inspect mode
				elif key == ord(self.CFG.KEYS.INSPECT):
					self.zoom = 1
					self.inspect_mode = False
					break

				## Exit out of the program
				elif key == ord(self.CFG.KEYS.EXIT):
					return True, images_to_save

				## Take a screenshot of the current step being displayed (save it to an image file)
				## Note that even if the label is being displayed, the raw image (without the label) will be saved
				elif key == ord(self.CFG.KEYS.SCREENSHOT):
					images_to_save.add(self.images[i], i, self.step_names[i])
					# images_to_save.add(img, i, self.step_names[i]) ## Switch this line with the one above to save screenshots with visualizations

		## Show the last image (most likely the final processing step) given to the visualizer
		else:
			cv2.imshow(self.CFG.WINDOW_NAME, self.images[-1])

			## Get user input
			key = cv2.waitKey(1)

			## Enter into inspect mode if specified
			if key == ord(self.CFG.KEYS.INSPECT):
				self.inspect_mode = True

			## Exit out of the program entirely
			elif key == ord(self.CFG.KEYS.EXIT):
				return True, images_to_save

		return False, images_to_save