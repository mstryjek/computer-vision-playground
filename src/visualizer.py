"""
Visualizer class for visualizing intermediate steps of image processing.
"""

from typing import Any
import cv2
import numpy as np

from config import Config
from utils import ImagesToSave

from typing import Tuple, Union


class ProcessingVisualizer():
	"""
	Class for visualizing images given to it.
	"""
	def __init__(self, cfg: Config) -> None:
		self.CFG = cfg
		self.reset()
		self.inspect_mode = False


	def __enter__(self):
		"""Enter context and create widnow."""
		cv2.namedWindow(self.CFG.WINDOW_NAME)
		return self


	def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
		"""Safely exit."""
		cv2.destroyAllWindows()


	def reset(self) -> None:
		"""Reset image list."""
		self.images = []
		self.step_names = []


	def store(self, img: np.ndarray, step_name: Union[str, None] = None) -> None:
		"""Store an image to be shown later."""
		self.images.append(img.copy())
		self.step_names.append(step_name)


	def draw_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
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

		## Copy of the image to add with alpha weight later
		ret = img.copy()

		## Draw full opacity color on the copy
		ret[mask != 0] = self.CFG.COLOR

		## Return the mask drawn on the image with the given alpha
		return cv2.addWeighted(ret, self.CFG.ALPHA, img, 1.-self.CFG.ALPHA, 0)


	def _draw_frame_info(self, frame_id: int, step: int) -> np.ndarray:
		"""
		Draw a label with information about the current frame number, processing step id and processing
		step name (if given). Uses the `ALPHA` value from config.
		"""
		## Copy image to allow for alpha-weighted composition
		draw_img = self.images[step].copy()

		## Bottom label text with current frame index, processing step number and processing step name if given
		text = f'FRAME: {frame_id} | STEP {step+1}'
		text = text + f' ({self.step_names[step].upper()})' if self.step_names[step] is not None else text

		## Text size and total textbox size
		(tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.CFG.TEXT.SCALE, 1)
		textbox_size = (th+2*self.CFG.TEXT.MARGIN, tw+2*self.CFG.TEXT.MARGIN)

		## Rectangle top-left corner, text bottom-left corner and image corner (rectangle bottom-right corner)
		rect_org = (draw_img.shape[1] - textbox_size[1] - 1, draw_img.shape[0] - textbox_size[0] - 1)
		txt_org = (draw_img.shape[1] - textbox_size[1] - 1 + self.CFG.MARGIN, draw_img.shape[0] - 1 - self.CFG.MARGIN)
		corner = (draw_img.shape[1] - 1, draw_img.shape[0] - 1)

		## Draw rectangle and text on the image
		draw_img = cv2.rectangle(draw_img, rect_org, corner, self.CFG.TEXT.BG_COLOR, -1)
		draw_img = cv2.putText(draw_img, text, txt_org, cv2.FONT_HERSHEY_SIMPLEX, self.CFG.TEXT.SCALE, self.CFG.TEXT.COLOR, 1, cv2.LINE_AA)

		## Return an alpha-weighted composition of the original image and the image with the label
		return cv2.addWeighted(draw_img, self.CFG.ALPHA, self.images[step], 1.-self.CFG.ALPHA, 0)


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
				img = self._draw_frame_info(frame_id, i) if draw_label else self.images[i]

				## Show the image on the window
				cv2.imshow(self.CFG.WINDOW_NAME, img)

				## Get user input to interact with the program
				key = cv2.waitKey(0)

				## Move the displayed processing forward one step
				if key == ord(self.CFG.KEYS.CONTINUE):
					i = min(i+1, len(self.images) - 1)

				## Move the displayed processing backwards one step
				elif key == ord(self.CFG.KEYS.BACK):
					i = max(i-1, 0)

				## Exit out of inspect mode
				elif key == ord(self.CFG.KEYS.INSPECT):
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