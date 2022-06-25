"""
Image I/O module. Contains class responsible for reading images from
camera/video stream as well as saving the stream to a video file and saving screenshots.
"""
import cv2
import numpy as np

import os
import datetime

from config import Config

from typing import Iterable, Union, Tuple, Any

from utils import ImagesToSave



class ImageIO():
	"""
	Class for image reading and saving.
	"""
	## Accepted image formats
	IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']

	## Accepted video formats
	## You can add your own formats here, but remember to add their codecs below
	VIDEO_FORMATS = ['mp4', 'avi']

	## Video codecs for saving files as videos
	VIDEO_CODECS = {
		'mp4' : 'MP4V',
		'avi' : 'DIVX'
	}

	def __init__(self, cfg: Config) -> None:
		## Keep config internally for quick access
		self.CFG = cfg

		## Keep internal number of frames
		self.cnt = 0

		## Whether the `save()` method has been called. This will initialize the output stream
		self.output_initialized = False

		## Create output directory if it does not exist
		if not os.path.isdir(self.CFG.SAVE_PATH):
			os.mkdir(self.CFG.SAVE_PATH)

		## Get video capture
		self.cap = cv2.VideoCapture(self.CFG.CAPTURE)
		self.cap.set(cv2.CAP_PROP_FPS, self.CFG.FPS)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CFG.COMMON.SHAPE[0])
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CFG.COMMON.SHAPE[1])





	def __enter__(self):
		"""Enter context."""
		return self


	def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
		"""Release writer on object delete to ensure video gets created correctly."""
		if hasattr(self, 'writer'):
			self.writer.release()


	def _init_output_stream(self) -> None:
		"""
		Initialize the output video writer, create the output directory.
		"""
		if self.output_initialized:
			raise RuntimeError('Init output stream called but is already initialized!')
		
		## Get file saver if saving is specified as video
		if self.CFG.VIDEO_SAVE_FORMAT.lstrip('.') in ImageIO.VIDEO_FORMATS:
			fourcc = cv2.VideoWriter_fourcc(ImageIO.VIDEO_CODECS[self.CFG.VIDEO_SAVE_FORMAT.lower()])
			video_path = os.path.join(self.CFG.SAVE_PATH, self._get_video_filename())

			self.writer = cv2.VideoWriter(video_path, fourcc, self.CFG.FPS, tuple(self.CFG.COMMON.SHAPE[::-1]))
		## Keep internal counter for file naming if image saving is specified
		else:
			raise RuntimeError('Invalid stream save format!')
		
		## Mark the output stream as initialized ok
		self.output_initialized = True


	def read_images(self) -> Iterable[Tuple[int, Union[np.ndarray, None]]]:
		"""
		Read in images consecutively from cap.
		Note that this function can still return `None` (an empty image) since you may want to implement
		your own handling of None-type values.
		"""
		while self.cap.isOpened():
			ret, frame = self.cap.read()
			self.cnt += 1
			yield self.cnt, cv2.resize(frame, tuple(self.CFG.COMMON.SHAPE[::-1]))


	def save(self, img: np.ndarray) -> None:
		"""
		Save an image to the video writer or a a separate image if specfifed.
		"""
		## Initialize the output video writer if this is the first call
		if not self.output_initialized:
			self._init_output_stream()

		## Write the frame to the output stream
		self.writer.write(cv2.resize(img, tuple(self.CFG.COMMON.SHAPE[::-1])))



	def save_screenshots(self, images_to_save: ImagesToSave) -> None:
		"""
		Save images marked as screenshots to separate files.
		Function skeleton for now.
		"""
		if not images_to_save:
			return

		for img, step, step_name in images_to_save:
			filename = self._get_image_filename(step, step_name)
			cv2.imwrite(filename, img)


	def _get_image_filename(self, step: int, step_name: Union[str, None] = None) -> str:
		"""
		Get the filename of a screenshot.
		"""
		## Warn if invalid file extenstion specified
		if self.CFG.IMAGE_SAVE_FORMAT not in ImageIO.IMAGE_FORMATS:
			raise RuntimeError('Invalid image format specified!')

		filename = f'FRAME_{self.cnt}-STEP_{step+1}'
		filename = filename + f'-{step_name.upper()}' if step_name is not None else filename
		filename = filename + '.' + self.CFG.IMAGE_SAVE_FORMAT.lstrip('.')
		return os.path.join(self.CFG.SAVE_PATH, filename)


	def _get_video_filename(self) -> str:
		"""Get save filename with extension (excluding path)."""
		base_filename = datetime.datetime.now().strftime('IMG_%d-%m-%Y_%-H-%M-%S')
		return base_filename + '.' + self.CFG.VIDEO_SAVE_FORMAT.strip('.')


