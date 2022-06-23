"""
Image I/O module. Contains class responsible for reading images from
camera/video stream as well as saving as separate files or to a video file.

MS
"""
import cv2
import numpy as np

import os
import datetime

from config import Config

from typing import Generator, Union, Any


class ImageIO():
    """
    Class for image reading and saving.
    """
    ## Accepted video formats
    VIDEO_FORMATS = ['mp4', 'avi']

    ## Accepted image formats
    IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']

    ## Video codecs for saving files as videos
    VIDEO_CODECS = {
        'mp4' : 'MP4V',
        'avi' : 'DIVX'
    }

    def __init__(self, cfg: Config) -> None:
        ## Keep config internally for quick access
        self.CFG = cfg

        ## Get video capture
        self.cap = cv2.VideoCapture(self.CFG.IO.CAPTURE)
        self.cap.set(cv2.CAP_PROP_FPS, self.CFG.IO.FPS)

        if not os.path.isdir(self.CFG.IO.SAVE_PATH):
            os.mkdir(self.CFG.IO.SAVE_PATH)

        ## Get file saver if saving is specified as video
        if self.CFG.IO.SAVE_FORMAT.lstrip('.') in ImageIO.VIDEO_FORMATS and self.CFG.IO.SAVE:
            fourcc = cv2.VideoWriter_fourcc(ImageIO.VIDEO_CODECS[self.CFG.IO.SAVE_FORMAT.lower()])
            video_path = os.path.join(self.CFG.IO.SAVE_PATH, self._get_filename())

            self.writer = cv2.VideoWriter(video_path, fourcc, self.CFG.IO.FPS, tuple(self.CFG.IO.COMMON.SHAPE[::-1]))

        ## Keep internal counter for file naming if image saving is specified
        else:
            self.cnt = 0


    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        """Release writer on object delete to ensure video gets created correctly."""
        if hasattr(self, 'writer'):
            self.writer.release()


    def read_images(self) -> Generator[Union[np.ndarray, None], None, None]:
        """
        Read in images consecutively from cap.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            yield cv2.resize(frame, tuple(self.CFG.COMMON.SHAPE[::-1]))


    def save(self, img: np.ndarray) -> None:
        """
        Save an image to the video writer or a a separate image if specfifed.
        """
        ## Raise exception if saving is disabled but saving called
        if not self.CFG.IO.SAVE:
            raise ValueError("Save called but is disabled in config")

        ## Get file save format stored in config
        file_save_format = self.CFG.IO.SAVE_FORMAT.lstrip('.')

        ## Save as image if specified
        if file_save_format in ImageIO.IMAGE_FORMATS:
            cv2.imwrite(self._get_filename(), img)

        ## Save as video if specified
        elif file_save_format in ImageIO.VIDEO_FORMATS:
            self.writer.write(cv2.resize(img, tuple(self.CFG.COMMON.SHAPE[::-1])))

        ## Invalid save format specified -> Exception
        else:
            raise ValueError("Invalid save format")



    def _get_filename(self) -> str:
        """Get save filename with extension (excluding path)."""
        base_filename = datetime.datetime.now().strftime('IMG_%d-%m-%Y_%-H-%M-%S')

        ## Add padded counter to images so that they stay in order
        if self.CFG.IO.SAVE_FORMAT in ImageIO.IMAGE_FORMATS:
            base_filename = base_filename + '__' + self.pad(self.cnt)
            self.cnt += 1

        return base_filename + '.' + self.CFG.IO.SAVE_FORMAT.strip('.')


    @staticmethod
    def pad(num: int, width: int = 4) -> str:
        """Pad an int with zeros to ensure constant filename length."""
        snum = str(num)
        zeros = '0'*(width - len(snum))
        return zeros + snum