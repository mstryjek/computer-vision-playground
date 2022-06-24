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

        ## Get video capture
        self.cap = cv2.VideoCapture(self.CFG.CAPTURE)
        self.cap.set(cv2.CAP_PROP_FPS, self.CFG.FPS)

        ## Create output directory if it does not exist
        if not os.path.isdir(self.CFG.SAVE_PATH):
            os.mkdir(self.CFG.SAVE_PATH)

        ## Get file saver if saving is specified as video
        if self.CFG.SAVE_FORMAT.lstrip('.') in ImageIO.VIDEO_FORMATS and self.CFG.SAVE:
            fourcc = cv2.VideoWriter_fourcc(ImageIO.VIDEO_CODECS[self.CFG.SAVE_FORMAT.lower()])
            video_path = os.path.join(self.CFG.SAVE_PATH, self._get_filename())

            self.writer = cv2.VideoWriter(video_path, fourcc, self.CFG.FPS, tuple(self.CFG.COMMON.SHAPE[::-1]))
        ## Keep internal counter for file naming if image saving is specified
        else:
            print('Invalid stream save format!')


    def __enter__(self):
        """Enter context."""
        return self


    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        """Release writer on object delete to ensure video gets created correctly."""
        if hasattr(self, 'writer'):
            self.writer.release()


    def read_images(self) -> Iterable[Tuple[int, Union[np.ndarray, None]]]:
        """
        Read in images consecutively from cap.
        Note that this function can still return `None` (an empty image) since you may want to implement
        your own handling of None-type values.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            self.cnt += 1
            yield cv2.resize(frame, tuple(self.CFG.COMMON.SHAPE[::-1]))


    def save(self, img: np.ndarray) -> None:
        """
        Save an image to the video writer or a a separate image if specfifed.
        """
        ## Raise exception if saving is disabled but saving called
        if not self.CFG.SAVE:
            raise TypeError("Save called but is disabled in config")

        ## Get file save format stored in config
        file_save_format = self.CFG.SAVE_FORMAT.lstrip('.')

        ## Save as video if specified
        if file_save_format in ImageIO.VIDEO_FORMATS:
            self.writer.write(cv2.resize(img, tuple(self.CFG.COMMON.SHAPE[::-1])))

        ## Invalid save format specified ==> Exception
        else:
            raise TypeError("Invalid save format")


    def save_screenshots(self, images_to_save: ImagesToSave) -> None:
        """
        Save images marked as screenshots to separate files.
        Function skeleton for now.        
        """



    def _get_filename(self) -> str:
        """Get save filename with extension (excluding path)."""
        base_filename = datetime.datetime.now().strftime('IMG_%d-%m-%Y_%-H-%M-%S')
        return base_filename + '.' + self.CFG.SAVE_FORMAT.strip('.')


    @staticmethod
    def pad(num: int, width: int = 4) -> str:
        """Pad an int with zeros to ensure constant filename length."""
        snum = str(num)
        zeros = '0'*(width - len(snum))
        return zeros + snum