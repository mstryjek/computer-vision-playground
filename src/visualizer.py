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
        ## TODO Precalculate label rect values here


    def __enter__(self):
        """Enter context and create widnow."""
        cv2.namedWindow(self.CFG.WINNAME)
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
        return cv2.addWeighted(img, self.CFG.ALPHA, ret, 1.-self.CFG.ALPHA, 0)


    def _draw_frame_info(self, frame_id: int, step: int) -> np.ndarray:
        """
        Draw a label with information about the current frame number, processing step id and processing
        step name (if given). Uses the `ALPHA` value from config.
        """



    def show(self, draw_info: bool = True, frame_id: int = 0) -> Tuple[bool, ImagesToSave]:
        """
        Show images and return whether program should exit.
        Shows first image given if in stream mode or iterates through processing steps with keyboard
        interaction if in inspect mode.
        Returns whether program should exit.
        """
        images_to_save = ImagesToSave()

        ## Safeguard against uninitialized visualizer
        if not self.images:
            return False, images_to_save

        ## Iterate through images via keyboard in inspect mode
        if self.inspect_mode:
            i = 0
            while True:
                img = self._draw_frame_info(frame_id, i) if draw_info else self.images[i]
                cv2.imshow(self.CFG.WINDOW_NAME, img)
                key = cv2.waitKey(0)
                if key == ord(self.CFG.KEYS.CONTINUE):
                    i += 1
                elif key == ord(self.CFG.KEYS.BACK):
                    i = max(i-1, 0)
                elif key == ord(self.CFG.KEYS.INSPECT):
                    self.inspect_mode = False
                    break

        ## Show the last image (most likely the final processing step) given to the visualizer
        else:
            cv2.imshow(self.CFG.WINDOW_NAME, self.images[-1])

            ## Check for change to inspect mode
            key = cv2.waitKey(1)
            if key == ord(self.CFG.KEYS.INSPECT):
                self.inspect_mode = True

        if key == ord(self.CFG.KEYS.BREAK):
            return True, images_to_save
        return False, images_to_save