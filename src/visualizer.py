"""
Visualizer class for visualizing intermediate steps of image processing.

MS
"""

from typing import Any
import cv2
import numpy as np

from config import Config


class ProcessingVisualizer():
    """
    Class for visualizing images given to it.
    """
    def __init__(self, cfg: Config) -> None:
        self.CFG = cfg
        self.reset()
        self.inspect_mode = False
        (text_w, text_h), baseline = cv2.getTextSize(self.CFG.VIS.TEXT, cv2.FONT_HERSHEY_SIMPLEX, self.CFG.VIS.TEXT_SCALE, 2)
        textbox_w = text_w + self.CFG.VIS.RECT_W * 2 * self.CFG.VIS.NUM_RECTS + 2 * self.CFG.VIS.MARGIN
        textbox_h = 2 * self.CFG.VIS.MARGIN + max(text_h, self.CFG.VIS.RECT_H)
        self.textbox_size = (textbox_w, textbox_h)
        self.text_size = (text_w, text_h)


    def __enter__(self):
        """Enter context and create widnow."""
        cv2.namedWindow(self.CFG.VIS.WINNAME)
        return self


    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        """Safely exit."""
        cv2.destroyAllWindows()


    def draw_contour_outlines(self, mask: np.ndarray) -> np.ndarray:
        """
        Draw contours as outlines only.
        """
        res = np.zeros(mask.shape, dtype=np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return cv2.drawContours(res, cnts, -1, (255,255,255), -1)


    def draw_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Draw a color mask on the image.
        """
        masked = img.copy()
        masked[mask != 0] = self.CFG.VIS.RECT_COLOR
        cv2.addWeighted(masked, self.CFG.VIS.ALPHA, img, 1.-self.CFG.VIS.ALPHA, 0, masked)
        return masked


    def draw_textbox(self, img: np.ndarray, num_fingers: int) -> np.ndarray:
        """
        Draw textbox info on an image (top left corner).
        """
        draw_img = img.copy()
        draw_img = cv2.rectangle(draw_img, (0, 0), self.textbox_size, self.CFG.VIS.TEXT_BG, -1)
        draw_img = cv2.putText(draw_img, self.CFG.VIS.TEXT, (self.CFG.VIS.MARGIN, int(self.textbox_size[1]/2 + self.text_size[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, self.CFG.VIS.TEXT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)

        for i in range(self.CFG.VIS.NUM_RECTS):
            color = self.CFG.VIS.RECT_COLOR if i < num_fingers else self.CFG.VIS.RECT_GRAY

            rect_org_x = self.CFG.VIS.MARGIN + self.text_size[0] + self.CFG.VIS.RECT_W*(2*i + 1)
            rect_org_y = int(self.textbox_size[1]/2 - self.CFG.VIS.RECT_H/2)
            draw_img = cv2.rectangle(draw_img, (rect_org_x, rect_org_y), (rect_org_x+self.CFG.VIS.RECT_W, rect_org_y+self.CFG.VIS.RECT_H), color, -1)

        return cv2.addWeighted(draw_img, self.CFG.VIS.ALPHA, img, 1.-self.CFG.VIS.ALPHA, 0)


    def draw(self, img: np.ndarray, mask: np.ndarray, num_fingers: int) -> np.ndarray:
        """
        Draw detection results on an image.
        """
        outlines = self.draw_contour_outlines(mask)
        drawn = self.draw_mask(img, outlines)
        drawn = self.draw_textbox(drawn, num_fingers)
        return drawn


    def reset(self) -> None:
        """Reset image list."""
        self.images = []


    def store(self, img: np.ndarray) -> None:
        """Store an image to be shown later."""
        self.images.append(img)
    

    def show(self) -> bool:
        """
        Show images and return whether program should exit.
        Shows first image given if in stream mode or iterates through processing steps with keyboard
        interaction if in inspect mode.
        Returns whether program should exit.
        """
        ## Safeguard against uninitialized visualizer
        if not self.images:
            return False

        ## Iterate through images via keyboard in inspect mode
        if self.inspect_mode:
            i = 0
            while i < len(self.images):
                cv2.imshow(self.CFG.VIS.WINNAME, self.images[i])
                key = cv2.waitKey(0)
                if key == ord(self.CFG.VIS.CONTINUE_KEY):
                    i += 1
                elif key == ord(self.CFG.VIS.BACK_KEY):
                    i = max(i-1, 0)
                elif key == ord(self.CFG.VIS.INSPECT_KEY):
                    self.inspect_mode = False
                    break
        ## Show raw stream
        else:
            cv2.imshow(self.CFG.VIS.WINNAME, self.images[-1])

            ## Check for change to inspect mode
            key = cv2.waitKey(1)
            if key == ord(self.CFG.VIS.INSPECT_KEY):
                self.inspect_mode = True

        if key == ord(self.CFG.VIS.BREAK_KEY):
            return True
        return False