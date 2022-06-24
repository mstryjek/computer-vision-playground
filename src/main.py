from config import Config
from imio import ImageIO
from visualizer import ProcessingVisualizer

from improc import ImageProcessor

def main() -> None:
    """
    Test function for intermediate visualizations
    """
    cfg = Config.auto()
    done = False


    with ProcessingVisualizer(cfg.VISUALIZATION) as vis, ImageIO(cfg.IO) as io, ImageProcessor(cfg.PROCESSING) as proc:
        for frame_id, img in io.read_images():
            vis.reset()
            vis.store(img)

            smoothed = proc.smooth(img)
            vis.store(smoothed)

            done, images_to_save = vis.show(frame=frame_id)
            io.save_screenshots(images_to_save)

            if done: break




if __name__ == '__main__':
    main()
