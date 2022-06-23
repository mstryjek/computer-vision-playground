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


    with ProcessingVisualizer(cfg) as vis, ImageIO(cfg) as io, ImageProcessor(cfg) as proc:
        for img in io.read_images():
            vis.reset()
            vis.store(img)

            smoothed = proc.smooth(img)
            vis.store(smoothed)

            eq = proc.equalize(smoothed)
            vis.store(eq)

            mask = proc.extract_color_mask(eq)
            vis.store(mask)

            cnt = proc.extract_largest_contour(mask)
            vis.store(cnt)

            distance, radius, center = proc.inscribe_circle(cnt)
            vis.store(distance)

            no_wrist = proc.remove_wrist(distance, radius, center)
            vis.store(no_wrist)

            blob_filtered = proc.remove_small_blobs(no_wrist)
            vis.store(blob_filtered)

            valid_fingers = proc.remove_bent_fingers(blob_filtered)
            vis.store(valid_fingers)

            num_fingers = proc.count_fingers(valid_fingers)

            drawn = vis.draw(img, valid_fingers, num_fingers)
            vis.store(drawn)

            done = vis.show()
            if done: break




if __name__ == '__main__':
    main()
