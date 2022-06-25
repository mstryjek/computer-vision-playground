from config import Config
from imio import ImageIO
from visualizer import ProcessingVisualizer

from improc import ImageProcessor

def main() -> None:
	"""
	TODO Desc
	"""
	cfg = Config.auto()
	done = False

	with ProcessingVisualizer(cfg.VISUALIZATION) as vis, ImageIO(cfg.IO) as io, ImageProcessor(cfg.PROCESSING) as proc:
		for frame_id, img in io.read_images():
			vis.reset()
			vis.store(img, 'Original')

			smoothed = proc.smooth(img)
			vis.store(smoothed, 'Smooth')

			eq = proc.equalize(smoothed)
			vis.store(eq, 'Equalized')

			done, images_to_save = vis.show(frame_id=frame_id)
			io.save_screenshots(images_to_save)

			if done: break




if __name__ == '__main__':
	main()
