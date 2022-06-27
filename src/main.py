from config import Config
from imio import ImageIO
from visualizer import ProcessingVisualizer

from improc import ImageProcessor


def main() -> None:
	## Find the config file - automatically, even of it's been moved
	cfg = Config.auto()
	done = False

	with ProcessingVisualizer(cfg.VISUALIZATION) as vis, ImageIO(cfg.IO) as io, ImageProcessor(cfg.PROCESSING) as proc:
		for frame_id, img in io.read_images():
			vis.reset() ## Important - you should reset the images stored each loop iterations, otherwise you'll quickly run out of memory. Simple keep this line in
			vis.store(img, 'Original')

			"""
			You are provided with an example processing path here (detecting the largest white region in an image),
			but feel free to replace it with your own!
			Adding `vis.store(img, 'Name')` after each step will let you inspect your processing path in detail,
			hopefully helping you improve your algorithm.

			Feel free to delete this comment after you've read through it.
			"""
			## <<======================= START OF PROCESSING ==============================>>

			smoothed = proc.smooth(img)
			vis.store(smoothed, 'Smoothed')

			gray = proc.to_grayscale(smoothed)
			vis.store(gray, 'Grayscale')

			eq = proc.equalize(gray)
			vis.store(eq, 'Equalized')

			thresh = proc.thresh(eq)
			vis.store(thresh, 'Thresholded')

			morph = proc.dilate(thresh)
			vis.store(morph, 'Dilated')

			cnt = proc.separate_largest_contour(morph)
			vis.store(cnt, 'Filtered')

			drawn = vis.draw_mask(img, cnt)
			vis.store(drawn, 'Final')

			## <<========================= END OF PROCESSING ==============================>>

			## Save images - images from only one processing step should be saved, since `io` manages only
			## one video output at a time. Most likely you'll want to create a video with the results of your
			## algorithm nicely visualized on the original image
			io.save(drawn)

			done, images_to_save = vis.show(frame_id=frame_id)
			io.save_screenshots(images_to_save) ## Save screenshots - processing steps marked with `s`

			if done: break



if __name__ == '__main__':
	main()
