from config import Config
from imio import ImageIO
from visualizer import ProcessingVisualizer

from improc import ImageProcessor


def main() -> None:
	## Find the config file - automatically, even of it's been moved
	cfg = Config.auto()
	done = False

	with ProcessingVisualizer(cfg.VISUALIZATION, start_inspect=True) as vis, ImageIO(cfg.IO) as io, ImageProcessor(cfg.PROCESSING) as proc:
		for frame_id, img in io.read_images():

			vis.reset() ## Important - you should reset the images stored each loop iterations, otherwise you'll quickly run out of memory. Simply keep this line in
			vis.store(img, 'Original')

			"""
			You are provided with an example processing path here (detecting the largest white region in an image),
			but feel free to replace it with your own!
			Adding `vis.store(img, 'Name')` after each step will let you inspect your processing path in detail,
			hopefully helping you improve your algorithm.

			Feel free to delete this comment after you've read through it.
			"""
			## <<======================= START OF PROCESSING ==============================>>
			gray = proc.to_grayscale(img)
			vis.store(gray, 'Grayscale')

			blurred = proc.smooth(gray)
			vis.store(blurred, 'Blurrred')

			threshed = proc.otsu(blurred)
			vis.store(threshed, 'Otsu')

			contours = proc.get_largest_contours(threshed, 4)
			boxes = proc.contour_bounding_rects(contours)
			drawn = vis.draw_bounding_contours(threshed, boxes)
			vis.store(drawn, 'Contours')

			warped = proc.warp_contours(threshed, boxes)
			## TODO: Sort by closeness to warped image center, but rejecting anything that touches window 
			## Choose 2 closes contours, keep only those & then dilate, to join them into one
			## Eights have 2 holes & >1.5 aspect ratio, blocks are same but closer to square
			## +2's have no holes
			## sixes & nines are similar, except the hole is on the opposite side of the arithmetic center 
			## from the baseline, which is defined as having more than half(?) of the width of the contour bbox
			## ^^^ Try taking a couple top/bottom rows in case the symbol is slanted, should work
			## ^^^ Width condition is not neccessary, just take the one that has more width
			warped = [proc.dilate(w) for w in warped]

			matchlocs = [proc.match_templates_to_image(w) for w in warped]
			for w, (m, loc, isud) in zip(warped, matchlocs):
				vis.store(vis.draw_point(w, loc), f'{m}-{1 if isud else 0}')





			## <<========================= END OF PROCESSING ==============================>>

			## Save images - images from only one processing step should be saved, since `io` manages only
			## one video output at a time. Most likely you'll want to create a video with the results of your
			## algorithm nicely visualized on the original image
			# io.save(drawn)

			done, images_to_save = vis.show(frame_id=frame_id)
			io.save_screenshots(images_to_save) ## Save screenshots - processing steps marked with `s`

			if done: break



if __name__ == '__main__':
	main()
