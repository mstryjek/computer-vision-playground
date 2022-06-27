# Computer Vision Playground
An OpenCV-based development & testing space for classical image processing algorithms.

<p align="center">
	<img src="resources/visualized.gif"/>
</p>

<p align="center">
<b>Sample processing path</b>
</p>
<div align="center">

![](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)
![](https://img.shields.io/badge/python-3.6+-blue?style=for-the-badge&logo=python&logoColor=blue)
![](https://img.shields.io/badge/opencv-4.x-yellow?style=for-the-badge&logo=opencv&logoColor=yellow)
![](https://img.shields.io/badge/numpy-1.22+-red?style=for-the-badge&logo=numpy&logoColor=red)

</div>

<!-- TOC here -->

## About
This is a relatively simple set of Python tools helping with development of classical image processing algorithms meant for continuous streams of images, such as videos or live camera streams. You can easily view each step of your image processing algorithm, as well as save each step individually or as a video file.

## Setup
You'll need to install the [requirements](requirements.txt):
```shell
python -m pip install -r requirements.txt
```

## Getting started
### Demo
To play around with the functionalities, simply run
```shell
python src/main.py
```
You can stop at any frame by pressing `c`. You can then jump forwards and back between each step of processing with `a` (one step back) and `d` (one step forward). At any step, you can press `s` in order to save that step separately to an image file. Press `c` again to return to resume, and press `q` at any time to exit. Note that you can change the key mappings to whatever you want in [the config file](config/config.yaml). You can also play around with the config values to get a different result.

### Implementing your own algorithms
You can write your own algorithm by writing the processing steps in [`main.py`](src/main.py). Adding the line 
```py
vis.store(...)
``` 
after each step will let you view your steps one by one. You can also add 
```py
io.save(...)
```
in order to save a specific processing step of every frame into a video file. Note that this should only be called on one processing step, otherwise the video file will be hard to decode.
There are also some simple methods for visualization included in [`visualizer.py`](src/visualizer.py).

Although you can write your processing functions anywhere, it may be easiest to make them methods of 
[the processing class](src/improc.py). This will let you have easy access to all config values.

You can also add your own values to config, in order to support your processing methods. For instance, after adding the value:
```yaml
PROCESSING:
  ...
  MY_PROCESSING:
    SOME_VALUE: 1.5
```
you can access it from the processing class by simply using:
```py
self.CFG.MY_PROCESSING.SOME_VALUE
```

By default, images in inspect mode have a label drawn on them, including information about the current frame and the current processing step:
<p align="center">
 <img src="resources/label.png"/>
</p>

You can turn that label off, by replacing the line
```py
done, images_to_save = vis.show(frame_id=frame_id)
```
with
```py
done, images_to_save = vis.show(draw_label=False, frame_id=frame_id)
```
For algorithm development or testing, the only two files you should be modifying are [`main.py`](src/main.py) and [`improc.py`](src/improc.py). However, if you feel the need to modify any other files, feel free to do so!


## License
Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information. If you end up using this repo in a public project, I'd appreciated if you left a link to this repo in your code or in your README.md.


## Bugs & contributing
If you find any bugs, or implement your own features that you'd like merged in the repo, make sure to
submit a PR!




## TODO
- [ ] Add RGB pixel inspector to inspect mode (with pixel indices) (turn on/off by clicking right mouse button)
- [ ] Add contour inspection to cursor inspection (area, min/max bounds, center position)
- [ ] Add zoom in/out functions for better pixel inspection (also a reset key)
- [ ] Add a slider for value tuning (e.g. threshold)