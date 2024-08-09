# petroscope
Petroscope is a python package to analyze and work with microscopic geological images.

## Installation
Download wheels distro from the [GitHub releases](https://github.com/khvostikov/petroscope/releases) page then install it:

```bash
python -m pip install petroscope-0.0.1-py3-none-any.whl
```

The minimal required Python version is 3.10.

# Segmentation module

This module is dedicated to image segmentation. It contains a number of helpful utils for segmentation related tasks, abstract class GeoSegmModel, classes to perform segmentation evaluation and metrics calculation.

It also has a special patch-based balancer. An example of balancer usage is shown in [balancer_usage.py](./segmentation/examples/balancer_usage.py).

Several models for segmentation of geological images of polished sections developed for LumenStone dataset are available.

# Panorama module

This module is used for stitching geological images of polished sections into panoramas.



