# petroscope

Petroscope is a python package to analyze and work with microscopic geological images.

## Installation (outdated)

Download wheels distro from the [GitHub releases](https://github.com/khvostikov/petroscope/releases) page then install it:

```bash
python -m pip install petroscope-0.0.5-py3-none-any.whl
```

Or you can install the package from PyPI:

```bash
pip install petroscope
```

The minimal required Python version is 3.10.

## Acknowledgements

The development of methods for processing and analyzing of geological images implemented in this package was supported by the Russian Science Foundation (grant No. 24-21-00061).

## Calibration module

The Calibration module automatically corrects illumination and distortion in microscopic geological images using reference calibration images. These reference images can be:

- OLED/AMOLED screen images (for uniform illumination correction)
- Mirror images (for reflection-based illumination correction)

### Example Usage (predefined script)

To run the calibration process from the command line, navigate to the example data directory and execute:

```bash
cd examples/data/
python -m petroscope.calibrate.run -i ./calibration_test -o ./calibration_test_out -m ./mirror1200.jpg
```

### Using in Python Code

If you prefer to use the module programmatically, check out the provided [example script](./petroscope/examples/calibration.py).

## Segmentation module

This module is dedicated to image segmentation. It contains a number of helpful utils for segmentation related tasks, abstract class [GeoSegmModel](./petroscope/segmentation/model.py), classes to perform segmentation evaluation and metrics calculation.

This module is designed primarily to be used with [LumenStone](https://imaging.cs.msu.ru/en/research/geology/lumenstone) dataset, collected and annotated by our team. Annotation labels for this dataset are provided in [LumenStoneClasses](./petroscope/segmentation/classes.py) class.

### Segmentation metrics

IoU per class, mean IoU, total accuracy metrics are used for segmentation evaluation. For the correct calculation of these metrics, it is necessary to use information about the areas of all objects in the dataset, not just in individual images. These calculations are implemented in [metrics.py](./petroscope/segmentation/metrics.py).

### Patch-sampling balancer

One of the main challenges in developing segmentation methods for geological images of polished sections is the severe class imbalance in datasets for minerals, which naturally occurs due to the varying frequencies of mineral occurrence in nature. Some minerals may appear as small clusters of only a few dozen pixels, while others may occupy a large portion of the images.

Training neural network segmentation models directly on such data leads to extremely poor results. It has also been empirically shown that using various loss functions focused on class imbalance and class weighting does not yield the desired results in this case.

To address class imbalance, a simple yet quite effective patch-based sampling method was proposed, leaning on constructing probability maps of mineral occurrence in the dataset. The implementation is presented in [SelfBalancingDataset](./petroscope/segmentation/balancer/balancer.py) class.

#### References

The implemented patch-based balancer is inspired by our previous works:

- Alexey Kochkarev, Alexander Khvostikov, Dmitry Korshunov, Andrey Krylov, and Mikhail Boguslavskiy. Data balancing method for training segmentation neural networks. CEUR Workshop Proceedings, 2744:1–10, 2020. [DOI](http://dx.doi.org/10.51130/graphicon-2020-2-4-19);

- Zh Sun, A. Khvostikov, A. S. Krylov, A. Sethi, I. Mikhailov, and P. Malkov. Joint super-resolution and tissue patch classification for whole slide histological images. Programming and Computer Software, 50(3):257–263, 2024 [DOI](http://dx.doi.org/10.1134/s0361768824700063);

#### Example of usage

- [segm_balancer.py](./petroscope/examples/segm_balancer.py) - an example of sampling patches from the dataset with simple augmentations, visualizing the probability maps of classes and the obtained accumulators.

### ResUnet segmentation model

[ResUnet](./petroscope/segmentation/models/resunet/model.py) is the base mineral segmentation model which is built upon the UNet architecture with residual conv blocks. It was trained on LumenStone S1v1 dataset for 7 segmenting classes.

> *To use this model you have to install pytorch.*

#### Achieved metrics for LumenStone S1v1 (to be updated)

| class    | IoU    | IoU, void borders |
| -------- | ------ | ----------------- |
| BG       | 0.8326 | 0.8505            |
| Brt      | 0.8868 | 0.8955            |
| Ccp      | 0.9191 | 0.9363            |
| Gl       | 0.7464 | 0.7630            |
| Py/Mrc   | 0.9628 | 0.9732            |
| Sph      | 0.7534 | 0.7653            |
| Tnt/Ttr  | 0.7601 | 0.7706            |
| **mean** | 0.8373 | 0.8506            |

<!-- #### References
The architecture of the model is described in:
- A. V. Khvostikov, D. M. Korshunov, A. S. Krylov, and M. A. Boguslavskiy. Automatic identification of minerals in images of polished sections. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 44:113–118, 2021. [DOI](http://dx.doi.org/10.5194/isprs-archives-XLIV-2-W1-2021-113-2021); -->

#### Examples of usage

- [segm_resunet_inference.py](./petroscope/examples/segm_resunet_inference.py) - an example of using ResUnet model for inference (making prediction for one image);

- [segm_cnn_test.py](./petroscope/examples/segm_cnn_test.py) - a more detailed example demonstrating the testing of ResUnet model on LumenStone S1v1 dataset with calculating of all metrics;

## Panorama module

The panorama module is based on the automatic panoramic image stitching method that produces highly detailed panoramic images from a sequence of partially overlapping images of polished sections.

At the moment the module is available in the experimental branch [panoramas](https://github.com/xubiker/petroscope/tree/panoramas).

### References

The proposed method of panorama stitching is described in:
- Nikolaev, G., Korshunov, D., and Khvostikov, A.: Automatic stitching of panoramas for geological images of polished sections, ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., X-2/W1-2024, 39–46, 2024. [DOI](https://doi.org/10.5194/isprs-annals-X-2-W1-2024-39-2024)