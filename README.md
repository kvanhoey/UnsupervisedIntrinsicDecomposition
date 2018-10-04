Code for executing the Single-Image Intrinsic Decomposition algorithm presented in the paper:

**Unsupervised Deep Single-Image Intrinsic Decomposition using Illumination-Varying Image Sequences**

by *[Louis Lettry](mailto:lettryl@vision.ee.ethz.ch), [Kenneth Vanhoey](https://www.kvanhoey.eu)* and *Luc Van Gool*

in Computer Graphics Forum, vol. 37, no. 10 (Proceedings of Pacific Graphics 2018).

Available here: http://kenneth.vanhoey.free.fr/index.php?page=research&lang=en#LVvG18b

If you use, compare to, or get inspired by this code and/or work, please cite the paper above.

# Citation
@article{LVvG18,
author={Lettry, Louis and Vanhoey, Kenneth and {Van Gool}, Luc},
title="{Unsupervised Deep Single-Image Intrinsic Decomposition using Illumination-Varying Image Sequences}",
booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
year = "2018",
month = "October",
journal = "Computer Graphics Forum (Proceedings of Pacific Graphics)",
volume = "37",
number = "10"}

# Content
This repository contains two python scripts, the main entry point being main.py.
Essentially, it loads the pre-trained neural network, and batch-executes it on a list of files.
An example file is given in the folder input/
Global parameters at the top of main.py allow to edit location of input and result folders.
The folder model/ contains the pre-trained CNN definition and weights.

# Installation & Dependencies
Tested on Linux Ubuntu 18.04.
Should probably work on other systems with minor effort.
* Tensorflow (developed on v 1.2.0)
* Python libraries: NumPy, PIL.


