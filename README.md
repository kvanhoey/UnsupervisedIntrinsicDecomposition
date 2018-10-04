* Content
Code for executing the Single-Image Intrinsic Decomposition algorithm presented in the paper:
'Unsupervised Deep Single-Image Intrinsic Decomposition using Illumination-Varying Image Sequences'
by Louis Lettry (lettryl@vision.ee.ethz.ch), Kenneth Vanhoey (kenneth@research.kvanhoey.eu) and Luc Van Gool
in Computer Graphics Forum, vol. 37, no. 10 (Proceedings of Pacific Graphics 2018).
If you use, compare to, or get inspired by this code and/or work, please cite the paper above.

This repository contains two python scripts, the main entry point being main.py.
Essentially, it loads the pre-trained neural network, and batch-executes it on a list of files.
An example file is given in the folder input/
Global parameters at the top of main.py allow to edit location of input and result folders.
The folder model/ contains the pre-trained CNN definition and weights.


* Dependencies
- Tensorflow (MINIMAL VERSION) ?
- Python libraries: NumPy, ImageIO (depends on PILlow).

Tested on Linux Ubuntu 18.04.
Should probably work on other systems with minor effort.

