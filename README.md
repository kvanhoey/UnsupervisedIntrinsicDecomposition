Code for executing the Single-Image Intrinsic Decomposition algorithm presented in the paper:

**Unsupervised Deep Single-Image Intrinsic Decomposition using Illumination-Varying Image Sequences**

by *[Louis Lettry](mailto:lettryl@vision.ee.ethz.ch), [Kenneth Vanhoey](https://www.kvanhoey.eu)* and *Luc Van Gool*

in Computer Graphics Forum, vol. 37, no. 10 (Proceedings of [Pacific Graphics 2018](http://sweb.cityu.edu.hk/pg2018/)).

Paper available on [kvanhoey.eu](http://kenneth.vanhoey.free.fr/index.php?page=research&lang=en#LVvG18b) and [ArXiv](https://arxiv.org/abs/1803.00805).

# Citation
Please cite our paper if you use, compare to, or get inspired by this code and/or work.
```
@article{LVvG18,
    author={Lettry, Louis and Vanhoey, Kenneth and {Van Gool}, Luc},
    title="{Unsupervised Deep Single-Image Intrinsic Decomposition using Illumination-Varying Image Sequences}",
    booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
    year = "2018",
    month = "October",
    journal = "Computer Graphics Forum (Proceedings of Pacific Graphics)",
    volume = "37",
    number = "10"
}
```

# Content
This repository contains two python scripts.
The main entry point is *main.py*.
It
1. loads the pre-trained neural network, and
2. batch-executes it on a list of files.

An example file is given in the folder *input/*
Global parameters at the top of *main.py* allow to edit location of input and result folders.
The folder *model/* contains the pre-trained CNN definition and weights.

# Installation & Usage
1. Clone this repository
```
git clone git@git.ee.ethz.ch:lettryl/UnsupervisedIntrinsicDecomposition.git
```
2. Enter the repository folder
```
cd UnsupervisedIntrinsicDecomposition/
```
3. (Optional) place the files you want to process in the folder *input/*
4. Run the application
```
python main.py
```

# Dependencies
Tested on Linux Ubuntu 18.04.
Should probably work on other systems with minor effort.

Depends on:
* Python (developed on 2.7, compatible with Python3)
* Tensorflow (developed on v1.2.0)
* Python libraries: NumPy, PIL.


