# Introduction to Machine Learning with Python

This repository holds the code for the forthcoming book "Introduction to Machine
Learning with Python" by [Andreas Mueller](http://amueller.io) and [Sarah Guido](https://twitter.com/sarah_guido).
You can find details about the book on the [O'Reilly website](http://shop.oreilly.com/product/0636920030515.do>).

The books requires the current development version of scikit-learn, that is
0.18-dev.  Most of the book can also be used with previous versions of
scikit-learn, though you need to adjust the import for everything from the
``model_selection`` module, mostly ``cross_val_score``, ``train_test_split``
and ``GridSearchCV``.


This repository provides the notebooks from which the book is created, together
with the ``mglearn`` library of helper functions to create figures and
datasets.

For the curious ones, the cover depicts a [hellbender](https://en.wikipedia.org/wiki/Hellbender).

All datasets are included in the repository, with the exception of the aclImdb dataset, which you can download from
the page of [Andrew Maas](http://ai.stanford.edu/~amaas/data/sentiment/). See the book for details.


## Errata
Please note that the first print of the book is missing the following line when listing the assumed imports:

```python
from IPython.display import display
```
Please add this line if you see an error involving ``display``.

## Setup

To run the code, you need the packages ``numpy``, ``scipy``, ``scikit-learn``, ``matplotlib``, ``pandas`` and ``pillow``.
Some of the visualizations of decision trees and neural networks structures also require ``graphviz``.

The easiest way to set up an environment is by installing [Anaconda](https://www.continuum.io/downloads).

### Installing packages with conda:
If you already have a Python environment set up, and you are using the ``conda`` package manager, you can get all packages by running

    conda install numpy scipy scikit-learn matplotlib pandas pillow graphviz

and then *also*

    pip install graphviz

(Explanation: the conda package graphiz is the C library, not the python library)

### Installing packages with pip
If you already have a Python environment and are using pip to install packages, you need to run

    pip install numpy scipy scikit-learn matplotlib pandas pillow graphviz

You also need to install the graphiz C-library, which is easiest using a package manager.
If you are using OS X and homebrew, you can ``brew install graphviz``. If you are on Ubuntu or debian, you can ``apt-get install graphviz``.
Installing graphviz on Windows can be tricky and using conda / anaconda is recommended.

## Errata

If you have errata for the (e-)book, please submit them via the [O'Reilly Website](http://www.oreilly.com/catalog/errata.csp?isbn=0636920030515).
You can submit fixed to the code as pull-requests here, but I'd appreciate it if you would also submit them there, as this repository doesn't hold the
"master notebooks".

![cover](cover.jpg)
