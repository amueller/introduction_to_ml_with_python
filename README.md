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

For the curious ones, the cover depicts a [hellbender](https://en.wikipedia.org/wiki/Hellbender)

![cover](cover.jpg)
