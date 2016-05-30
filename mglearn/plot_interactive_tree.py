import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.externals.six import StringIO  # doctest: +SKIP
from sklearn.tree import export_graphviz
from scipy.misc import imread
from scipy import ndimage
from sklearn.datasets import make_moons

import re


def tree_image(tree, fout=None):
    try:
        import graphviz
    except ImportError:
        # make a hacky white plot
        x = np.ones((10, 10))
        x[0, 0] = 0
        return x
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data, max_depth=3, impurity=False)
    data = dot_data.getvalue()
    #data = re.sub(r"gini = 0\.[0-9]+\\n", "", dot_data.getvalue())
    data = re.sub(r"samples = [0-9]+\\n", "", data)
    data = re.sub(r"\\nsamples = [0-9]+", "", data)
    data = re.sub(r"value", "counts", data)

    graph = graphviz.Source(data, format="png")
    if fout is None:
        fout = "tmp"
    graph.render(fout)
    return imread(fout + ".png")


def plot_tree_progressive():
    fig, axes = plt.subplots(4, 2, figsize=(15, 25), subplot_kw={'xticks': (), 'yticks': ()})
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

    for i, max_depth in enumerate([1, 2, 9]):
        tree = plot_tree(X, y, max_depth=max_depth, ax=axes[i + 1, 0])
        axes[i + 1, 1].imshow(tree_image(tree))
        axes[i + 1, 1].set_axis_off()
    axes[0, 1].set_visible(False)
    for ax in axes[:, 0]:
        ax.scatter(X[:, 0], X[:, 1], c=np.array(['r', 'b'])[y], s=60)
    X, y = make_moons(noise=0.3, random_state=0)


def plot_tree_partition(X, y, tree, ax=None):
    if ax is None:
        ax = plt.gca()
    eps = X.std() / 2.

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]

    Z = tree.predict(X_grid)
    Z = Z.reshape(X1.shape)
    faces = tree.apply(X_grid)
    faces = faces.reshape(X1.shape)
    border = ndimage.laplace(faces) != 0
    ax.contourf(X1, X2, Z, alpha=.4, colors=['red', 'blue'], levels=[0, .5, 1])
    ax.scatter(X1[border], X2[border], marker='.', s=1)

    ax.scatter(X[:, 0], X[:, 1], c=np.array(['r', 'b'])[y], s=60)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    return ax


def plot_tree(X, y, max_depth=1, ax=None):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
    ax = plot_tree_partition(X, y, tree, ax=ax)
    ax.set_title("depth = %d" % max_depth)
    return tree
