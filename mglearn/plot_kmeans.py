from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from .plot_2d_separator import plot_2d_classification
from .plot_helpers import cm3


def plot_kmeans_algorithm():

    X, y = make_blobs(random_state=1)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})
    center_args = {'marker': '^', 'c': [0, 1, 2], 'cmap': cm3, 's': 100, 'linewidth': 2}

    axes[0, 0].set_title("Input data")
    axes[0, 0].scatter(X[:, 0], X[:, 1], c='w', s=60)

    axes[0, 1].set_title("Initialization")
    init = X[:3, :]
    axes[0, 1].scatter(X[:, 0], X[:, 1], c='w', s=60)
    axes[0, 1].scatter(init[:, 0], init[:, 1], **center_args)

    axes[0, 2].set_title("Assign Points (1)")
    km = KMeans(n_clusters=3, init=init, max_iter=1, n_init=1).fit(X)
    centers = km.cluster_centers_
    axes[0, 2].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap=cm3, alpha=.6, s=60)
    axes[0, 2].scatter(init[:, 0], init[:, 1], **center_args)

    axes[1, 0].set_title("Recompute Centers (1)")
    axes[1, 0].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap=cm3, alpha=.6, s=60)
    axes[1, 0].scatter(centers[:, 0], centers[:, 1], **center_args)

    axes[1, 1].set_title("Reassign Points (2)")
    km = KMeans(n_clusters=3, init=init, max_iter=2, n_init=1).fit(X)
    axes[1, 1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap=cm3, alpha=.6, s=60)
    axes[1, 1].scatter(centers[:, 0], centers[:, 1], **center_args)

    axes[1, 2].set_title("Recompute Centers (2)")
    centers = km.cluster_centers_
    axes[1, 2].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap=cm3, alpha=.6, s=60)
    axes[1, 2].scatter(centers[:, 0], centers[:, 1], **center_args)


def plot_kmeans_boundaries():
    X, y = make_blobs(random_state=1)
    init = X[:3, :]
    km = KMeans(n_clusters=3, init=init, max_iter=2, n_init=1).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap=cm3, alpha=.6, s=60)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                marker='^', c=[0, 1, 2], cmap=cm3, s=100, linewidth=2)
    plot_2d_classification(km, X, cm=cm3, alpha=.4)
