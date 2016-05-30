import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

import mglearn

cm = ListedColormap(["#FF0000", "#0000FF"])


def plot_knn_classification(n_neighbors=1):
    X, y = mglearn.datasets.make_forge()

    X_test = np.array([[8.2, 3.66214339], [9.9, 3.2], [11.2, .5]])
    dist = euclidean_distances(X, X_test)
    closest = np.argsort(dist, axis=0)

    for x, neighbors in zip(X_test, closest.T):
        for neighbor in neighbors[:n_neighbors]:
            plt.arrow(x[0], x[1], X[neighbor, 0] - x[0],
                      X[neighbor, 1] - x[1], head_width=0, fc='k', ec='k')

    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    plt.scatter(X_test[:, 0], X_test[:, 1], marker="x", s=60,
                c=clf.predict(X_test), cmap=cm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=60, linewidth=0, cmap=cm)
