import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import euclidean_distances

from mglearn.datasets import make_wave


def plot_knn_regression(n_neighbors=1):
    X, y = make_wave(n_samples=40)
    X_test = np.array([[-1.5], [0.9], [1.5]])

    dist = euclidean_distances(X, X_test)
    closest = np.argsort(dist, axis=0)

    plt.figure(figsize=(10, 6))

    reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)
    y_pred = reg.predict(X_test)

    for x, y_, neighbors in zip(X_test, y_pred, closest.T):
        for neighbor in neighbors[:n_neighbors]:
                plt.arrow(x[0], y_, X[neighbor, 0] - x[0], y[neighbor] - y_,
                          head_width=0, fc='k', ec='k')

    plt.plot(X, y, 'o')
    plt.plot(X, -3 * np.ones(len(X)), 'o')
    plt.plot(X_test, -3 * np.ones(len(X_test)), 'x', c='g', markersize=20)
    plt.plot(X_test, y_pred, 'x', c='b', markersize=20)

    plt.ylim(-3.1, 3.1)
