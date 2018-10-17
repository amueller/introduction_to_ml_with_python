import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor


def plot_kneighbors_regularization():
    rnd = np.random.RandomState(42)
    x = np.linspace(-3, 3, 100)
    y_no_noise = np.sin(4 * x) + x
    y = y_no_noise + rnd.normal(size=len(x))
    X = x[:, np.newaxis]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x_test = np.linspace(-3, 3, 1000)

    for n_neighbors, ax in zip([2, 5, 20], axes.ravel()):
        kneighbor_regression = KNeighborsRegressor(n_neighbors=n_neighbors)
        kneighbor_regression.fit(X, y)
        ax.plot(x, y_no_noise, label="true function")
        ax.plot(x, y, "o", label="data")
        ax.plot(x_test, kneighbor_regression.predict(x_test[:, np.newaxis]),
                label="prediction")
        ax.legend()
        ax.set_title("n_neighbors = %d" % n_neighbors)


if __name__ == "__main__":
    plot_kneighbors_regularization()
    plt.show()
