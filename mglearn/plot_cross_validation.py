import numpy as np
import matplotlib.pyplot as plt


def plot_group_kfold():
    from sklearn.model_selection import GroupKFold
    groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]

    plt.figure(figsize=(10, 2))
    plt.title("GroupKFold")

    axes = plt.gca()
    axes.set_frame_on(False)

    n_folds = 12
    n_samples = 12
    n_iter = 3
    n_samples_per_fold = 1

    cv = GroupKFold(n_splits=3)
    mask = np.zeros((n_iter, n_samples))
    for i, (train, test) in enumerate(cv.split(range(12), groups=groups)):
        mask[i, train] = 1
        mask[i, test] = 2

    for i in range(n_folds):
        # test is grey
        colors = ["grey" if x == 2 else "white" for x in mask[:, i]]
        # not selected has no hatch

        boxes = axes.barh(y=range(n_iter), width=[1 - 0.1] * n_iter,
                          left=i * n_samples_per_fold, height=.6, color=colors,
                          hatch="//", edgecolor="k", align='edge')
        for j in np.where(mask[:, i] == 0)[0]:
            boxes[j].set_hatch("")

    axes.barh(y=[n_iter] * n_folds, width=[1 - 0.1] * n_folds,
              left=np.arange(n_folds) * n_samples_per_fold, height=.6,
              color="w", edgecolor='k', align="edge")

    for i in range(12):
        axes.text((i + .5) * n_samples_per_fold, 3.5, "%d" %
                  groups[i], horizontalalignment="center")

    axes.invert_yaxis()
    axes.set_xlim(0, n_samples + 1)
    axes.set_ylabel("CV iterations")
    axes.set_xlabel("Data points")
    axes.set_xticks(np.arange(n_samples) + .5)
    axes.set_xticklabels(np.arange(1, n_samples + 1))
    axes.set_yticks(np.arange(n_iter + 1) + .3)
    axes.set_yticklabels(
        ["Split %d" % x for x in range(1, n_iter + 1)] + ["Group"])
    plt.legend([boxes[0], boxes[1]], ["Training set", "Test set"], loc=(1, .3))
    plt.tight_layout()


def plot_shuffle_split():
    from sklearn.model_selection import ShuffleSplit
    plt.figure(figsize=(10, 2))
    plt.title("ShuffleSplit with 10 points"
              ", train_size=5, test_size=2, n_splits=4")

    axes = plt.gca()
    axes.set_frame_on(False)

    n_folds = 10
    n_samples = 10
    n_iter = 4
    n_samples_per_fold = 1

    ss = ShuffleSplit(n_splits=4, train_size=5, test_size=2, random_state=43)
    mask = np.zeros((n_iter, n_samples))
    for i, (train, test) in enumerate(ss.split(range(10))):
        mask[i, train] = 1
        mask[i, test] = 2

    for i in range(n_folds):
        # test is grey
        colors = ["grey" if x == 2 else "white" for x in mask[:, i]]
        # not selected has no hatch

        boxes = axes.barh(y=range(n_iter), width=[1 - 0.1] * n_iter,
                          left=i * n_samples_per_fold, height=.6, color=colors,
                          hatch="//", edgecolor='k', align='edge')
        for j in np.where(mask[:, i] == 0)[0]:
            boxes[j].set_hatch("")

    axes.invert_yaxis()
    axes.set_xlim(0, n_samples + 1)
    axes.set_ylabel("CV iterations")
    axes.set_xlabel("Data points")
    axes.set_xticks(np.arange(n_samples) + .5)
    axes.set_xticklabels(np.arange(1, n_samples + 1))
    axes.set_yticks(np.arange(n_iter) + .3)
    axes.set_yticklabels(["Split %d" % x for x in range(1, n_iter + 1)])
    # legend hacked for this random state
    plt.legend([boxes[1], boxes[0], boxes[2]], [
               "Training set", "Test set", "Not selected"], loc=(1, .3))
    plt.tight_layout()


def plot_stratified_cross_validation():
    fig, both_axes = plt.subplots(2, 1, figsize=(12, 5))
    # plt.title("cross_validation_not_stratified")
    axes = both_axes[0]
    axes.set_title("Standard cross-validation with sorted class labels")

    axes.set_frame_on(False)

    n_folds = 3
    n_samples = 150

    n_samples_per_fold = n_samples / float(n_folds)

    for i in range(n_folds):
        colors = ["w"] * n_folds
        colors[i] = "grey"
        axes.barh(y=range(n_folds), width=[n_samples_per_fold - 1] *
                  n_folds, left=i * n_samples_per_fold, height=.6,
                  color=colors, hatch="//", edgecolor='k', align='edge')

    axes.barh(y=[n_folds] * n_folds, width=[n_samples_per_fold - 1] *
              n_folds, left=np.arange(3) * n_samples_per_fold, height=.6,
              color="w", edgecolor='k', align='edge')

    axes.invert_yaxis()
    axes.set_xlim(0, n_samples + 1)
    axes.set_ylabel("CV iterations")
    axes.set_xlabel("Data points")
    axes.set_xticks(np.arange(n_samples_per_fold / 2.,
                              n_samples, n_samples_per_fold))
    axes.set_xticklabels(["Fold %d" % x for x in range(1, n_folds + 1)])
    axes.set_yticks(np.arange(n_folds + 1) + .3)
    axes.set_yticklabels(
        ["Split %d" % x for x in range(1, n_folds + 1)] + ["Class label"])
    for i in range(3):
        axes.text((i + .5) * n_samples_per_fold, 3.5, "Class %d" %
                  i, horizontalalignment="center")

    ax = both_axes[1]
    ax.set_title("Stratified Cross-validation")
    ax.set_frame_on(False)
    ax.invert_yaxis()
    ax.set_xlim(0, n_samples + 1)
    ax.set_ylabel("CV iterations")
    ax.set_xlabel("Data points")

    ax.set_yticks(np.arange(n_folds + 1) + .3)
    ax.set_yticklabels(
        ["Split %d" % x for x in range(1, n_folds + 1)] + ["Class label"])

    n_subsplit = n_samples_per_fold / 3.
    for i in range(n_folds):
        test_bars = ax.barh(
            y=[i] * n_folds, width=[n_subsplit - 1] * n_folds,
            left=np.arange(n_folds) * n_samples_per_fold + i * n_subsplit,
            height=.6, color="grey", hatch="//", edgecolor='k', align='edge')

    w = 2 * n_subsplit - 1
    ax.barh(y=[0] * n_folds, width=[w] * n_folds, left=np.arange(n_folds)
            * n_samples_per_fold + (0 + 1) * n_subsplit, height=.6, color="w",
            hatch="//", edgecolor='k', align='edge')
    ax.barh(y=[1] * (n_folds + 1), width=[w / 2., w, w, w / 2.],
            left=np.maximum(0, np.arange(n_folds + 1) * n_samples_per_fold -
                            n_subsplit), height=.6, color="w", hatch="//",
            edgecolor='k', align='edge')
    training_bars = ax.barh(y=[2] * n_folds, width=[w] * n_folds,
                            left=np.arange(n_folds) * n_samples_per_fold,
                            height=.6, color="w", hatch="//", edgecolor='k',
                            align='edge')

    ax.barh(y=[n_folds] * n_folds, width=[n_samples_per_fold - 1] *
            n_folds, left=np.arange(n_folds) * n_samples_per_fold, height=.6,
            color="w", edgecolor='k', align='edge')

    for i in range(3):
        ax.text((i + .5) * n_samples_per_fold, 3.5, "Class %d" %
                i, horizontalalignment="center")
    ax.set_ylim(4, -0.1)
    plt.legend([training_bars[0], test_bars[0]], [
               'Training data', 'Test data'], loc=(1.05, 1), frameon=False)

    fig.tight_layout()


def plot_cross_validation():
    plt.figure(figsize=(12, 2))
    plt.title("cross_validation")
    axes = plt.gca()
    axes.set_frame_on(False)

    n_folds = 5
    n_samples = 25

    n_samples_per_fold = n_samples / float(n_folds)

    for i in range(n_folds):
        colors = ["w"] * n_folds
        colors[i] = "grey"
        bars = plt.barh(
            y=range(n_folds), width=[n_samples_per_fold - 0.1] * n_folds,
            left=i * n_samples_per_fold, height=.6, color=colors, hatch="//",
            edgecolor='k', align='edge')
    axes.invert_yaxis()
    axes.set_xlim(0, n_samples + 1)
    plt.ylabel("CV iterations")
    plt.xlabel("Data points")
    plt.xticks(np.arange(n_samples_per_fold / 2., n_samples,
                         n_samples_per_fold),
               ["Fold %d" % x for x in range(1, n_folds + 1)])
    plt.yticks(np.arange(n_folds) + .3,
               ["Split %d" % x for x in range(1, n_folds + 1)])
    plt.legend([bars[0], bars[4]], ['Training data', 'Test data'],
               loc=(1.05, 0.4), frameon=False)


def plot_threefold_split():
    plt.figure(figsize=(15, 1))
    axis = plt.gca()
    bars = axis.barh([0, 0, 0], [11.9, 2.9, 4.9], left=[0, 12, 15], color=[
                     'white', 'grey', 'grey'], hatch="//", edgecolor='k',
                     align='edge')
    bars[2].set_hatch(r"")
    axis.set_yticks(())
    axis.set_frame_on(False)
    axis.set_ylim(-.1, .8)
    axis.set_xlim(-0.1, 20.1)
    axis.set_xticks([6, 13.3, 17.5])
    axis.set_xticklabels(["training set", "validation set",
                          "test set"], fontdict={'fontsize': 20})
    axis.tick_params(length=0, labeltop=True, labelbottom=False)
    axis.text(6, -.3, "Model fitting",
              fontdict={'fontsize': 13}, horizontalalignment="center")
    axis.text(13.3, -.3, "Parameter selection",
              fontdict={'fontsize': 13}, horizontalalignment="center")
    axis.text(17.5, -.3, "Evaluation",
              fontdict={'fontsize': 13}, horizontalalignment="center")
