import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from .tools import discrete_scatter
from .plot_2d_separator import plot_2d_separator


def plot_tree_not_monotone():
    import graphviz
    # make a simple 2d dataset
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    plt.figure()
    discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(["Class 0", "Class 1"], loc="best")

    # learn a decision tree model
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)
    plot_2d_separator(tree, X, linestyle="dashed")

    # visualize the tree
    export_graphviz(tree, out_file="mytree.dot", impurity=False, filled=True)
    with open("mytree.dot") as f:
        dot_graph = f.read()
    print("Feature importances: %s" % tree.feature_importances_)
    return graphviz.Source(dot_graph)
