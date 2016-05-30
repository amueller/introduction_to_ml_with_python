import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from matplotlib.colors import ListedColormap


cm = ListedColormap(["#FF0000", "#0000FF"])


def plot_tree_not_monotone():
    import graphviz
    # make a simple 2d dataset
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=cm)

    # learn a decision tree model
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)

    # visualize the tree
    export_graphviz(tree, out_file="mytree.dot", impurity=False, filled=True)
    with open("mytree.dot") as f:
        dot_graph = f.read()
    print("Feature importances: %s" % tree.feature_importances_)
    return graphviz.Source(dot_graph)
