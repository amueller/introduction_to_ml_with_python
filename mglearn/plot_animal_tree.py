from scipy.misc import imread
import matplotlib.pyplot as plt


def plot_animal_tree(ax=None):
    import graphviz
    if ax is None:
        ax = plt.gca()
    mygraph = graphviz.Digraph(node_attr={'shape': 'box'},
                               edge_attr={'labeldistance': "10.5"},
                               format="png")
    mygraph.node("0", "Has feathers?")
    mygraph.node("1", "Can fly?")
    mygraph.node("2", "Has fins?")
    mygraph.node("3", "Hawk")
    mygraph.node("4", "Penguin")
    mygraph.node("5", "Dolphin")
    mygraph.node("6", "Bear")
    mygraph.edge("0", "1", label="True")
    mygraph.edge("0", "2", label="False")
    mygraph.edge("1", "3", label="True")
    mygraph.edge("1", "4", label="False")
    mygraph.edge("2", "5", label="True")
    mygraph.edge("2", "6", label="False")
    mygraph.render("tmp")
    ax.imshow(imread("tmp.png"))
    ax.set_axis_off()
