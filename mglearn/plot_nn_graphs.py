

def plot_logistic_regression_graph():
    import graphviz
    lr_graph = graphviz.Digraph(node_attr={'shape': 'circle', 'fixedsize': 'True'},
                                graph_attr={'rankdir': 'LR', 'splines': 'line'})
    inputs = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_0")
    output = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_2")

    for i in range(4):
        inputs.node("x[%d]" % i, labelloc="c")
    inputs.body.append('label = "inputs"')
    inputs.body.append('color = "white"')

    lr_graph.subgraph(inputs)

    output.body.append('label = "output"')
    output.body.append('color = "white"')
    output.node("y")

    lr_graph.subgraph(output)

    for i in range(4):
        lr_graph.edge("x[%d]" % i, "y", label="w[%d]" % i)
    return lr_graph


def plot_single_hidden_layer_graph():
    import graphviz
    nn_graph = graphviz.Digraph(node_attr={'shape': 'circle', 'fixedsize': 'True'},
                                graph_attr={'rankdir': 'LR', 'splines': 'line'})

    inputs = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_0")
    hidden = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_1")
    output = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_2")

    for i in range(4):
        inputs.node("x[%d]" % i)

    inputs.body.append('label = "inputs"')
    inputs.body.append('color = "white"')

    hidden.body.append('label = "hidden layer"')
    hidden.body.append('color = "white"')

    for i in range(3):
        hidden.node("h%d" % i, label="h[%d]" % i)

    output.node("y")
    output.body.append('label = "output"')
    output.body.append('color = "white"')

    nn_graph.subgraph(inputs)
    nn_graph.subgraph(hidden)
    nn_graph.subgraph(output)

    for i in range(4):
        for j in range(3):
            nn_graph.edge("x[%d]" % i, "h%d" % j)

    for i in range(3):
        nn_graph.edge("h%d" % i, "y")
    return nn_graph


def plot_two_hidden_layer_graph():
    import graphviz
    nn_graph = graphviz.Digraph(node_attr={'shape': 'circle', 'fixedsize': 'True'},
                                graph_attr={'rankdir': 'LR', 'splines': 'line'})

    inputs = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_0")
    hidden = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_1")
    hidden2 = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_2")

    output = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_3")

    for i in range(4):
        inputs.node("x[%d]" % i)

    inputs.body.append('label = "inputs"')
    inputs.body.append('color = "white"')

    for i in range(3):
        hidden.node("h1[%d]" % i)

    for i in range(3):
        hidden2.node("h2[%d]" % i)

    hidden.body.append('label = "hidden layer 1"')
    hidden.body.append('color = "white"')

    hidden2.body.append('label = "hidden layer 2"')
    hidden2.body.append('color = "white"')

    output.node("y")
    output.body.append('label = "output"')
    output.body.append('color = "white"')

    nn_graph.subgraph(inputs)
    nn_graph.subgraph(hidden)
    nn_graph.subgraph(hidden2)

    nn_graph.subgraph(output)

    for i in range(4):
        for j in range(3):
            nn_graph.edge("x[%d]" % i, "h1[%d]" % j, label="")

    for i in range(3):
        for j in range(3):
            nn_graph.edge("h1[%d]" % i, "h2[%d]" % j, label="")

    for i in range(3):
        nn_graph.edge("h2[%d]" % i, "y", label="")

    return nn_graph
