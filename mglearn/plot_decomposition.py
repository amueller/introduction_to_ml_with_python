import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_decomposition(people, pca):
    image_shape = people.images[0].shape
    plt.figure(figsize=(20, 3))
    ax = plt.gca()

    imagebox = OffsetImage(people.images[0], zoom=1.5, cmap="gray")
    ab = AnnotationBbox(imagebox, (.05, 0.4), pad=0.0, xycoords='data')
    ax.add_artist(ab)

    for i in range(4):
        imagebox = OffsetImage(pca.components_[i].reshape(image_shape), zoom=1.5, cmap="viridis")

        ab = AnnotationBbox(imagebox, (.3 + .2 * i, 0.4),
                            pad=0.0,
                            xycoords='data'
                            )
        ax.add_artist(ab)
        if i == 0:
            plt.text(.18, .25, 'x_%d *' % i, fontdict={'fontsize': 50})
        else:
            plt.text(.15 + .2 * i, .25, '+ x_%d *' % i, fontdict={'fontsize': 50})

    plt.text(.95, .25, '+ ...', fontdict={'fontsize': 50})

    plt.rc('text', usetex=True)
    plt.text(.13, .3, r'\approx', fontdict={'fontsize': 50})
    plt.axis("off")
