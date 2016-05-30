from IPython.display import set_matplotlib_formats
import numpy as np
import matplotlib.pyplot as plt
import mglearn

set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['image.interpolation'] = "none"
plt.rcParams['savefig.bbox'] = "tight"
np.set_printoptions(precision=3)

np, mglearn
