import matplotlib
import numpy as np
import tensorly as tl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

tl.set_backend("numpy", local_threadsafe=True)

accent1 = "#ef5513"
accent2 = "#f9c134"
color = (239, 85, 19)

cmap_res = 256
vals = np.ones((cmap_res, 4))
for i in range(len(color)):
    vals[:, i] = np.linspace(1, color[i] / 256, cmap_res)
cmap = ListedColormap(vals)

matplotlib.rcParams["axes.linewidth"] = 1
