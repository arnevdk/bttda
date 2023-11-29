#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hoda.gpu_opt import combine_pvalues
from hoda.hoda import HODA, f_oneway
from moabb.datasets import *
from moabb.paradigms import P300

from figures import accent1, accent2, cmap

# Fit model ===================================================================

tmin = 0
tmax = 0.8
fmin = 0.5
fmax = 16
sfreq = 32

paradigm = P300(resample=sfreq, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
dataset = BNCI2014009()
epochs, labels, meta = paradigm.get_data(
    dataset=dataset, subjects=[1], return_epochs=True
)


X = epochs.get_data()
y = labels

hoda = HODA(
    rank=None,
    max_iter=128,
    tol=1e-16,
    init="svd",
    shrinkage=("lw", "lw"),
    toeplitz=None,
    obj="rt",
    solver="lanczos",
    taper=False,
    keep_train_info=False,
    verbose=True,
    lasso=False,
    prune=False,
    combine_pvalue="fisher",
    prune_pvalue=0.05,
)
hoda.fit(X, y)
Xt = hoda.transform(X)

# Plot ========================================================================
F, p = f_oneway(Xt, y)
g = sns.JointGrid()
sns.heatmap(F, vmin=0, ax=g.ax_joint, cmap=cmap, cbar=False)
res_sp = np.array(
    [combine_pvalues(p[i, :], method="fisher") for i in range(p.shape[0])]
)
res_tmp = np.array(
    [combine_pvalues(p[:, i], method="fisher") for i in range(p.shape[1])]
)
# sp_sel = res_sp[:,1]<0.05/p.shape[0]
# tmp_sel = res_tmp[:,1]<0.05/p.shape[1]
sp_sel = res_sp[:, 1] < 0.05 / 1e20
tmp_sel = res_tmp[:, 1] < 0.05 / 1e20

g.ax_marg_x.bar(
    np.arange(p.shape[1])[tmp_sel] + 0.5, res_tmp[:, 0][tmp_sel], color="#ef5513"
)
g.ax_marg_x.bar(
    np.arange(p.shape[1])[~tmp_sel] + 0.5, res_tmp[:, 0][~tmp_sel], color="#f9c134"
)

g.ax_marg_y.barh(
    np.arange(p.shape[0])[sp_sel] + 0.5, res_sp[:, 0][sp_sel], color="#ef5513"
)
g.ax_marg_y.barh(
    np.arange(p.shape[0])[~sp_sel] + 0.5, res_sp[:, 0][~sp_sel], color="#f9c134"
)

g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
g.ax_joint.set_xlabel("temporal components")
g.ax_joint.set_ylabel("spatial components")

darkgray = (0.3, 0.3, 0.3)
g.ax_marg_y.spines["left"].set_color(darkgray)  # setting up Y-axis tick color to red
g.ax_marg_x.spines["bottom"].set_color(darkgray)
g.ax_joint.xaxis.label.set_color(darkgray)  # setting up X-axis label color to yellow
g.ax_joint.yaxis.label.set_color(darkgray)

(line_r1,) = g.ax_marg_x.plot((0, 2), (1400, 1400), color=darkgray, linewidth=0.7)
(line_r2,) = g.ax_marg_y.plot((500, 500), (0, 8), color=darkgray, linewidth=0.7)
g.ax_marg_x.annotate("$r_2$", xy=(2.5, 1100), color=darkgray)
g.ax_marg_y.annotate("$r_1$", xy=(400, 9.8), color=darkgray)

g.figure.set_size_inches(4 / 2, 4 / 2)
g.figure.savefig(
    "figures/out/component_selection.pgf", bbox_inches="tight", pad_inches=0
)
plt.show()
