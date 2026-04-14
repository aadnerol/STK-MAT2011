## Implement method for nice plots in universal style

import matplotlib.pyplot as plt

def set_style(
    figsize=(10, 5),
    title_size=14,
    label_size=12,
    legend_size=11,
    tick_size=10,
):
    plt.rcParams.update({
        "figure.figsize": figsize,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "legend.fontsize": legend_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 0.8,
    })
