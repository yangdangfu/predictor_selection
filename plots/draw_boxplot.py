# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def draw_boxplot(ax: plt.Axes,
                 datas: list,
                 labels: list,
                 colors: list,
                 num_pos: float = 0.965,
                 quantile_ref=False):
    bplot = ax.boxplot(
        datas,
        whis=(5, 95),
        showfliers=False,
        vert=True,
        labels=labels,
        patch_artist=True,  # fill with color
        showmeans=True  #, meanline=True
    )
    # ax.set_xticklabels(labels,
    #                    rotation=45,
    #                    rotation_mode="default",
    #                    multialignment="right")
    ax.grid(b=True, axis="y")
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    num_boxes = len(bplot['boxes'])
    pos = np.arange(num_boxes) + 1
    means = [mn.get_ydata()[0] for mn in bplot["means"]]
    upper_labels = [f"{s:.3f}" for s in means]
    for tick_idx in range(num_boxes):
        ax.text(pos[tick_idx],
                num_pos,
                upper_labels[tick_idx],
                transform=ax.get_xaxis_transform(),
                horizontalalignment='center',
                size='x-small',
                color="black")

    if quantile_ref:
        ax.axhline(y=bplot["whiskers"][0].get_ydata()[0], ls=":")
        ax.axhline(y=bplot["whiskers"][0].get_ydata()[1], ls=":")
        ax.axhline(y=bplot["whiskers"][1].get_ydata()[0], ls=":")
        ax.axhline(y=bplot["whiskers"][1].get_ydata()[1], ls=":")