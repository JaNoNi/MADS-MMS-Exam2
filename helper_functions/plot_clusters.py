import itertools
import math
from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import OPTICS, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm

CMAP_PLT = {
    -1: "black",
    0: "red",
    1: "darkorange",
    2: "gold",
    3: "lawngreen",
    4: "lightseagreen",
    5: "blue",
    6: "darkviolet",
    7: "deeppink",
    8: "brown",
    9: "grey",
    10: "yellow",
    11: "green",
    12: "cyan",
    13: "magenta",
    14: "pink",
}


@dataclass
class OPTICSResults:
    optics: OPTICS
    space: list[int]
    reachability: list[float]
    targets: list[int]
    params: dict


def __basic_cluster(x, y, hue, alpha, ax=None):
    if ax is None:
        ax = plt.gca()
    if hue is not None:
        hue = [CMAP_PLT[_key] for _key in hue]
    scatter = ax.scatter(x, y, marker="o", c=hue, alpha=alpha, s=25, edgecolor="k")
    return scatter


def __draw_clusters(x, y, ax, alpha=None, centers=None, hue=None, labels=None, legend_loc="best"):
    __basic_cluster(x, y, hue, alpha=alpha, ax=ax)
    if hue is not None:
        handles = [mpatches.Patch(color=CMAP_PLT[_key], label=_key) for _key in set(hue)]
        ax.legend(handles=handles, loc=legend_loc)
    if centers is not None:
        for center in centers:
            ax.scatter(center[0], center[1], marker="*", c="red", s=100)
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])


def draw_clusters_grid(
    data,
    axes,
    alpha=None,
    centers=None,
    hue=None,
    labels: Optional[list[str]] = None,
    legend_loc="best",
):
    if isinstance(axes, np.ndarray):
        if len(axes.shape) == 1:
            raise ValueError("Not implemented.")
        else:
            grid_size_combinations = [list(range(axes.shape[0])), list(range(axes.shape[1]))]
            figure_combinations = list(itertools.product(*grid_size_combinations))
            feature_combinations = list(itertools.combinations(range(data.shape[1]), 2))
            if len(feature_combinations) > len(figure_combinations):
                raise ValueError("The grid size is smaller than the number of features.")
            # Plot
            for (comb_1, comb_2), (ax_1, ax_2) in zip(feature_combinations, figure_combinations):
                if centers is not None:
                    centers_ = centers[:, [comb_1, comb_2]]
                else:
                    centers_ = centers
                if labels is not None:
                    labels_ = [f"{labels[0]}_{comb_1}", f"{labels[1]}_{comb_2}"]
                else:
                    labels_ = labels
                __draw_clusters(
                    data.iloc[:, comb_1],
                    data.iloc[:, comb_2],
                    ax=axes[ax_1, ax_2],
                    alpha=alpha,
                    centers=centers_,
                    hue=hue,
                    labels=labels_,
                    legend_loc=legend_loc,
                )
    else:
        __draw_clusters(
            data.iloc[:, 0],
            data.iloc[:, 1],
            alpha=alpha,
            centers=centers,
            hue=hue,
            labels=labels,
            legend_loc=legend_loc,
            ax=axes,
        )


def __basic_hist(data, x, ax=None):
    if ax is None:
        ax = plt.gca()
    sns.histplot(data=data, x=x, ax=ax)


def __draw_hist(data, x, ax, labels=None):
    __basic_hist(data, x, ax=ax)
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])


def draw_hist_grid(
    data,
    axes,
    labels: Optional[list[str]] = None,
):
    if isinstance(axes, np.ndarray):
        if len(axes.shape) == 1:
            raise ValueError("Not implemented.")
        else:
            grid_size_combinations = [list(range(axes.shape[0])), list(range(axes.shape[1]))]
            figure_combinations = list(itertools.product(*grid_size_combinations))
            feature_combinations = data.columns
            if len(feature_combinations) > len(figure_combinations):
                raise ValueError("The grid size is smaller than the number of features.")
            # Plot
            for comb_1, (ax_1, ax_2) in zip(feature_combinations, figure_combinations):
                if labels is not None:
                    labels_ = [f"{labels[0]}_{comb_1}", labels[1]]
                else:
                    labels_ = labels
                __draw_hist(
                    data=data,
                    x=comb_1,
                    ax=axes[ax_1, ax_2],
                    labels=labels_,
                )
    else:
        __draw_hist(
            data,
            data.columns[0],
            labels=labels,
            ax=axes,
        )


def draw_ksscore(
    data, ks, labels, ax, random_state=None, kmeansparams: dict[str, str] = {"init": "k-means++"}
) -> list[int]:
    """
    Function that takes an array (or tuple of arrays) and calculates the silhouette scores for the
    different ks. Plots the silhouette scores against the ks and saves the image as a pdf.

    Args:
        data_list (array-like of shape (n_samples, n_features) or tuple): Input samples.
        ks (list): List containing all choices for k.
    """
    sscore = []
    for k in tqdm(ks):
        kkm = KMeans(n_clusters=k, random_state=random_state, **kmeansparams)
        cluster_labels = kkm.fit_predict(data)
        sscore.append(silhouette_score(data, cluster_labels))
    ax.plot(ks, sscore)

    ax.axhline(y=max(sscore), color="red", linestyle="--")
    # Labels ----------------------
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    print(f"Max score: {max(sscore)}")
    return sscore


def draw_silhouette(
    data,
    number_k,
    labels,
    ax,
    random_state=None,
    no_zero=False,
    kmeansparams: dict[str, str] = {"init": "k-means++"},
) -> list[int]:
    """Creates Silhouette plot for the given dataset.

    Credit:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    """
    kkm = KMeans(n_clusters=number_k, random_state=random_state, **kmeansparams)
    cluster_labels = kkm.fit_predict(data)
    if no_zero:
        cluster_labels = cluster_labels + 1
        number_k = number_k + 1
    sscore = silhouette_score(data, cluster_labels)
    sscore_values = silhouette_samples(data, cluster_labels)

    # Gap between silhouette plots ----------------------
    gap = 0.01
    y_lower = len(data) * gap

    # The silhouette coefficient can range from -1, 1 but here most lie within [-0.1, 1]
    ax.set_xlim([-0.1, 1])
    # The (k_clusters+1)*len(data)*gap is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(data) + (number_k + 1) * len(data) * gap])

    # Plot Cluster ----------------------
    for ith_k in range(no_zero, number_k):
        # Order the sscores
        ith_sscore_values = sscore_values[cluster_labels == ith_k]
        ith_sscore_values.sort()
        cluster_score = sum(ith_sscore_values)

        # Get the size of each cluster
        size_cluster_i = ith_sscore_values.shape[0]
        ith_sscore = cluster_score / size_cluster_i
        y_upper = y_lower + size_cluster_i
        print(
            "Cluster:%d with %d entries. Score: %f" % (ith_k, size_cluster_i, round(ith_sscore, 2))
        )

        # Create different colors for the clusters
        color = CMAP_PLT[ith_k]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_sscore_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(ith_k))

        # Compute the new y_lower for next plot
        y_lower = y_upper + len(data) * gap  # 10 for the 0 samples

    # Labels ----------------------
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    ax.axvline(x=sscore, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    return cluster_labels


def draw_boxplot(data, hue, labels, ax):
    df = pd.merge(
        pd.DataFrame(data),
        pd.DataFrame(hue, columns=["labels"]),
        left_index=True,
        right_index=True,
    )
    if not isinstance(ax, np.ndarray):
        sns.boxplot(data=df, x="labels", palette=CMAP_PLT, y=0, ax=ax)
        if labels:
            ax.set_xlabel("Labels")
            ax.set_ylabel(f"{labels[1]}_0")
        return

    if len(ax.shape) == 1:
        ax = np.array([ax])
    grid_size_combinations = [list(range(ax.shape[0])), list(range(ax.shape[1]))]
    figure_combinations = list(itertools.product(*grid_size_combinations))
    column_names = [col for col in df.columns if col != "labels"]
    for column, (ax_1, ax_2) in zip(column_names, figure_combinations):
        sns.boxenplot(data=df, x="labels", palette=CMAP_PLT, y=column, ax=ax[ax_1, ax_2])
        if labels:
            ax[ax_1, ax_2].set_xlabel("Labels")
            ax[ax_1, ax_2].set_ylabel(f"{labels[1]}_{column}")


def create_linkage(agg):
    n_samples = len(agg.labels_)
    counts = np.zeros(agg.children_.shape[0])
    for i, merge in enumerate(agg.children_):
        current_count = 0
        # print (i, merge)
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([agg.children_, agg.distances_, counts]).astype(float)
    return linkage_matrix


def draw_dendo(agg, cut: int, dendo_distance: str, labels, ax) -> list[int]:
    hierarchy.set_link_color_palette(list(CMAP_PLT.values())[1:])
    dn = dendrogram(
        create_linkage(agg),
        # hierarchy.linkage(agg.children_, dendo_distance),
        color_threshold=cut,
        no_labels=True,
        leaf_rotation=0,
        ax=ax,
    )
    ax.axhline(y=cut, color="black", linestyle="--")
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    flipped_CMAP_PLT = {value: key for key, value in CMAP_PLT.items()}
    try:
        return [flipped_CMAP_PLT[color] for color in dn["leaves_color_list"]]
    except KeyError:
        return None


def draw_plot(
    data,
    plot_type: str = "scatter",
    hue=None,
    alpha=None,
    labels: Optional[list[str]] = None,
    figsize: Optional[tuple[int, int]] = None,
    grid_size: tuple[int, int] = (1, 1),
    legend_loc: str = "best",
    dpi: Optional[int] = None,
    title: Optional[str] = None,
    shareaxes: bool = False,
    ks: Optional[Union[int, list[int]]] = None,
    centers: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    no_zero: bool = False,
    top_cut_off: Optional[float] = None,
    dendo_cut: int = None,
    dendo_distance: str = "euclidean",
    kmeansparams: dict[str, str] = {"init": "k-means++"},
):
    if no_zero and hue is not None:
        hue = hue + 1
    _, axes = plt.subplots(figsize=figsize, dpi=dpi, nrows=grid_size[0], ncols=grid_size[1])
    if plot_type == "scatter":
        # Create indexes
        return_result = draw_clusters_grid(
            data,
            alpha=alpha,
            centers=centers,
            hue=hue,
            labels=labels,
            legend_loc=legend_loc,
            axes=axes,
        )
    elif plot_type == "histplot":
        return_result = draw_hist_grid(data, labels=labels, axes=axes)
    elif plot_type == "ksscore":
        return_result = draw_ksscore(
            data, ks, labels=labels, ax=axes, random_state=random_state, kmeansparams=kmeansparams
        )
    elif plot_type == "silhouette":
        return_result = draw_silhouette(
            data,
            ks,
            labels=labels,
            ax=axes,
            random_state=random_state,
            no_zero=no_zero,
            kmeansparams=kmeansparams,
        )
    elif plot_type == "boxplot":
        return_result = draw_boxplot(data, hue, labels=labels, ax=axes)
    elif plot_type == "dendo":
        return_result = draw_dendo(
            data, dendo_cut, dendo_distance=dendo_distance, labels=labels, ax=axes
        )
    elif plot_type == "reachability":
        return_result = draw_reachability_grid(
            data,
            alpha=alpha,
            labels=labels,
            legend_loc=legend_loc,
            top_cut_off=top_cut_off,
            axes=axes,
        )
    else:
        raise ValueError(f"Plot type {plot_type} not supported")

    if shareaxes:
        lim_range = (math.floor(np.min(data)), math.ceil(np.max(data)))
        plt.setp(axes, xlim=lim_range, ylim=lim_range)
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    return return_result


def draw_reachability_grid(
    data: Union[OPTICSResults, list[OPTICSResults]],
    alpha: Optional[float],
    labels: Optional[list[str]],
    top_cut_off: Optional[float],
    legend_loc: str,
    axes: list[list[plt.Axes]],
):
    if isinstance(axes, np.ndarray):
        if not isinstance(data, list):
            raise TypeError("Data must be a list of OPTICSResults")
        if len(axes.shape) == 1:
            for _data, ax in zip(data, axes):
                _draw_reachability(
                    _data,
                    alpha=alpha,
                    labels=labels,
                    legend_loc=legend_loc,
                    top_cut_off=top_cut_off,
                    ax=ax,
                )
        else:
            grid_size_combinations = [list(range(axes.shape[0])), list(range(axes.shape[1]))]
            figure_combinations = list(itertools.product(*grid_size_combinations))
            for _data, (ax_1, ax_2) in zip(data, figure_combinations):
                _draw_reachability(
                    _data,
                    alpha=alpha,
                    labels=labels,
                    legend_loc=legend_loc,
                    top_cut_off=top_cut_off,
                    ax=axes[ax_1, ax_2],
                )
    elif isinstance(axes, plt.Axes):
        if isinstance(data, list):
            raise TypeError("Data must be a OPTICSResults")
        _draw_reachability(
            data=data,
            alpha=alpha,
            labels=labels,
            legend_loc=legend_loc,
            top_cut_off=top_cut_off,
            ax=axes,
        )
    else:
        raise TypeError("Axes must be a list of Axes or a single Axes")


def _draw_reachability(
    data: OPTICSResults,
    alpha: Optional[float],
    labels: Optional[list[str]],
    top_cut_off: Optional[float],
    legend_loc: str,
    ax: plt.Axes,
):
    classes = np.unique(data.targets)
    for clas in list(classes):
        Xk = data.space[data.targets == clas]
        Rk = data.reachability[data.targets == clas]
        try:
            if clas == -1:
                ax.plot(Xk, Rk, "k.", alpha=0.3)
            else:
                ax.plot(Xk, Rk, c=CMAP_PLT[clas], label=clas, alpha=alpha, marker=".")
        except KeyError:
            ax.plot(Xk, Rk, label=clas, alpha=alpha, marker=".")
        title = (
            f"Parameters: 'min_samples': {data.params['min_samples']}"
            f" 'metric': {data.params['metric']}"
            f" 'xi': {data.params['xi']}"
            f" 'min_cluster_size': {data.params['min_cluster_size']}"
        )
        ax.set_title(title)
    # ax.legend(loc=legend_loc)
    if top_cut_off is not None:
        ax.set_ylim(0, top_cut_off)
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])


def main():
    return ()


if __name__ == "__main__":
    main()
