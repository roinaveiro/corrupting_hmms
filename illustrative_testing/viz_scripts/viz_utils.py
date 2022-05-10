import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np

plt.style.use("fivethirtyeight")


def ratio_df_to_res_dict(df):
    in_d = {}
    in_d["rt_list"] = list(df["ratio"])
    in_d["res_list"] = list(df["res"])
    in_d["res_std_plus"] = list(df["res_std_plus"])
    in_d["res_std_minus"] = list(df["res_std_minus"])

    return in_d


def ratio_plot(
    d,
    title_str,
    problem_params={"metric": "Jaccard index", "yticks": None, "xticks": None},
    size_params={
        "label_size": 13.5,
        "tick_size": 12,
        "loc": (1.04, 0.6),
        "title_size": 15.5,
        "linewidth": 1.5,
    },
    color_params={"color_mean": "red", "color_std": "blue", "color_sh": "lightblue"},
    save=False,
    dir2save=None,
):
    plt.plot(
        d["rt_list"],
        d["res_list"],
        color=color_params["color_mean"],
        linewidth=size_params["linewidth"],
    )
    plt.plot(
        d["rt_list"],
        d["res_std_plus"],
        "--",
        color=color_params["color_std"],
        linewidth=size_params["linewidth"],
    )
    plt.plot(
        d["rt_list"],
        d["res_std_minus"],
        "--",
        color=color_params["color_std"],
        linewidth=size_params["linewidth"],
    )
    plt.fill_between(
        d["rt_list"], d["res_list"], d["res_std_plus"], color=color_params["color_sh"]
    )
    plt.fill_between(
        d["rt_list"], d["res_list"], d["res_std_minus"], color=color_params["color_sh"]
    )
    plt.xlabel("Ratio (w1/w2)", size=size_params["label_size"])
    plt.ylabel(problem_params["metric"], size=size_params["label_size"])

    if problem_params["xticks"] == None:
        plt.xticks(size=size_params["tick_size"])
    else:
        plt.xticks(problem_params["xticks"], size=size_params["tick_size"])

    if problem_params["yticks"] == None:
        plt.yticks(size=size_params["tick_size"])
    else:
        plt.yticks(problem_params["yticks"], size=size_params["tick_size"])

    plt.title(title_str, size=size_params["title_size"])
    if save == True:
        plt.savefig(dir2save, bbox_inches="tight")


def ratio_plot_grid(
    res_d_low_high,
    title_str,
    save=False,
    dir2save="",
    out_params={"figsize": (12, 8), "tl_size": 18},
    in_problem_params={"metric": "Jaccard index", "yticks": None, "xticks": None},
    in_size_params={
        "color_mean": "red",
        "color_std": "blue",
        "color_sh": "lightblue",
        "label_size": 13.5,
        "tick_size": 12,
        "figsize": (8, 6),
        "loc": (1.04, 0.6),
        "title_size": 15.5,
        "linewidth": 1.5,
    },
    in_color_params={"color_mean": "red", "color_std": "blue", "color_sh": "lightblue"},
):

    res_d_low_high["high"] = ratio_df_to_res_dict(res_d_low_high["high"])
    res_d_low_high["low"] = ratio_df_to_res_dict(res_d_low_high["low"])

    plt.figure(figsize=out_params["figsize"])
    plt.subplot(2, 2, 1)
    ratio_plot(
        res_d_low_high["low"],
        "Low uncertainty",
        problem_params=in_problem_params,
        size_params=in_size_params,
        color_params=in_color_params,
    )
    plt.subplot(2, 2, 2)
    ratio_plot(
        res_d_low_high["high"],
        "High uncertainty",
        problem_params=in_problem_params,
        size_params=in_size_params,
        color_params=in_color_params,
    )
    plt.suptitle(title_str, size=out_params["tl_size"], fontweight="bold")
    if save == True:
        plt.savefig(dir2save, bbox_inches="tight")
    plt.show()


def contour_df_to_res_dict(df):
    in_d = {}
    out_d = {}
    dim1 = len(set(df["w1_values"]))
    dim2 = len(set(df["w2_values"]))
    W1 = np.array(df["w1_values"]).reshape(dim1, dim2)
    W2 = np.array(df["w2_values"]).reshape(dim1, dim2)
    res = np.array(df["res"]).reshape(dim1, dim2)
    in_d["W1"] = W1
    in_d["W2"] = W2
    in_d["res"] = res
    return in_d


def scatter_plot(
    res_d,
    title_str,
    save=False,
    dir2save=None,
    size_params={
        "label_size": 13.5,
        "tick_size": 12,
        "title_size": 14.8,
        "points_size": 30,
    },
    value_params={"c": 1, "vmin": 0, "vmax": 1},
):

    df = pd.DataFrame(
        data={
            "A": res_d["W1"].reshape(-1),
            "B": res_d["W2"].reshape(-1),
            "C": res_d["res"].reshape(-1),
        }
    )
    if value_params["c"] == 1:
        points = plt.scatter(
            df.A,
            df.B,
            c=df.C,
            cmap=plt.cm.get_cmap("RdYlGn_r"),
            lw=0,
            s=size_params["points_size"],
            vmin=value_params["vmin"],
            vmax=value_params["vmax"],
        )
    elif value_params["c"] == -1:
        points = plt.scatter(
            df.A,
            df.B,
            c=df.C,
            cmap=plt.cm.get_cmap("RdYlGn"),
            lw=0,
            s=size_params["points_size"],
            vmin=value_params["vmin"],
            vmax=value_params["vmax"],
        )

    plt.xlabel("w1", size=size_params["label_size"])
    plt.ylabel("w2", size=size_params["label_size"])
    plt.xticks(size=size_params["tick_size"])
    plt.yticks(size=size_params["tick_size"])
    plt.colorbar(points)
    plt.title(title_str, size=size_params["title_size"])
    if save == True:
        plt.savefig(dir2save)


def scatter_plot_grid(
    res_d_low_high,
    title_str,
    save=False,
    dir2save="",
    out_params={"figsize": (12, 8), "tl_size": 18},
    in_params={
        "label_size": 13.5,
        "tick_size": 12,
        "title_size": 14.8,
        "points_size": 40,
    },
    color_params={"c": 1, "min_max_0_1": True},
):

    res_d_low_high["high"] = contour_df_to_res_dict(res_d_low_high["high"])
    res_d_low_high["low"] = contour_df_to_res_dict(res_d_low_high["low"])
    value_params = {"c": 1}
    if color_params["min_max_0_1"] == True:
        value_params["vmin"] = 0
        value_params["vmax"] = 1
    else:
        h_res = res_d_low_high["high"]["res"].reshape(-1)
        l_res = res_d_low_high["low"]["res"].reshape(-1)
        hl_res = np.append(h_res, l_res)
        value_params["vmin"] = min(hl_res)
        value_params["vmax"] = max(hl_res)

    plt.figure(figsize=out_params["figsize"])
    plt.subplot(2, 2, 1)
    scatter_plot(
        res_d_low_high["low"],
        "Low uncertainty",
        size_params=in_params,
        value_params=value_params,
    )
    plt.subplot(2, 2, 2)
    scatter_plot(
        res_d_low_high["high"],
        "High uncertainty",
        size_params=in_params,
        value_params=value_params,
    )
    plt.suptitle(title_str, size=out_params["tl_size"], fontweight="bold")
    if save == True:
        plt.savefig(dir2save, bbox_inches="tight")


def box_plot_grid(
    box_d,
    title_str,
    problem_params={
        "metric": "Prob. tainted data",
        "yticks": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    out_params={"fig_size": (11.5, 4), "tl_size": 15, "y": 1.05},
    in_params={
        "title_size": 15.5,
        "label_size": 13.5,
        "linewidth": 3.5,
        "labeltick": 12,
    },
    save=False,
    dir2save=None,
):

    plt.figure(figsize=out_params["fig_size"])
    ax = plt.subplot(1, 2, 1)
    d = {n: [] for n in list(set(box_d["low"]["diff_n_comp"]))}
    for i in range(len(box_d["low"]["diff_n_comp"])):
        d[box_d["low"]["diff_n_comp"][i]].append(box_d["low"]["res"][i])
    data = [d[k] for k in range(len(d))]
    bp = ax.boxplot(
        data, patch_artist=True, medianprops={"linewidth": in_params["linewidth"]}
    )
    plt.xticks([1, 2, 3, 4, 5, 6], range(len(d.keys())), size=in_params["labeltick"])
    plt.yticks(problem_params["yticks"], size=in_params["labeltick"])
    ax.set_xlabel("# of observations changed by attack", size=in_params["label_size"])
    ax.set_ylabel(problem_params["metric"], size=in_params["label_size"])
    plt.title("Low uncertainty", size=in_params["title_size"])
    ax = plt.subplot(1, 2, 2)
    d = {n: [] for n in list(set(box_d["high"]["diff_n_comp"]))}
    for i in range(len(box_d["high"]["diff_n_comp"])):
        d[box_d["high"]["diff_n_comp"][i]].append(box_d["high"]["res"][i])
    data = [d[k] for k in range(len(d))]
    bp = ax.boxplot(
        data, patch_artist=True, medianprops={"linewidth": in_params["linewidth"]}
    )
    plt.xticks([1, 2, 3, 4, 5, 6], range(len(d.keys())), size=in_params["labeltick"])
    plt.yticks(problem_params["yticks"], size=in_params["labeltick"])
    ax.set_xlabel("# of observations changed by attack", size=in_params["label_size"])
    ax.set_ylabel(problem_params["metric"], size=in_params["label_size"])
    plt.title("High uncertainty", size=in_params["title_size"])
    plt.suptitle(
        title_str, size=out_params["tl_size"], y=out_params["y"], fontweight="bold"
    )
    if save == True:
        plt.savefig(dir2save, bbox_inches="tight")
    plt.show()
