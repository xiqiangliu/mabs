import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_mean(
    pickle_path: Path, group_name: str, x_axis: str = "mu", save_path: Path = None
):
    with open(pickle_path, "rb") as f:
        result_raw = pickle.load(f)

    mean_regret = [(regret.mean(), x, g) for regret, x, g in result_raw]
    mean_regret_df = pd.DataFrame(
        mean_regret, columns=["expected_regret", x_axis, group_name]
    )

    sns.lineplot(data=mean_regret_df, x=x_axis, y="expected_regret", hue=group_name)
    plt.title("Expected Regret")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_var(
    pickle_path: Path, group_name: str, x_axis: str = "mu", save_path: Path = None
):
    with open(pickle_path, "rb") as f:
        result_raw = pickle.load(f)

    var_regret = [(regret.var(), x, g) for regret, x, g in result_raw]
    var_regret_df = pd.DataFrame(var_regret, columns=["regret_var", x_axis, group_name])

    sns.lineplot(data=var_regret_df, x=x_axis, y="regret_var", hue=group_name)
    plt.title("Regret Variance")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
