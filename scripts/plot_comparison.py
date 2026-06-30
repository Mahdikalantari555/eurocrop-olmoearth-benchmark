"""Overlapped-bar comparison: embedding classifiers layered over their
baseline equivalents, showing the gain at each class count.
Academic-style multi-panel figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "results" / "comparison_table.csv"
OUT_PATH = REPO / "results" / "comparison_overlap.png"

PAIRS = [
    ("LogReg",   "Baseline + LogReg",   "Embeddings + LogReg"),
    ("LightGBM", "Baseline + LightGBM", "Embeddings + LightGBM"),
]

PANEL_LABELS = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]


def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH).sort_values(["Classes", "Method"]).reset_index(drop=True)


def plot_panel(ax: plt.Axes, df_sub: pd.DataFrame, metric: str,
               n_cls: int, panel_letter: str,
               y_upper: float | None = None) -> None:
    ax.set_facecolor("white")

    x = np.arange(len(PAIRS))
    w_baseline = 0.70
    w_embed = 0.46

    for i, (fam, base_m, emb_m) in enumerate(PAIRS):
        base_val = df_sub.loc[df_sub["Method"] == base_m, metric].values[0]
        emb_val  = df_sub.loc[df_sub["Method"] == emb_m,  metric].values[0]

        ax.bar(i, base_val, width=w_baseline,
               color="#c0c0c0", edgecolor="#888888", linewidth=0.3,
               alpha=0.50, zorder=1)

        c_emb = "#1f77b4" if emb_m == "Embeddings + LogReg" else "#5fa8d3"
        ax.bar(i, emb_val, width=w_embed,
               color=c_emb, edgecolor="white", linewidth=0.5, zorder=3)

        delta = emb_val - base_val
        sign = "+" if delta >= 0 else ""
        y_off = max(abs(emb_val), 0.001) * 0.10
        ax.annotate(f"{sign}{delta:.3f}",
                    xy=(i, emb_val),
                    xytext=(i, emb_val + y_off),
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color=c_emb, zorder=5)

        if base_val < emb_val:
            ax.plot([i - 0.14, i], [base_val, base_val],
                    color="#666", linewidth=0.5, zorder=2)
            ax.plot([i, i], [base_val, emb_val],
                    color=c_emb, linewidth=0.6, linestyle=":", zorder=2)

    hyb_val = df_sub.loc[df_sub["Method"] == "Hybrid + LightGBM", metric].values[0]
    x_hyb = len(PAIRS)
    ax.bar(x_hyb, hyb_val, width=w_embed,
           color="#2ca02c", edgecolor="white", linewidth=0.5, zorder=3)
    y_off_h = max(abs(hyb_val), 0.001) * 0.10
    ax.annotate(f"{hyb_val:.3f}",
                xy=(x_hyb, hyb_val),
                xytext=(x_hyb, hyb_val + y_off_h),
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color="#2ca02c", zorder=5)

    all_xticks = list(x) + [x_hyb]
    ax.set_xticks(all_xticks)
    ax.set_xticklabels(["LogReg", "LightGBM", "Hybrid"],
                       fontsize=8)
    ax.tick_params(axis="x", length=0)

    ax.set_title(f"({panel_letter})  {metric}, {n_cls} classes",
                 fontsize=9, fontweight="bold", color="#111", pad=6)

    ax.set_ylim(0, y_upper)
    ax.set_ylabel(metric, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.20, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def main() -> None:
    df = load_data()
    classes = sorted(df["Classes"].unique())
    metrics = ["Accuracy", "Macro F1", "Kappa"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
    })

    fig, axes = plt.subplots(
        len(metrics), len(classes),
        figsize=(16, 10),
        facecolor="white",
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.40, left=0.06, right=0.97,
                        top=0.935, bottom=0.090)

    panel_idx = 0
    for row, metric in enumerate(metrics):
        global_max = df[metric].max()
        y_upper = global_max * 1.38
        for col, n_cls in enumerate(classes):
            ax = axes[row, col]
            df_sub = df[df["Classes"] == n_cls]
            plot_panel(ax, df_sub, metric, n_cls, PANEL_LABELS[panel_idx], y_upper)
            panel_idx += 1

    fig.suptitle(
        "Embedding vs baseline classifiers — overlapping bar comparison",
        fontsize=14, fontweight="bold", y=0.985, color="#000",
    )

    legend_handles = [
        Patch(facecolor="#c0c0c0", alpha=0.50, edgecolor="#888888",
              label="Baseline (behind)"),
        Patch(facecolor="#1f77b4", alpha=1.0, edgecolor="white",
              label="Embeddings + LogReg"),
        Patch(facecolor="#5fa8d3", alpha=1.0, edgecolor="white",
              label="Embeddings + LightGBM"),
        Patch(facecolor="#2ca02c", alpha=1.0, edgecolor="white",
              label="Hybrid + LightGBM"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=True, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.012),
               facecolor="white", edgecolor="#cccccc",
               title="Method", title_fontsize=9)

    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
