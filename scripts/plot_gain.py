"""Embedding gain figures — absolute and relative lift over baselines.
Academic-style derived-metric panels that complement the overlapped-bar chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "results" / "comparison_table.csv"
OUT_PATH = REPO / "results" / "comparison_gain.png"

PAIRS = [
    ("LogReg",   "Baseline + LogReg",   "Embeddings + LogReg"),
    ("LightGBM", "Baseline + LightGBM", "Embeddings + LightGBM"),
]

PANEL_LABELS = ["a", "b", "c", "d", "e", "f"]


def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH).sort_values(["Classes", "Method"]).reset_index(drop=True)


def compute_gains(df: pd.DataFrame) -> dict:
    classes = sorted(df["Classes"].unique())
    metrics = ["Accuracy", "Macro F1", "Kappa"]
    rows = []
    for n_cls in classes:
        sub = df[df["Classes"] == n_cls]
        for fam, base_m, emb_m in PAIRS:
            base = sub.loc[sub["Method"] == base_m, metrics].values[0]
            emb  = sub.loc[sub["Method"] == emb_m,  metrics].values[0]
            for j, m in enumerate(metrics):
                delta = emb[j] - base[j]
                pct   = delta / base[j] * 100 if base[j] != 0 else 0.0
                rows.append({"Classes": n_cls, "Family": fam, "Metric": m,
                             "AbsGain": delta, "RelGain": pct})
    return pd.DataFrame(rows)


def plot_gain_panel(ax: plt.Axes, gain_df: pd.DataFrame,
                    metric: str, row_type: str, panel_letter: str) -> None:
    ax.set_facecolor("white")

    families = ["LogReg", "LightGBM"]
    colours  = ["#1f77b4", "#5fa8d3"]
    classes  = sorted(gain_df["Classes"].unique())
    n_cls    = len(classes)

    x = np.arange(n_cls)
    w = 0.32

    value_col = "AbsGain" if row_type == "abs" else "RelGain"
    ylabel    = r"$\Delta$ (Embedding $-$ Baseline)" if row_type == "abs" else r"$\Delta$ / Baseline (%)"

    for i, (fam, c) in enumerate(zip(families, colours)):
        vals = gain_df[(gain_df["Family"] == fam) & (gain_df["Metric"] == metric)]
        vals = vals.set_index("Classes").loc[classes, value_col].values
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, width=w * 0.90, color=c,
                      edgecolor="white", linewidth=0.4, zorder=3)

        for bar, v in zip(bars, vals):
            y_pos = bar.get_height() + (abs(bar.get_height()) * 0.04 if bar.get_height() >= 0 else -abs(bar.get_height()) * 0.10)
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{v:.3f}" if row_type == "abs" else f"{v:.1f}",
                    ha="center", va=va, fontsize=7, color=c, fontweight="bold")

    ax.axhline(y=0, color="#222", linewidth=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} cls" for c in classes], fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_title(f"({panel_letter})  {ylabel}, {metric}",
                 fontsize=9, fontweight="bold", color="#111", pad=4)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.20, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    all_vals = gain_df[(gain_df["Metric"] == metric)][value_col]
    ymax = all_vals.max()
    ymin = all_vals.min()
    margin = max(abs(ymax), abs(ymin), 0.01) * 0.25
    ax.set_ylim(ymin - margin, ymax + margin)


def main() -> None:
    df = load_data()
    gain_df = compute_gains(df)
    metrics = ["Accuracy", "Macro F1", "Kappa"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
    })

    fig, axes = plt.subplots(
        2, len(metrics),
        figsize=(16, 8),
        facecolor="white",
    )
    fig.subplots_adjust(hspace=0.50, wspace=0.38, left=0.065, right=0.97,
                        top=0.930, bottom=0.100)

    panel_idx = 0
    for row_type in ["abs", "rel"]:
        for col, metric in enumerate(metrics):
            ax = axes[0 if row_type == "abs" else 1, col]
            plot_gain_panel(ax, gain_df, metric, row_type, PANEL_LABELS[panel_idx])
            panel_idx += 1

    fig.suptitle(
        "Embedding gain over baselines — absolute and relative lift",
        fontsize=14, fontweight="bold", y=0.995, color="#000",
    )
    fig.text(0.5, 0.955,
             "Positive values mean the embedding classifier outperforms its direct baseline counterpart. "
             "LightGBM pairs show smaller or negative gains.",
             ha="center", fontsize=9, style="italic", color="#444")

    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="white", label="LogReg gain  (Embeddings $-$ Baseline)"),
        Patch(facecolor="#5fa8d3", edgecolor="white", label="LightGBM gain (Embeddings $-$ Baseline)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               frameon=True, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.012),
               facecolor="white", edgecolor="#cccccc")

    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
