"""
Few-shot learning curve visualization.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_fewshot_curve(results: dict, save_path=None):
    """
    Plot few-shot learning curves with error bars.
    results: dict {n_shots: {"rf_f1": ..., "rf_f1_std": ..., "lgbm_f1": ..., etc.}}
    """
    shots = sorted(results.keys(), key=lambda x: int(x))
    shot_vals = [int(s) for s in shots]

    rf_scores = [results[s]["rf_f1"] for s in shots]
    rf_stds = [results[s].get("rf_f1_std", 0) for s in shots]
    lgbm_scores = [results[s]["lgbm_f1"] for s in shots]
    lgbm_stds = [results[s].get("lgbm_f1_std", 0) for s in shots]
    olmo_scores = [results[s]["olmo_lgbm_f1"] for s in shots]
    olmo_stds = [results[s].get("olmo_lgbm_f1_std", 0) for s in shots]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(shot_vals, rf_scores, 'o-', label='RF (classical)', linewidth=2)
    ax.fill_between(shot_vals,
                    [r - s for r, s in zip(rf_scores, rf_stds)],
                    [r + s for r, s in zip(rf_scores, rf_stds)],
                    alpha=0.15)

    ax.plot(shot_vals, lgbm_scores, 's-', label='LightGBM (classical)', linewidth=2)
    ax.fill_between(shot_vals,
                    [r - s for r, s in zip(lgbm_scores, lgbm_stds)],
                    [r + s for r, s in zip(lgbm_scores, lgbm_stds)],
                    alpha=0.15)

    ax.plot(shot_vals, olmo_scores, '^-', label='OLMoEarth + LightGBM', linewidth=2)
    ax.fill_between(shot_vals,
                    [r - s for r, s in zip(olmo_scores, olmo_stds)],
                    [r + s for r, s in zip(olmo_scores, olmo_stds)],
                    alpha=0.15)

    ax.set_xlabel('Number of shots per class')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Few-Shot Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
