"""
Few-shot learning curve visualization.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_fewshot_curve(results: dict, save_path=None):
    """
    Plot few-shot learning curves.
    results: dict {n_shots: {"rf_f1": ..., "lgbm_f1": ..., "olmo_lgbm_f1": ...}}
    """
    shots = sorted(results.keys())
    rf_scores = [results[s]["rf_f1"] for s in shots]
    lgbm_scores = [results[s]["lgbm_f1"] for s in shots]
    olmo_scores = [results[s]["olmo_lgbm_f1"] for s in shots]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(shots, rf_scores, 'o-', label='RF (classical)', linewidth=2)
    ax.plot(shots, lgbm_scores, 's-', label='LightGBM (classical)', linewidth=2)
    ax.plot(shots, olmo_scores, '^-', label='OLMoEarth + LightGBM', linewidth=2)

    ax.set_xlabel('Number of shots per class')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Few-Shot Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
