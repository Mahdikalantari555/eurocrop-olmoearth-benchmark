"""
UMAP visualization of embeddings.
"""

import matplotlib.pyplot as plt
import numpy as np
import umap


def _get_cmap(n_classes):
    """Get a colormap that supports more than 20 classes."""
    cmap20 = plt.cm.get_cmap('tab20', 20)
    cmap20b = plt.cm.get_cmap('tab20b', 20)
    cmap20c = plt.cm.get_cmap('tab20c', 20)
    all_colors = []
    for i in range(n_classes):
        idx = i % 60
        if idx < 20:
            all_colors.append(cmap20(idx))
        elif idx < 40:
            all_colors.append(cmap20b(idx - 20))
        else:
            all_colors.append(cmap20c(idx - 40))
    return all_colors


def plot_umap(embeddings: np.ndarray, labels: np.ndarray,
              title: str = "UMAP of Embeddings", save_path=None):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = _get_cmap(num_classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[colors[i]], label=f'Class {label}', s=10, alpha=0.7)

    ax.set_title(title)
    if num_classes <= 30:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                  markerscale=3, fontsize='small', ncol=2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
