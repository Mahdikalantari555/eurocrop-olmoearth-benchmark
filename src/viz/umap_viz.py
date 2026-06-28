"""
UMAP visualization of embeddings.
"""

import matplotlib.pyplot as plt
import numpy as np
import umap


def plot_umap(embeddings: np.ndarray, labels: np.ndarray,
              title: str = "UMAP of Embeddings", save_path=None):
    """
    Plot UMAP projection of embeddings.
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', num_classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[cmap(i)], label=f'Class {label}', s=10, alpha=0.7)

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
