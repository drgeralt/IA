"""Módulo de visualização e geração de gráficos de alta resolução."""
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA

from src.config import DPI, PLOT_STYLE


def set_plotting_theme():
    """Aplica estilo visual consistente."""
    try:
        plt.style.use(PLOT_STYLE)
    except Exception:
        plt.style.use("default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_elbow_and_silhouette(
    k_values: List[int],
    inertias: List[float],
    silhouette_scores: List[float],
    optimal_k: int,
    title_suffix: str,
    output_path: Path,
) -> None:
    """Gera gráfico lado a lado do Método do Cotovelo e do Coeficiente de Silhueta."""
    set_plotting_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Método do Cotovelo (WCSS / Inércia)
    ax1.plot(k_values, inertias, marker="o", color="#1f77b4", linewidth=2.5, markersize=7)
    ax1.axvline(x=optimal_k, color="#d62728", linestyle="--", alpha=0.8, label=f"k ótimo ({optimal_k})")
    ax1.set_title(f"Método do Cotovelo (Inércia WCSS)\n{title_suffix}", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Número de Clusters (k)", fontsize=11)
    ax1.set_ylabel("Inércia (Within-Cluster Sum of Squares)", fontsize=11)
    ax1.set_xticks(k_values)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Gráfico 2: Coeficiente de Silhueta
    ax2.plot(k_values, silhouette_scores, marker="s", color="#2ca02c", linewidth=2.5, markersize=7)
    ax2.axvline(x=optimal_k, color="#d62728", linestyle="--", alpha=0.8, label=f"Pico de Silhueta (k={optimal_k})")
    ax2.scatter(
        [optimal_k],
        [silhouette_scores[k_values.index(optimal_k)]],
        color="#d62728",
        s=120,
        zorder=5,
    )
    ax2.set_title(f"Coeficiente de Silhueta Médio por k\n{title_suffix}", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Número de Clusters (k)", fontsize=11)
    ax2.set_ylabel("Silhouette Score Médio", fontsize=11)
    ax2.set_xticks(k_values)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    optimal_k: int,
    silhouette_avg: float,
    sample_silhouette_values: np.ndarray,
    output_path: Path,
) -> None:
    """Gera o diagrama detalhado de silhueta por cluster para cada amostra."""
    set_plotting_theme()
    fig, ax = plt.subplots(figsize=(10, 7))

    y_lower = 10
    colors = cm.nipy_spectral(np.linspace(0, 0.9, optimal_k))

    for i in range(optimal_k):
        # Coleta e ordena os coeficientes do cluster i
        ith_cluster_values = sample_silhouette_values[labels == i]
        ith_cluster_values.sort()

        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors[i]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.75,
            label=f"Cluster {i} (n={size_cluster_i})",
        )

        # Rótulo com número do cluster
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=11, fontweight="bold")
        y_lower = y_upper + 10  # Espaçamento entre clusters

    ax.set_title(
        f"Análise de Silhueta por Amostra para k = {optimal_k}\n(Silhueta Média = {silhouette_avg:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Valor do Coeficiente de Silhueta", fontsize=11)
    ax.set_ylabel("Amostras agrupadas por cluster", fontsize=11)

    # Linha vertical com a média global da silhueta
    ax.axvline(
        x=silhouette_avg,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Média Global ({silhouette_avg:.3f})",
    )

    ax.set_yticks([])  # Remove marcas de y
    ax.set_xlim([-0.15, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    features: List[str],
    optimal_k: int,
    output_path: Path,
) -> None:
    """Gera dispersão 2D com centroides destacados."""
    set_plotting_theme()
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = cm.tab10(np.linspace(0, 1, optimal_k))
    for i in range(optimal_k):
        cluster_mask = labels == i
        ax.scatter(
            X[cluster_mask, 0],
            X[cluster_mask, 1],
            s=65,
            color=colors[i],
            alpha=0.85,
            edgecolors="none",
            label=f"Cluster {i} ({np.sum(cluster_mask)} clientes)",
        )

    # Plotar centroides
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=260,
        c="#d62728",
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="Centroides",
        zorder=10,
    )

    ax.set_title(
        f"Agrupamento K-Means (k = {optimal_k})\n{features[0]} vs {features[1]}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(features[0], fontsize=11)
    ax.set_ylabel(features[1], fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True, fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_clusters_pca(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids_scaled: np.ndarray,
    optimal_k: int,
    output_path: Path,
) -> None:
    """Gera projeção 2D via PCA para o caso multidimensional com centróides transformados."""
    set_plotting_theme()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(centroids_scaled)

    var_ratio = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = cm.tab10(np.linspace(0, 1, optimal_k))

    for i in range(optimal_k):
        cluster_mask = labels == i
        ax.scatter(
            X_pca[cluster_mask, 0],
            X_pca[cluster_mask, 1],
            s=65,
            color=colors[i],
            alpha=0.85,
            edgecolors="none",
            label=f"Cluster {i} ({np.sum(cluster_mask)} clientes)",
        )

    # Centroides transformados no mesmo espaço PCA
    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        s=260,
        c="#d62728",
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="Centroides (PCA)",
        zorder=10,
    )

    ax.set_title(
        f"Agrupamento Multidimensional com K-Means (k = {optimal_k}) via PCA\n"
        f"Variância Explicada Total: {var_ratio.sum():.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(f"Primeiro Componente Principal - PC1 ({var_ratio[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"Segundo Componente Principal - PC2 ({var_ratio[1]:.1f}%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True, fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
