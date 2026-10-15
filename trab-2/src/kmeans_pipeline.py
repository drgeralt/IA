"""Pipeline de treinamento K-Means, avaliação por silhueta e profiling dos clusters."""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from src.config import RANDOM_STATE, K_MIN, K_MAX, N_INIT, MAX_ITER


@dataclass
class ClusteringResult:
    """Estrutura para armazenar os resultados da avaliação de clusters."""
    k_values: List[int]
    inertias: List[float]
    silhouette_scores: List[float]
    optimal_k: int
    best_model: KMeans
    best_labels: np.ndarray
    best_silhouette_score: float
    silhouette_samples: np.ndarray


def evaluate_kmeans_range(
    X: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    random_state: int = RANDOM_STATE,
) -> ClusteringResult:
    """Treina modelos K-Means para um intervalo de k e calcula Inércia e Silhueta.
    
    Args:
        X: Matriz de features (numérica).
        k_min: Valor inicial de clusters (mínimo 2 para silhueta).
        k_max: Valor final de clusters.
        random_state: Semente de reprodutibilidade.
        
    Returns:
        ClusteringResult com as métricas e o melhor modelo identificado.
    """
    if k_min < 2:
        raise ValueError("O cálculo de silhueta requer pelo menos k=2 clusters.")
    if len(X) <= k_max:
        raise ValueError(f"Número de amostras ({len(X)}) deve ser maior que k_max ({k_max}).")

    k_values = list(range(k_min, k_max + 1))
    inertias = []
    sil_scores = []
    models = {}
    labels_dict = {}

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=N_INIT,
            max_iter=MAX_ITER,
            random_state=random_state,
        )
        labels = model.fit_predict(X)
        score = float(silhouette_score(X, labels, metric="euclidean"))

        inertias.append(float(model.inertia_))
        sil_scores.append(score)
        models[k] = model
        labels_dict[k] = labels

    # Determinar k ótimo baseado no maior score de silhueta
    best_idx = int(np.argmax(sil_scores))
    optimal_k = k_values[best_idx]
    best_model = models[optimal_k]
    best_labels = labels_dict[optimal_k]
    best_score = sil_scores[best_idx]

    # Calcular valores individuais de silhueta para cada ponto
    samples_sil = silhouette_samples(X, best_labels, metric="euclidean")

    return ClusteringResult(
        k_values=k_values,
        inertias=inertias,
        silhouette_scores=sil_scores,
        optimal_k=optimal_k,
        best_model=best_model,
        best_labels=best_labels,
        best_silhouette_score=best_score,
        silhouette_samples=samples_sil,
    )


def profile_clusters_2d(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: List[str],
) -> pd.DataFrame:
    """Gera resumo estatístico e caracterização semântica dos clusters 2D.
    
    Args:
        df: DataFrame original com colunas de clientes.
        labels: Rótulos atribuídos pelo K-Means.
        features: Lista com os nomes dos 2 atributos.
        
    Returns:
        DataFrame com médias, desvios e perfil de negócio por cluster.
    """
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    col_income = features[0]
    col_score = features[1]

    grouped = df_clustered.groupby("Cluster")
    summary = grouped.agg(
        Total_Clientes=("CustomerID", "count"),
        Renda_Media=(col_income, "mean"),
        Renda_Std=(col_income, "std"),
        Score_Medio=(col_score, "mean"),
        Score_Std=(col_score, "std"),
        Idade_Media=("Age", "mean"),
    ).reset_index()

    total_pts = len(df_clustered)
    summary["Percentual (%)"] = (summary["Total_Clientes"] / total_pts * 100).round(1)

    # Identificação do perfil de negócio baseado nos centroides
    perfis = []
    for _, row in summary.iterrows():
        inc = row["Renda_Media"]
        sc = row["Score_Medio"]
        if inc < 50 and sc < 50:
            perfil = "Econômicos / Cautelosos de Baixa Renda (Baixa Renda, Baixo Gasto)"
        elif inc < 50 and sc >= 50:
            perfil = "Gastadores Impulsivos (Baixa Renda, Alto Gasto)"
        elif 40 <= inc <= 75 and 35 <= sc <= 65:
            perfil = "Consumidores Padrão / Médios (Renda Média, Gasto Médio)"
        elif inc >= 65 and sc < 45:
            perfil = "Poupadores Prudente (Alta Renda, Baixo Gasto)"
        elif inc >= 65 and sc >= 65:
            perfil = "Clientes Alvo / VIP (Alta Renda, Alto Gasto)"
        else:
            perfil = "Cluster Intermediário"
        perfis.append(perfil)

    summary["Perfil_Sugerido"] = perfis
    return summary


def profile_clusters_multi(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: List[str],
) -> pd.DataFrame:
    """Gera resumo estatístico para agrupamento multidimensional (Idade, Renda, Score).
    
    Args:
        df: DataFrame original.
        labels: Rótulos de cluster.
        features: Lista de features numéricas analisadas.
        
    Returns:
        DataFrame com médias e desvios agrupados por cluster.
    """
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    agg_dict: Dict[str, Any] = {"Total_Clientes": ("CustomerID", "count")}
    for feat in features:
        feat_clean = feat.split(" ")[0].replace("(", "").replace(")", "")
        agg_dict[f"{feat_clean}_Media"] = (feat, "mean")
        agg_dict[f"{feat_clean}_Std"] = (feat, "std")

    summary = df_clustered.groupby("Cluster").agg(**agg_dict).reset_index()
    summary["Percentual (%)"] = (summary["Total_Clientes"] / len(df_clustered) * 100).round(1)
    return summary
