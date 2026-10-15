#!/usr/bin/env python3
"""Ponto de entrada principal para o trabalho de K-Means e Análise de Silhueta."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adicionar pasta base ao sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATA_FILE,
    OUTPUTS_DIR,
    K_MIN,
    K_MAX,
    RANDOM_STATE,
    FEATURES_2D,
    FEATURES_MULTI,
)
from src.data_loader import (
    load_raw_data,
    get_2d_data,
    get_multidimensional_data,
)
from src.kmeans_pipeline import (
    evaluate_kmeans_range,
    profile_clusters_2d,
    profile_clusters_multi,
)
from src.visualization import (
    plot_elbow_and_silhouette,
    plot_silhouette_analysis,
    plot_clusters_2d,
    plot_clusters_pca,
)


def print_banner():
    banner = """
================================================================================
    IA - CLUSTERING COM K-MEANS & DETERMINACAO DE K VIA SILHUETA
    Base de Dados: Mall Customer Segmentation Dataset
    Metricas: Metodo do Cotovelo (WCSS) & Coeficiente de Silhueta
================================================================================
"""
    print(banner)


def run_pipeline():
    print_banner()

    # 1. Carregamento dos dados
    print("[1/5] Carregando e validando dataset...")
    df = load_raw_data(DATA_FILE)
    print(f"      Total de registros carregados: {len(df)}")
    print(f"      Atributos disponíveis: {list(df.columns)}")
    print(f"      Valores nulos: {df.isnull().sum().to_dict()}\n")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Pipeline 1: 2D (Renda Anual vs Pontuação de Gastos)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("[2/5] Executando Pipeline 1: Modelo 2D (Renda Anual vs Pontuação de Gastos)")
    print("=" * 80)
    X_2d, feats_2d = get_2d_data(df)
    res_2d = evaluate_kmeans_range(X_2d, k_min=K_MIN, k_max=K_MAX, random_state=RANDOM_STATE)

    # Tabela de métricas 2D
    metrics_2d_df = pd.DataFrame({
        "k": res_2d.k_values,
        "Inercia_WCSS": [round(x, 2) for x in res_2d.inertias],
        "Silhouette_Score": [round(x, 4) for x in res_2d.silhouette_scores],
    })
    metrics_2d_path = OUTPUTS_DIR / "metrics_2d_summary.csv"
    metrics_2d_df.to_csv(metrics_2d_path, index=False)

    print("\n--- Metricas por valor de k (2D) ---")
    print(metrics_2d_df.to_string(index=False))
    print(f"\n>>> k Otimo determinado pelo Coeficiente de Silhueta: k = {res_2d.optimal_k} "
          f"(Score = {res_2d.best_silhouette_score:.4f}) <<<\n")

    # Gerar gráficos 2D
    print("Salvando graficos do pipeline 2D em outputs/...")
    p_elbow_sil_2d = OUTPUTS_DIR / "elbow_silhouette_2d.png"
    plot_elbow_and_silhouette(
        res_2d.k_values,
        res_2d.inertias,
        res_2d.silhouette_scores,
        res_2d.optimal_k,
        "Abordagem 2D (Renda vs Gastos)",
        p_elbow_sil_2d,
    )
    print(f"  -> Salvo: {p_elbow_sil_2d.name}")

    p_sil_samples_2d = OUTPUTS_DIR / "silhouette_samples_2d.png"
    plot_silhouette_analysis(
        X_2d,
        res_2d.best_labels,
        res_2d.optimal_k,
        res_2d.best_silhouette_score,
        res_2d.silhouette_samples,
        p_sil_samples_2d,
    )
    print(f"  -> Salvo: {p_sil_samples_2d.name}")

    p_clusters_2d = OUTPUTS_DIR / "clusters_2d.png"
    plot_clusters_2d(
        X_2d,
        res_2d.best_labels,
        res_2d.best_model.cluster_centers_,
        feats_2d,
        res_2d.optimal_k,
        p_clusters_2d,
    )
    print(f"  -> Salvo: {p_clusters_2d.name}")

    # Profiling dos clusters 2D
    profile_2d = profile_clusters_2d(df, res_2d.best_labels, feats_2d)
    profile_2d_path = OUTPUTS_DIR / "cluster_profiles_2d.csv"
    profile_2d.to_csv(profile_2d_path, index=False)
    print("\n--- Perfil dos Clusters Encontrados (2D) ---")
    display_cols = ["Cluster", "Total_Clientes", "Percentual (%)", "Renda_Media", "Score_Medio", "Perfil_Sugerido"]
    print(profile_2d[display_cols].to_string(index=False))
    print()

    # -------------------------------------------------------------------------
    # 3. Pipeline 2: Multidimensional (Idade, Renda, Gastos) com StandardScaler
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("[3/5] Executando Pipeline 2: Multidimensional (Idade + Renda + Gastos com StandardScaler)")
    print("=" * 80)
    X_multi_scaled, feats_multi, scaler = get_multidimensional_data(df, scale=True)
    res_multi = evaluate_kmeans_range(X_multi_scaled, k_min=K_MIN, k_max=K_MAX, random_state=RANDOM_STATE)

    # Tabela de métricas Multidimensional
    metrics_multi_df = pd.DataFrame({
        "k": res_multi.k_values,
        "Inercia_WCSS": [round(x, 2) for x in res_multi.inertias],
        "Silhouette_Score": [round(x, 4) for x in res_multi.silhouette_scores],
    })
    metrics_multi_path = OUTPUTS_DIR / "metrics_multi_summary.csv"
    metrics_multi_df.to_csv(metrics_multi_path, index=False)

    print("\n--- Metricas por valor de k (Multidimensional Padronizado) ---")
    print(metrics_multi_df.to_string(index=False))
    print(f"\n>>> k Otimo determinado pelo Coeficiente de Silhueta: k = {res_multi.optimal_k} "
          f"(Score = {res_multi.best_silhouette_score:.4f}) <<<\n")

    # Gerar gráficos Multidimensional
    print("Salvando graficos do pipeline Multidimensional em outputs/...")
    p_elbow_sil_multi = OUTPUTS_DIR / "elbow_silhouette_multi.png"
    plot_elbow_and_silhouette(
        res_multi.k_values,
        res_multi.inertias,
        res_multi.silhouette_scores,
        res_multi.optimal_k,
        "Abordagem Multidimensional (Idade + Renda + Gastos)",
        p_elbow_sil_multi,
    )
    print(f"  -> Salvo: {p_elbow_sil_multi.name}")

    p_clusters_pca = OUTPUTS_DIR / "clusters_pca_multi.png"
    plot_clusters_pca(
        X_multi_scaled,
        res_multi.best_labels,
        res_multi.best_model.cluster_centers_,
        res_multi.optimal_k,
        p_clusters_pca,
    )
    print(f"  -> Salvo: {p_clusters_pca.name}")

    profile_multi = profile_clusters_multi(df, res_multi.best_labels, feats_multi)
    profile_multi_path = OUTPUTS_DIR / "cluster_profiles_multi.csv"
    profile_multi.to_csv(profile_multi_path, index=False)
    print("\n--- Perfil dos Clusters Encontrados (Multidimensional) ---")
    print(profile_multi.to_string(index=False))
    print()

    # -------------------------------------------------------------------------
    # 4. Síntese Comparativa
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("[4/5] Síntese Comparativa e Discussão Teórica")
    print("=" * 80)
    print(f"* Cenário 2D (Renda vs Gastos):")
    print(f"  - k ótimo = {res_2d.optimal_k}")
    print(f"  - Silhouette Score Máximo = {res_2d.best_silhouette_score:.4f}")
    print(f"  - Interpretação: Agrupamentos geométricos altamente densos e bem delimitados,")
    print(f"    com correspondência perfeita aos 5 quadrantes econômicos clássicos.")
    print(f"* Cenário Multidimensional (Idade + Renda + Gastos):")
    print(f"  - k ótimo = {res_multi.optimal_k}")
    print(f"  - Silhouette Score Máximo = {res_multi.best_silhouette_score:.4f}")
    print(f"  - Interpretação: A adição da idade dilui a separação euclidiana no hiperespaço,")
    print(f"    refletindo menor coeficiente de silhueta devido à maior dispersão contínua da idade.")

    # -------------------------------------------------------------------------
    # 5. Conclusão
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("[5/5] Finalizado com sucesso! Todos os artefatos foram salvos em trab-2/outputs/:")
    for f in sorted(OUTPUTS_DIR.iterdir()):
        print(f"  - {f.name}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
