"""Módulo de carregamento e pré-processamento dos dados."""
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import DATA_FILE, FEATURES_2D, FEATURES_MULTI


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Carrega o dataset Mall Customers e valida integridade.
    
    Args:
        file_path: Caminho opcional do arquivo CSV. Se None, usa DATA_FILE do config.
        
    Returns:
        DataFrame com os dados validados.
    """
    path = file_path or DATA_FILE
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {path}")

    df = pd.read_csv(path)

    expected_cols = {"CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {missing_cols}")

    # Checar valores ausentes
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(f"Valores nulos detectados no dataset:\n{null_counts[null_counts > 0]}")

    return df


def get_2d_data(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Extrai os atributos 2D clássicos (Renda Anual vs Pontuação de Gastos).
    
    Args:
        df: DataFrame original.
        
    Returns:
        Tupla com (matriz numpy X, lista de nomes das features).
    """
    X = df[FEATURES_2D].to_numpy(dtype=float)
    return X, FEATURES_2D


def get_multidimensional_data(
    df: pd.DataFrame, scale: bool = True
) -> Tuple[np.ndarray, List[str], Optional[StandardScaler]]:
    """Extrai os atributos multidimensionais (Idade, Renda Anual, Pontuação de Gastos).
    
    Args:
        df: DataFrame original.
        scale: Se True, aplica StandardScaler para equalizar variâncias.
        
    Returns:
        Tupla com (matriz numpy X, lista de nomes das features, scaler ajustado ou None).
    """
    X_raw = df[FEATURES_MULTI].to_numpy(dtype=float)
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        return X_scaled, FEATURES_MULTI, scaler
    return X_raw, FEATURES_MULTI, None
