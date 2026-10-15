"""Configurações globais e hiperparâmetros do projeto K-Means."""
from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_FILE = DATA_DIR / "Mall_Customers.csv"

# Parâmetros de reprodutibilidade e modelo
RANDOM_STATE = 42
K_MIN = 2
K_MAX = 10
N_INIT = 10
MAX_ITER = 300

# Seleção de variáveis
FEATURES_2D = ["Annual Income (k$)", "Spending Score (1-100)"]
FEATURES_MULTI = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

# Configurações de plotagem
DPI = 300
PLOT_STYLE = "seaborn-v0_8-whitegrid"
