"""Representação de Cidades e cálculo da Matriz de Distâncias Euclidiana."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class City:
    """Representa uma cidade no plano bidimensional com identificador e coordenadas (X, Y)."""

    id: int
    x: float
    y: float

    def distance_to(self, other: "City") -> float:
        """Calcula a distância euclidiana direta até outra cidade."""
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def __repr__(self) -> str:
        return f"City(id={self.id}, x={self.x:.2f}, y={self.y:.2f})"


def generate_random_cities(
    num_cities: int = 25,
    x_max: float = 100.0,
    y_max: float = 100.0,
    seed: Optional[int] = None,
) -> List[City]:
    """Gera uma lista de cidades com coordenadas pseudoaleatórias dentro dos limites [0, x_max] e [0, y_max].

    Args:
        num_cities: Quantidade de cidades a serem geradas.
        x_max: Limite superior da coordenada X.
        y_max: Limite superior da coordenada Y.
        seed: Semente pseudoaleatória opcional para reprodutibilidade.

    Returns:
        Lista de instâncias de City indexadas de 0 a num_cities - 1.
    """
    if seed is not None:
        np.random.seed(seed)

    x_coords = np.random.uniform(0.0, x_max, size=num_cities)
    y_coords = np.random.uniform(0.0, y_max, size=num_cities)

    return [City(id=i, x=float(x_coords[i]), y=float(y_coords[i])) for i in range(num_cities)]


def compute_distance_matrix(cities: List[City]) -> np.ndarray:
    """Pré-calcula a matriz de distâncias euclidianas 2D entre todas as combinações de cidades.

    Args:
        cities: Lista de cidades ordenadas por seu ID.

    Returns:
        Array 2D (num_cities x num_cities) contendo as distâncias euclidianas.
    """
    num_cities = len(cities)
    coords = np.array([[c.x, c.y] for c in cities], dtype=np.float64)

    # Cálculo vetorizado: diff de dimensões (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    return dist_matrix
