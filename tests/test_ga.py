"""Testes unitários e de integração para o Algoritmo Genético do TSP."""

import pytest
import numpy as np

from src.city import City, generate_random_cities, compute_distance_matrix
from src.genetic_algorithm import GeneticAlgorithm


def test_city_creation_and_distance():
    c1 = City(id=0, x=0.0, y=0.0)
    c2 = City(id=1, x=3.0, y=4.0)
    assert c1.distance_to(c2) == pytest.approx(5.0)


def test_generate_random_cities_reproducibility():
    cities_a = generate_random_cities(num_cities=10, seed=123)
    cities_b = generate_random_cities(num_cities=10, seed=123)
    assert len(cities_a) == 10
    for ca, cb in zip(cities_a, cities_b):
        assert ca.id == cb.id
        assert ca.x == pytest.approx(cb.x)
        assert ca.y == pytest.approx(cb.y)


def test_compute_distance_matrix():
    # Quadrado unitário com 4 vértices
    cities = [
        City(0, 0.0, 0.0),
        City(1, 0.0, 1.0),
        City(2, 1.0, 1.0),
        City(3, 1.0, 0.0),
    ]
    matrix = compute_distance_matrix(cities)
    assert matrix.shape == (4, 4)

    # Diagonal nula
    for i in range(4):
        assert matrix[i, i] == 0.0

    # Simetria
    for i in range(4):
        for j in range(4):
            assert matrix[i, j] == pytest.approx(matrix[j, i])

    # Lados do quadrado
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[1, 2] == pytest.approx(1.0)
    assert matrix[2, 3] == pytest.approx(1.0)
    assert matrix[3, 0] == pytest.approx(1.0)
    # Diagonal do quadrado
    assert matrix[0, 2] == pytest.approx(np.sqrt(2.0))


def test_distance_and_fitness_on_known_square():
    cities = [
        City(0, 0.0, 0.0),
        City(1, 0.0, 1.0),
        City(2, 1.0, 1.0),
        City(3, 1.0, 0.0),
    ]
    dist_matrix = compute_distance_matrix(cities)
    ga = GeneticAlgorithm(num_cities=4, distance_matrix=dist_matrix, seed=42)

    # Rota ao longo do perímetro: 0 -> 1 -> 2 -> 3 -> 0 = 4.0
    perimeter_route = [0, 1, 2, 3]
    dist = ga.calculate_total_distance(perimeter_route)
    assert dist == pytest.approx(4.0)

    fitness = ga.calculate_fitness(dist)
    assert fitness == pytest.approx(1.0 / 4.0)


def test_order_crossover_validity():
    dist_matrix = np.zeros((10, 10))
    ga = GeneticAlgorithm(num_cities=10, distance_matrix=dist_matrix, seed=42)

    # Executa múltiplos crossovers com permutações aleatórias
    for _ in range(50):
        p1 = list(np.random.permutation(10))
        p2 = list(np.random.permutation(10))
        child = ga.order_crossover(p1, p2)

        # 1. Comprimento correto
        assert len(child) == 10
        # 2. Todos os números de 0 a 9 estão presentes exatamente uma vez
        assert sorted(child) == list(range(10))
        # 3. Nenhuma duplicata
        assert len(set(child)) == 10


def test_swap_mutation_validity():
    dist_matrix = np.zeros((8, 8))
    ga = GeneticAlgorithm(num_cities=8, distance_matrix=dist_matrix, mutation_rate=1.0, seed=42)

    orig = [0, 1, 2, 3, 4, 5, 6, 7]
    mutated = ga.swap_mutation(orig)

    # Preserva o tamanho e os elementos
    assert len(mutated) == 8
    assert sorted(mutated) == orig
    # Com taxa 1.0, deve ser diferente do original (dois elementos trocados)
    diff_count = sum(1 for a, b in zip(orig, mutated) if a != b)
    assert diff_count == 2


def test_elitism_monotonicity():
    # O melhor fitness nunca deve piorar de uma geração para a seguinte devido ao elitismo
    cities = generate_random_cities(num_cities=15, seed=10)
    matrix = compute_distance_matrix(cities)
    ga = GeneticAlgorithm(
        num_cities=15,
        distance_matrix=matrix,
        population_size=50,
        crossover_rate=0.85,
        mutation_rate=0.05,
        elitism_size=2,
        max_generations=40,
        stagnation_limit=50,
        seed=10,
    )

    result = ga.evolve()
    best_dists = [m.best_distance for m in result.history]

    for i in range(len(best_dists) - 1):
        assert best_dists[i + 1] <= best_dists[i] + 1e-9


def test_stagnation_criterion():
    # Configura estagnação bem curta (5 gerações) e verifica se o algoritmo para antes de max_generations
    cities = generate_random_cities(num_cities=10, seed=99)
    matrix = compute_distance_matrix(cities)
    ga = GeneticAlgorithm(
        num_cities=10,
        distance_matrix=matrix,
        population_size=40,
        max_generations=200,
        stagnation_limit=5,
        seed=99,
    )

    result = ga.evolve()
    assert result.generations_run < 200
    assert "Estagnação atingida" in result.stopped_reason
