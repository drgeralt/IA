"""Implementação do Algoritmo Genético para o Problema do Caixeiro Viajante (TSP)."""

from dataclasses import dataclass, field
import random
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class GenerationMetric:
    """Métricas registradas para cada geração do algoritmo."""

    generation: int
    best_chromosome: List[int]
    best_distance: float
    best_fitness: float
    avg_distance: float
    worst_distance: float


@dataclass
class GAResult:
    """Resultado final da execução do Algoritmo Genético."""

    best_chromosome: List[int]
    best_distance: float
    best_fitness: float
    initial_chromosome: List[int]
    initial_distance: float
    initial_fitness: float
    generations_run: int
    stopped_reason: str
    history: List[GenerationMetric] = field(default_factory=list)


class GeneticAlgorithm:
    """Algoritmo Genético para o TSP com permutação, torneio, OX, Swap Mutation e Elitismo."""

    def __init__(
        self,
        num_cities: int,
        distance_matrix: np.ndarray,
        population_size: int = 100,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.03,
        tournament_k: int = 3,
        elitism_size: int = 2,
        max_generations: int = 500,
        stagnation_limit: int = 50,
        seed: Optional[int] = None,
    ):
        """Inicializa os parâmetros do Algoritmo Genético.

        Args:
            num_cities: Número total de cidades (N).
            distance_matrix: Matriz 2D de distâncias pré-calculadas (N x N).
            population_size: Tamanho da população (padrão: 100).
            crossover_rate: Probabilidade de cruzamento Order Crossover (0.80 a 0.90).
            mutation_rate: Probabilidade de Swap Mutation por indivíduo (0.01 a 0.05).
            tournament_k: Quantidade de competidores na seleção por torneio (padrão: 3).
            elitism_size: Quantidade de indivíduos de elite preservados intactos (padrão: 2).
            max_generations: Limite máximo de gerações (padrão: 500).
            stagnation_limit: Gerações consecutivas sem melhoria antes de parar (padrão: 50).
            seed: Semente pseudoaleatória opcional para reprodutibilidade.
        """
        if elitism_size > population_size:
            raise ValueError("O tamanho do elitismo não pode ser maior que o tamanho da população.")
        if num_cities < 3:
            raise ValueError("O TSP requer pelo menos 3 cidades para formar um circuito.")

        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.elitism_size = elitism_size
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def calculate_total_distance(self, chromosome: List[int]) -> float:
        """Calcula a distância total do percurso somando os trechos e o retorno à origem.

        Args:
            chromosome: Lista com a ordem de visitação das cidades.

        Returns:
            Distância total percorrida no circuito fechado.
        """
        n = len(chromosome)
        total = 0.0
        for i in range(n):
            from_city = chromosome[i]
            to_city = chromosome[(i + 1) % n]
            total += float(self.distance_matrix[from_city, to_city])
        return total

    def calculate_fitness(self, distance: float) -> float:
        """Calcula a aptidão (fitness) como o inverso da distância total.

        Args:
            distance: Distância total da rota.

        Returns:
            Fitness = 1.0 / distance.
        """
        if distance <= 0.0:
            return float("inf")
        return 1.0 / distance

    def initialize_population(self) -> List[List[int]]:
        """Cria rotas totalmente aleatórias embaralhando a lista de IDs de cidades [0, ..., N-1].

        Returns:
            Lista de cromossomos (permutações).
        """
        base_route = list(range(self.num_cities))
        population: List[List[int]] = []
        for _ in range(self.population_size):
            individual = base_route.copy()
            random.shuffle(individual)
            population.append(individual)
        return population

    def tournament_selection(
        self, population: List[List[int]], fitnesses: List[float]
    ) -> List[int]:
        """Realiza seleção por torneio sorteando k indivíduos e escolhendo o de maior fitness.

        Args:
            population: População de cromossomos.
            fitnesses: Lista de valores de fitness correspondentes a cada indivíduo.

        Returns:
            Cromossomo vencedor do torneio.
        """
        indices = random.sample(range(len(population)), k=self.tournament_k)
        best_idx = max(indices, key=lambda idx: fitnesses[idx])
        return population[best_idx].copy()

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Aplica o operador Order Crossover (OX) para permutações.

        1. Seleciona um trecho contínuo aleatório do Pai 1 e copia para a mesma posição no Filho.
        2. Preencha as posições vazias restantes do Filho com as cidades do Pai 2, na ordem
           em que aparecem, pulando as cidades que já estão no Filho.

        Args:
            parent1: Primeiro cromossomo progenitor.
            parent2: Segundo cromossomo progenitor.

        Returns:
            Novo cromossomo filho gerado.
        """
        size = self.num_cities
        child: List[Optional[int]] = [None] * size

        # Sorteia dois pontos de corte aleatórios
        idx1, idx2 = sorted(random.sample(range(size), 2))

        # Copia o trecho contínuo [idx1, idx2] do Pai 1
        child[idx1 : idx2 + 1] = parent1[idx1 : idx2 + 1]
        copied_set = set(parent1[idx1 : idx2 + 1])

        # Preenche os espaços restantes com cidades do Pai 2 na ordem em que aparecem
        p2_candidates = [city for city in parent2 if city not in copied_set]
        p2_iter = iter(p2_candidates)

        for i in range(size):
            if child[i] is None:
                child[i] = next(p2_iter)

        return [int(c) for c in child]  # type: ignore

    def swap_mutation(self, chromosome: List[int]) -> List[int]:
        """Aplica Swap Mutation: sorteia dois índices aleatórios e troca as duas cidades de posição.

        Args:
            chromosome: Cromossomo candidato à mutação.

        Returns:
            Cromossomo mutado (ou cópia se mutation_rate não for atingida).
        """
        mutated = chromosome.copy()
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.num_cities), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def evolve(self) -> GAResult:
        """Executa o ciclo evolutivo completo do Algoritmo Genético.

        Returns:
            GAResult contendo histórico de métricas e melhor solução encontrada.
        """
        # 1. População inicial
        population = self.initialize_population()

        history: List[GenerationMetric] = []
        best_overall_distance = float("inf")
        best_overall_chromosome: List[int] = []
        best_overall_fitness = 0.0
        stagnant_count = 0
        stopped_reason = f"Atingiu o número máximo de {self.max_generations} gerações."

        # Avaliação inicial
        initial_distances = [self.calculate_total_distance(ind) for ind in population]
        initial_fitnesses = [self.calculate_fitness(d) for d in initial_distances]
        initial_best_idx = int(np.argmax(initial_fitnesses))
        initial_best_chromosome = population[initial_best_idx].copy()
        initial_best_distance = initial_distances[initial_best_idx]
        initial_best_fitness = initial_fitnesses[initial_best_idx]

        for gen in range(self.max_generations):
            # Avaliar população atual
            distances = [self.calculate_total_distance(ind) for ind in population]
            fitnesses = [self.calculate_fitness(d) for d in distances]

            # Ordenar por fitness decrescente
            sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            population = [population[i] for i in sorted_indices]
            distances = [distances[i] for i in sorted_indices]
            fitnesses = [fitnesses[i] for i in sorted_indices]

            gen_best_chromosome = population[0].copy()
            gen_best_distance = distances[0]
            gen_best_fitness = fitnesses[0]
            avg_dist = float(np.mean(distances))
            worst_dist = distances[-1]

            history.append(
                GenerationMetric(
                    generation=gen,
                    best_chromosome=gen_best_chromosome,
                    best_distance=gen_best_distance,
                    best_fitness=gen_best_fitness,
                    avg_distance=avg_dist,
                    worst_distance=worst_dist,
                )
            )

            # Verificar estagnação (melhoria estrita acima de uma tolerância numérica)
            if gen_best_distance < (best_overall_distance - 1e-6):
                best_overall_distance = gen_best_distance
                best_overall_chromosome = gen_best_chromosome.copy()
                best_overall_fitness = gen_best_fitness
                stagnant_count = 0
            else:
                stagnant_count += 1

            if stagnant_count >= self.stagnation_limit:
                stopped_reason = (
                    f"Estagnação atingida ({self.stagnation_limit} gerações consecutivas sem melhoria) "
                    f"na geração {gen}."
                )
                break

            # Se for a última geração, não precisa construir a próxima
            if gen == self.max_generations - 1:
                break

            # Construir nova geração
            new_population: List[List[int]] = []

            # Elitismo: Copia os elitism_size melhores indivíduos diretamente
            for e in range(self.elitism_size):
                new_population.append(population[e].copy())

            # Preencher os espaços restantes (ex: 98 indivíduos)
            remaining_slots = self.population_size - self.elitism_size
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                # Order Crossover
                if random.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Swap Mutation
                child = self.swap_mutation(child)

                new_population.append(child)

            population = new_population

        return GAResult(
            best_chromosome=best_overall_chromosome,
            best_distance=best_overall_distance,
            best_fitness=best_overall_fitness,
            initial_chromosome=initial_best_chromosome,
            initial_distance=initial_best_distance,
            initial_fitness=initial_best_fitness,
            generations_run=len(history),
            stopped_reason=stopped_reason,
            history=history,
        )
