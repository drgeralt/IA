#!/usr/bin/env python3
"""Ponto de entrada principal para a execução do Algoritmo Genético no TSP."""

import argparse
import os
import sys
import time

from src.city import generate_random_cities, compute_distance_matrix
from src.genetic_algorithm import GeneticAlgorithm
from src.visualization import (
    plot_comparison_initial_vs_final,
    plot_convergence,
    create_route_animation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Otimização do Caixeiro Viajante (TSP) usando Algoritmo Genético."
    )
    parser.add_argument(
        "--cities", type=int, default=25, help="Número de cidades a visitar (padrão: 25)"
    )
    parser.add_argument(
        "--pop-size", type=int, default=100, help="Tamanho da população (padrão: 100)"
    )
    parser.add_argument(
        "--max-gen", type=int, default=500, help="Número máximo de gerações (padrão: 500)"
    )
    parser.add_argument(
        "--stagnation",
        type=int,
        default=50,
        help="Gerações sem melhoria para parada antecipada (padrão: 50)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.85,
        help="Probabilidade de Order Crossover [0.80 a 0.90] (padrão: 0.85)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.03,
        help="Probabilidade de Swap Mutation [0.01 a 0.05] (padrão: 0.03)",
    )
    parser.add_argument(
        "--tournament-k",
        type=int,
        default=3,
        help="Número de competidores no torneio (padrão: 3)",
    )
    parser.add_argument(
        "--elitism",
        type=int,
        default=2,
        help="Quantidade de indivíduos preservados por elitismo (padrão: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Semente pseudoaleatória para reprodutibilidade (padrão: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Diretório onde gráficos e animações serão salvos (padrão: outputs)",
    )
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Se definido, pula a geração do arquivo de animação (GIF/MP4)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Se definido, exibe as janelas de plotagem interativas",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("\n      ALGORITMO GENÉTICO - PROBLEMA DO CAIXEIRO VIAJANTE (TSP)       \n")
    print(f"Número de cidades:      {args.cities}")
    print(f"Tamanho da população:   {args.pop_size}")
    print(f"Elitismo:               {args.elitism} melhores indivíduos preservados")
    print(f"Seleção:                Torneio (k={args.tournament_k})")
    print(f"Cruzamento:             Order Crossover (OX) (Taxa: {args.crossover_rate * 100:.1f}%)")
    print(f"Mutação:                Swap Mutation (Taxa: {args.mutation_rate * 100:.1f}%)")
    print(f"Critérios de parada:    Máx {args.max_gen} gerações ou {args.stagnation} gerações estagnado")
    print(f"Semente (Seed):         {args.seed}")
    print(f"Diretório de saída:     {args.output_dir}")
    print("-" * 70)

    # 1. Geração de Cidades e Matriz de Distâncias
    cities = generate_random_cities(num_cities=args.cities, seed=args.seed)
    dist_matrix = compute_distance_matrix(cities)

    # 2. Inicialização do Algoritmo Genético
    ga = GeneticAlgorithm(
        num_cities=args.cities,
        distance_matrix=dist_matrix,
        population_size=args.pop_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_k=args.tournament_k,
        elitism_size=args.elitism,
        max_generations=args.max_gen,
        stagnation_limit=args.stagnation,
        seed=args.seed,
    )

    # 3. Execução Evolutiva
    print("\nIniciando evolução...\n")
    print(f"{'Geração':>8} | {'Melhor Distância':>16} | {'Distância Média':>15} | {'Melhor Fitness':>14}")
    print("-" * 62)

    start_time = time.time()
    result = ga.evolve()
    elapsed = time.time() - start_time

    # Exibe logs de amostragem durante as gerações
    log_interval = max(1, result.generations_run // 10)
    for m in result.history:
        if m.generation % log_interval == 0 or m.generation == result.generations_run - 1:
            print(
                f"{m.generation + 1:8d} | {m.best_distance:16.2f} | {m.avg_distance:15.2f} | {m.best_fitness:14.6f}"
            )

    print("-" * 62)
    print("\nEvolução finalizada!")
    print(f"Tempo decorrido:         {elapsed:.2f} segundos")
    print(f"Gerações executadas:     {result.generations_run}")
    print(f"Motivo da parada:        {result.stopped_reason}")
    print(f"Distância Inicial:       {result.initial_distance:.2f}")
    print(f"Melhor Distância Final:  {result.best_distance:.2f}")

    improvement = ((result.initial_distance - result.best_distance) / result.initial_distance) * 100.0
    print(f"Redução na rota:         {improvement:.2f}% de otimização")
    print(f"Melhor rota encontrada:  {result.best_chromosome}")

    # 4. Geração de Visualizações
    os.makedirs(args.output_dir, exist_ok=True)
    comparison_file = os.path.join(args.output_dir, "comparison_initial_vs_final.png")
    convergence_file = os.path.join(args.output_dir, "convergence_curve.png")

    print(f"\nGerando gráficos...")
    plot_comparison_initial_vs_final(
        cities=cities,
        initial_route=result.initial_chromosome,
        initial_distance=result.initial_distance,
        final_route=result.best_chromosome,
        final_distance=result.best_distance,
        output_path=comparison_file,
        show=args.show,
    )
    print(f"-> Salvo: {comparison_file}")

    plot_convergence(
        history=result.history,
        output_path=convergence_file,
        show=args.show,
    )
    print(f"-> Salvo: {convergence_file}")

    if not args.no_anim:
        gif_file = os.path.join(args.output_dir, "tsp_evolution.gif")
        mp4_file = os.path.join(args.output_dir, "tsp_evolution.mp4")
        print(f"\nGerando animação da evolução...")
        create_route_animation(cities, result.history, output_path=gif_file, fps=12)
        print(f"-> Salvo: {gif_file}")
        try:
            create_route_animation(cities, result.history, output_path=mp4_file, fps=12)
            print(f"-> Salvo: {mp4_file}")
        except Exception as e:
            print(f"(Aviso: MP4 não gerado, apenas GIF: {e})")

    print("\n")
    print("Execução concluída com sucesso! Todos os artefatos visuais estão prontos.")
    print("\n")


if __name__ == "__main__":
    main()
