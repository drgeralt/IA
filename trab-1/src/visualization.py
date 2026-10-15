"""Módulo de visualização com matplotlib para o Algoritmo Genético do TSP."""

from typing import List, Optional
import os
import matplotlib
# Configura backend headless se não houver display X11/Wayland ativo
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np

from src.city import City
from src.genetic_algorithm import GenerationMetric


def _extract_route_coordinates(cities: List[City], route: List[int]):
    """Extrai coordenadas X e Y ordenadas segundo o cromossomo da rota, fechando o ciclo."""
    city_map = {c.id: c for c in cities}
    x_coords = [city_map[city_id].x for city_id in route]
    y_coords = [city_map[city_id].y for city_id in route]
    # Retorna ao ponto de partida
    x_coords.append(city_map[route[0]].x)
    y_coords.append(city_map[route[0]].y)
    return x_coords, y_coords


def plot_comparison_initial_vs_final(
    cities: List[City],
    initial_route: List[int],
    initial_distance: float,
    final_route: List[int],
    final_distance: float,
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Gera um gráfico lado a lado comparando a 1ª geração (caótica) com a última geração (otimizada).

    Args:
        cities: Lista de objetos City com coordenadas.
        initial_route: Melhor cromossomo da geração inicial (Gen 0).
        initial_distance: Distância total da rota inicial.
        final_route: Melhor cromossomo da geração final.
        final_distance: Distância total da rota final.
        output_path: Caminho opcional para salvar o arquivo de imagem (PNG/JPG).
        show: Se True, invoca plt.show().

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        "Problema do Caixeiro Viajante (TSP) - Comparação Algoritmo Genético",
        fontsize=16,
        fontweight="bold",
    )

    all_x = [c.x for c in cities]
    all_y = [c.y for c in cities]

    # --- 1. Rota Inicial (Caótica) ---
    init_x, init_y = _extract_route_coordinates(cities, initial_route)
    ax1.plot(init_x, init_y, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.85, label="Caminho")
    ax1.scatter(all_x, all_y, color="#2c3e50", s=60, zorder=5, label="Cidades")

    # Rótulo numérico de cada cidade
    for c in cities:
        ax1.annotate(
            str(c.id),
            (c.x, c.y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax1.set_title(
        f"Geração Inicial (Gen 0 - Rota Caótica)\nDistância Total: {initial_distance:.2f}",
        fontsize=12,
        fontweight="bold",
        color="#c0392b",
    )
    ax1.set_xlabel("Coordenada X")
    ax1.set_ylabel("Coordenada Y")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # --- 2. Rota Final (Otimizada) ---
    final_x, final_y = _extract_route_coordinates(cities, final_route)
    improvement = ((initial_distance - final_distance) / initial_distance) * 100.0

    ax2.plot(final_x, final_y, color="#27ae60", linestyle="-", linewidth=2.0, alpha=0.9, label="Caminho Otimizado")
    ax2.scatter(all_x, all_y, color="#2c3e50", s=60, zorder=5, label="Cidades")

    for c in cities:
        ax2.annotate(
            str(c.id),
            (c.x, c.y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    # Destacar a cidade de início da rota
    start_city = next(c for c in cities if c.id == final_route[0])
    ax2.scatter([start_city.x], [start_city.y], color="#f39c12", s=130, zorder=6, label="Início / Fim")

    ax2.set_title(
        f"Geração Final (Rota Otimizada)\nDistância Total: {final_distance:.2f} (Melhoria: -{improvement:.1f}%)",
        fontsize=12,
        fontweight="bold",
        color="#27ae60",
    )
    ax2.set_xlabel("Coordenada X")
    ax2.set_ylabel("Coordenada Y")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_convergence(
    history: List[GenerationMetric],
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plota a curva de evolução das distâncias (Melhor vs Média) ao longo das gerações.

    Args:
        history: Lista com as métricas gravadas a cada geração.
        output_path: Caminho opcional para salvar a imagem do gráfico.
        show: Se True, invoca plt.show().

    Returns:
        Objeto Figure do matplotlib.
    """
    generations = [m.generation for m in history]
    best_distances = [m.best_distance for m in history]
    avg_distances = [m.avg_distance for m in history]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(generations, avg_distances, color="#95a5a6", linestyle="--", linewidth=1.5, label="Distância Média da População")
    ax.plot(generations, best_distances, color="#2980b9", linewidth=2.2, label="Melhor Distância (Elite)")

    # Ponto inicial e ponto final
    ax.scatter([generations[0]], [best_distances[0]], color="#e74c3c", s=70, zorder=5, label=f"Início: {best_distances[0]:.2f}")
    ax.scatter([generations[-1]], [best_distances[-1]], color="#27ae60", s=80, zorder=5, label=f"Ótimo Encontrado: {best_distances[-1]:.2f}")

    ax.set_title("Curva de Convergência do Algoritmo Genético", fontsize=14, fontweight="bold")
    ax.set_xlabel("Geração", fontsize=11)
    ax.set_ylabel("Distância Total", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def create_route_animation(
    cities: List[City],
    history: List[GenerationMetric],
    output_path: str,
    fps: int = 12,
    max_frames: int = 150,
) -> None:
    """Gera uma animação da melhor rota evoluindo a cada geração e salva como GIF ou MP4.

    Args:
        cities: Lista de cidades com coordenadas.
        history: Histórico de métricas das gerações.
        output_path: Arquivo de saída (.gif ou .mp4).
        fps: Quadros por segundo.
        max_frames: Limite de frames para manter o arquivo leve e dinâmico.
    """
    total_gens = len(history)

    # Amostragem inteligente de frames:
    # 1. Sempre incluir frames onde houve melhora na distância.
    # 2. Amostrar uniformemente as gerações intermediárias.
    # 3. Sempre incluir a primeira e a última geração.
    frames_indices = set()
    best_seen = float("inf")
    for idx, m in enumerate(history):
        if m.best_distance < best_seen - 1e-4:
            frames_indices.add(idx)
            best_seen = m.best_distance

    # Amostragem uniforme
    step = max(1, total_gens // max_frames)
    for idx in range(0, total_gens, step):
        frames_indices.add(idx)

    frames_indices.add(0)
    frames_indices.add(total_gens - 1)
    sorted_frames = sorted(list(frames_indices))

    all_x = [c.x for c in cities]
    all_y = [c.y for c in cities]

    fig, ax = plt.subplots(figsize=(8, 7))

    line, = ax.plot([], [], color="#2980b9", linewidth=2.0, marker="o", markersize=6, markerfacecolor="#e74c3c")
    scatter = ax.scatter(all_x, all_y, color="#2c3e50", s=60, zorder=5)

    # Rótulos das cidades
    for c in cities:
        ax.annotate(
            str(c.id),
            (c.x, c.y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    # Margens do gráfico
    x_margin = (max(all_x) - min(all_x)) * 0.08
    y_margin = (max(all_y) - min(all_y)) * 0.08
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")

    title_text = ax.set_title("", fontsize=12, fontweight="bold")

    def init():
        line.set_data([], [])
        title_text.set_text("")
        return line, title_text

    def update(frame_idx):
        metric = history[frame_idx]
        rx, ry = _extract_route_coordinates(cities, metric.best_chromosome)
        line.set_data(rx, ry)
        title_text.set_text(
            f"Evolução do TSP - Geração {metric.generation + 1}/{total_gens}\n"
            f"Melhor Distância: {metric.best_distance:.2f} | Fitness: {metric.best_fitness:.5f}"
        )
        return line, title_text

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=sorted_frames,
        interval=1000 // fps,
        blit=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if output_path.endswith(".mp4"):
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist="Antigravity"), bitrate=1800)
            anim.save(output_path, writer=writer)
        except Exception:
            # Fallback para GIF caso ocorra algum erro com mp4
            gif_path = output_path.replace(".mp4", ".gif")
            anim.save(gif_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(output_path, writer=PillowWriter(fps=fps))

    plt.close(fig)
