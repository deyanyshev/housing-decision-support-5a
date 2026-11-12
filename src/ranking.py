"""Ранжирование TOPSIS и проверка доминирования по Парето."""
import numpy as np

# Цена и время до метро минимизируются, оценка района максимизируется.
BENEFIT = np.array([False, True, False])


def topsis(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=float), where=norms > 0)
    weighted = normalized * weights
    low, high = weighted.min(axis=0), weighted.max(axis=0)
    best = np.where(BENEFIT, high, low)
    worst = np.where(BENEFIT, low, high)
    d_best = np.linalg.norm(weighted - best, axis=1)
    d_worst = np.linalg.norm(weighted - worst, axis=1)
    distances = d_best + d_worst
    return np.divide(d_worst, distances, out=np.full(len(matrix), 0.5), where=distances > 0)


def pareto_front(costs: np.ndarray) -> np.ndarray:
    """Оба показателя минимизируются. Полное равенство не является доминированием."""
    dominates = ((costs[:, None] <= costs[None, :]).all(axis=2)
                 & (costs[:, None] < costs[None, :]).any(axis=2))
    return ~dominates.any(axis=0)


def rank_order(scores: np.ndarray, object_ids: list[str]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], object_ids[index]))
