"""Метод Саати. Приоритеты из главного собственного вектора."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig

CR_LIMIT = 0.10
RANDOM_INDEX = {3: 0.58}
CRITERIA = ("Цена", "Район", "Доступность метро")


@dataclass(frozen=True)
class AHP:
    matrix: np.ndarray
    weights: np.ndarray
    lambda_max: float
    ci: float
    cr: float


def calculate_ahp(judgments: dict) -> AHP:
    price_district = judgments["price_over_district"]
    price_transport = judgments["price_over_transport"]
    transport_district = judgments["transport_over_district"]
    comparisons = np.array([price_district, price_transport, transport_district], dtype=float)
    if not np.all(np.isfinite(comparisons)) or np.any(comparisons < 1 / 9) or np.any(comparisons > 9):
        raise ValueError("Попарные оценки AHP должны быть от 1/9 до 9.")
    price_district, price_transport, transport_district = comparisons
    matrix = np.array([
        [1, price_district, price_transport],
        [1 / price_district, 1, 1 / transport_district],
        [1 / price_transport, transport_district, 1],
    ])
    eigenvalues, eigenvectors = eig(matrix)
    principal = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues[principal].real)
    vector = eigenvectors[:, principal].real
    weights = vector / vector.sum()
    n = len(CRITERIA)
    ci = max(0.0, (lambda_max - n) / (n - 1))
    cr = ci / RANDOM_INDEX[n]
    if cr > CR_LIMIT:
        raise ValueError(f"CR = {cr:.4f} > {CR_LIMIT:.2f}. Пересмотрите оценки AHP в config.json.")
    return AHP(matrix, weights, lambda_max, ci, cr)
