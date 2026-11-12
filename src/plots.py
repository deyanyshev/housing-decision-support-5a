"""Статичные PNG для результатов и презентации."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ahp import CRITERIA

NAVY = "#142C44"
TEAL = "#087A74"
ORANGE = "#B45325"
PALETTE = (TEAL, "#336AA0", ORANGE, "#745398", "#657A24", "#8D5269")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": NAVY, "text.color": NAVY,
    "axes.edgecolor": "#B8C6CE", "xtick.color": NAVY, "ytick.color": NAVY,
    "grid.color": "#DEE6EB", "axes.axisbelow": True,
})


def save_png(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, format="png", dpi=170, facecolor="white")
    plt.close(fig)


def weights_plot(weights, output):
    fig, ax = plt.subplots(figsize=(8.2, 4.5), layout="constrained")
    bars = ax.barh(CRITERIA, weights * 100, color=[TEAL, "#7DA9AE", "#336AA0"], height=0.55)
    ax.invert_yaxis()
    ax.set(xlim=(0, max(65, max(weights) * 100 + 10)), xlabel="Вес критерия, %")
    ax.grid(axis="x")
    ax.bar_label(bars, labels=[f"{value * 100:.1f}%".replace(".", ",") for value in weights], padding=6)
    save_png(fig, output / "ahp_weights.png")


def ranking_plot(rows, output):
    best = rows[:10]
    fig, ax = plt.subplots(figsize=(8.2, 5), layout="constrained")
    names = [f"{row['object_id']} · {row['district']}" for row in best]
    bars = ax.barh(names, [row["topsis_score"] for row in best], color=[TEAL] + ["#88A9BB"] * (len(best) - 1))
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.08), xticks=np.linspace(0, 1, 6), xlabel="Близость к идеальному решению")
    ax.grid(axis="x")
    ax.bar_label(bars, labels=[f"{row['topsis_score']:.4f}".replace(".", ",") for row in best], padding=4)
    save_png(fig, output / "topsis_ranking.png")


def pareto_plot(items, front, output):
    prices = np.array([item.price_rub / 1e6 for item in items])
    walks = np.array([item.metro_walk_min for item in items])
    fig, ax = plt.subplots(figsize=(8.2, 4.7), layout="constrained")
    ax.scatter(prices[~front], walks[~front], color="#AABBC7", s=34, label="Остальные объекты")
    ax.scatter(prices[front], walks[front], color=TEAL, s=64, label="Фронт Парето", zorder=3)
    for index in np.flatnonzero(front):
        ax.annotate(items[index].object_id, (prices[index], walks[index]), xytext=(6, 5),
                    textcoords="offset points", color=TEAL, fontsize=10)
    ax.set(xlabel="Цена, млн ₽", ylabel="Пешком до метро, мин")
    ax.grid()
    ax.legend(loc="upper right", frameon=False)
    save_png(fig, output / "pareto_front.png")


def mortgage_plot(mortgage, output):
    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    for label, color, schedule in (("Аннуитет", TEAL, mortgage.annuity),
                                  ("Дифференцированный", ORANGE, mortgage.differentiated)):
        ax.plot([p.month for p in schedule], [p.payment / 1000 for p in schedule],
                label=label, color=color, linewidth=2.5)
    ax.set(xlabel="Месяц кредита", ylabel="Платёж, тыс. ₽", ylim=(0, None))
    ax.grid()
    ax.legend(frameon=False)
    save_png(fig, output / "mortgage_payments.png")


def sensitivity_plot(scenarios, object_ids, shown_indices, output):
    fig, ax = plt.subplots(figsize=(8.2, 4.7), layout="constrained")
    for series, index in enumerate(shown_indices):
        ranks = [scenario["positions"][index] for scenario in scenarios]
        ax.plot(range(len(scenarios)), ranks, "o-", label=object_ids[index],
                color=PALETTE[series % len(PALETTE)], linewidth=2)
    ax.set_xticks(range(len(scenarios)), [f"{s['price_weight']:.1%}".replace(".", ",") for s in scenarios])
    ax.set(xlabel="Вес цены после нормировки", ylabel="Место в рейтинге")
    ax.invert_yaxis()
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False)
    save_png(fig, output / "sensitivity.png")
