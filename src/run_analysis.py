"""Воспроизводимый запуск варианта 5А."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .ahp import CRITERIA, CR_LIMIT, RANDOM_INDEX, calculate_ahp
from .data import ROOT, FIELDS, generate_csv, load_config, load_listings, write_csv
from .mortgage import compare_mortgage
from .plots import mortgage_plot, pareto_plot, ranking_plot, sensitivity_plot, weights_plot
from .ranking import pareto_front, rank_order, topsis

MORTGAGE_FIELDS = ("loan_rub", "annuity_first_payment_rub", "differentiated_first_payment_rub",
                   "annuity_overpayment_rub", "differentiated_overpayment_rub",
                   "overpayment_difference_rub")


def _mortgage_fields(mortgage):
    annuity_overpayment = round(mortgage.annuity_overpayment, 2)
    differentiated_overpayment = round(mortgage.differentiated_overpayment, 2)
    return {
        "loan_rub": round(mortgage.loan, 2),
        "annuity_first_payment_rub": round(mortgage.annuity[0].payment, 2),
        "differentiated_first_payment_rub": round(mortgage.differentiated[0].payment, 2),
        "annuity_overpayment_rub": annuity_overpayment,
        "differentiated_overpayment_rub": differentiated_overpayment,
        "overpayment_difference_rub": round(annuity_overpayment - differentiated_overpayment, 2),
    }


def _sensitivity(matrix, base_weights, base_scores, ids, factors):
    scenarios = []
    for factor in factors:
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Множители чувствительности должны быть положительными.")
        weights = base_weights.copy()
        weights[0] *= factor
        weights /= weights.sum()
        scores = topsis(matrix, weights)
        order = rank_order(scores, ids)
        positions = {index: rank for rank, index in enumerate(order, start=1)}
        rho = None
        if np.ptp(scores) > 0 and np.ptp(base_scores) > 0:
            rho = float(spearmanr(base_scores, scores).statistic)
        scenarios.append({"factor": factor, "price_weight": float(weights[0]), "leader": ids[order[0]],
                          "spearman_rho": rho, "positions": positions})
    return scenarios


def run_analysis(csv_path: Path | None = None, seed: int | None = None) -> dict:
    config = load_config()
    output = ROOT / "results"
    if csv_path is None:
        seed = config["seed"] if seed is None else seed
        csv_path = ROOT / "data" / "housing.csv"
        generate_csv(csv_path, config, seed)
    listings = load_listings(csv_path, config["districts"])
    ids = [item.object_id for item in listings]
    district_scores = [config["districts"][item.district]["score"] for item in listings]
    matrix = np.array([[item.price_rub, district_scores[i], item.metro_walk_min]
                       for i, item in enumerate(listings)], dtype=float)
    ahp = calculate_ahp(config["ahp_judgments"])
    scores = topsis(matrix, ahp.weights)
    order = rank_order(scores, ids)
    front = pareto_front(matrix[:, [0, 2]])

    ranked = []
    for rank, index in enumerate(order, start=1):
        item = listings[index]
        mortgage = compare_mortgage(item.price_rub, **config["mortgage"])
        ranked.append({**asdict(item), "rank": rank, "district_score": district_scores[index],
                       "topsis_score": float(scores[index]), "pareto_price_transport": bool(front[index]),
                       **_mortgage_fields(mortgage)})
    write_csv(output / "ranking.csv",
              ("rank", *FIELDS, "district_score", "topsis_score", "pareto_price_transport", *MORTGAGE_FIELDS),
              ranked)

    leader = ranked[0]
    mortgage = compare_mortgage(leader["price_rub"], **config["mortgage"])
    schedule_rows = []
    for scheme, schedule in (("аннуитет", mortgage.annuity), ("дифференцированный", mortgage.differentiated)):
        for payment in schedule:
            amounts = {key: round(amount, 2) for key, amount in asdict(payment).items() if key != "month"}
            schedule_rows.append({"object_id": leader["object_id"], "scheme": scheme,
                                  "month": payment.month, **amounts})
    write_csv(output / "mortgage_schedule.csv",
              ("object_id", "scheme", "month", "payment", "principal", "interest", "balance"), schedule_rows)

    scenarios = _sensitivity(matrix, ahp.weights, scores, ids, config["price_weight_factors"])
    scenario_fields = ("factor", "price_weight", "leader", "spearman_rho")
    scenario_rows = [{key: row[key] for key in scenario_fields} for row in scenarios]
    write_csv(output / "sensitivity.csv", scenario_fields, scenario_rows)

    weights_plot(ahp.weights, output)
    ranking_plot(ranked, output)
    pareto_plot(listings, front, output)
    mortgage_plot(mortgage, output)
    shown_indices = list(dict.fromkeys([*order[:4], *(ids.index(s["leader"]) for s in scenarios)]))
    sensitivity_plot(scenarios, ids, shown_indices, output)

    summary = {
        "data_origin": listings[0].data_origin, "seed": seed, "object_count": len(listings),
        "district_count": len({item.district for item in listings}), "criteria": list(CRITERIA),
        "ahp_matrix": ahp.matrix.tolist(), "ahp_weights": ahp.weights.tolist(), "lambda_max": ahp.lambda_max,
        "ci": ahp.ci, "ri": RANDOM_INDEX[len(CRITERIA)], "cr": ahp.cr, "cr_limit": CR_LIMIT,
        "mortgage": config["mortgage"], "top": ranked[:10],
        "pareto_ids": [item.object_id for i, item in enumerate(listings) if front[i]],
        "sensitivity": scenario_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Вариант 5А «Покупка жилья»")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help="Входной CSV в формате из README")
    source.add_argument("--seed", type=int, help="Переопределить seed учебного набора")
    args = parser.parse_args()
    try:
        result = run_analysis(args.csv, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Ошибка: {error}\n")
    leader = result["top"][0]
    print(f"{result['object_count']} объектов | {result['data_origin']} | CR = {result['cr']:.4f}")
    print(f"Лидер TOPSIS — {leader['object_id']}, {leader['district']} ({leader['topsis_score']:.4f})")
    print(ROOT / "results")


if __name__ == "__main__":
    main()
