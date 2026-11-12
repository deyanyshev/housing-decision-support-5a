"""Генерация и загрузка сопоставимых объектов."""
import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("object_id", "price_rub", "district", "metro_walk_min", "area_m2", "data_origin")
SYNTHETIC = "учебные данные"


@dataclass(frozen=True)
class Listing:
    object_id: str
    price_rub: int
    district: str
    metro_walk_min: int
    area_m2: float
    data_origin: str


def load_config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_csv(path: Path, config: dict, seed: int) -> None:
    """Создать учебный набор, в котором цена зависит от района, площади и метро."""
    rng = random.Random(seed)
    rows = []
    for district, params in config["districts"].items():
        for _ in range(config["objects_per_district"]):
            area = round(rng.uniform(32, 82), 1)
            walk = rng.randint(4, 25)
            metro_factor = 1.0 + (14 - walk) * 0.008
            price = area * params["rub_per_m2"] * metro_factor * rng.uniform(0.88, 1.12)
            rows.append(dict(
                object_id=f"Ж{len(rows) + 1:02d}", price_rub=round(price / 10_000) * 10_000,
                district=district, metro_walk_min=walk, area_m2=area, data_origin=SYNTHETIC,
            ))
    write_csv(path, FIELDS, rows)


def load_listings(path: Path, districts: dict) -> list[Listing]:
    """Загрузить CSV, проверить обязательные поля, диапазоны и уникальность ID."""
    listings = []
    ids = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"В CSV отсутствуют поля: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                item = Listing(row["object_id"].strip(), int(row["price_rub"]), row["district"].strip(),
                               int(row["metro_walk_min"]), float(row["area_m2"]), row["data_origin"].strip())
                if not item.object_id or item.object_id in ids or not item.data_origin:
                    raise ValueError("нужны уникальный ID и отметка об источнике")
                if item.district not in districts:
                    raise ValueError(f"в config.json нет оценки для района «{item.district}»")
                if item.price_rub <= 0 or item.metro_walk_min < 0 or not math.isfinite(item.area_m2) or item.area_m2 <= 0:
                    raise ValueError("проверьте цену, время до метро и площадь")
            except (TypeError, AttributeError, ValueError) as error:
                raise ValueError(f"CSV, строка {reader.line_num}: {error}") from error
            ids.add(item.object_id)
            listings.append(item)
    if len(listings) < 2 or len({item.data_origin for item in listings}) != 1:
        raise ValueError("Нужно минимум два объекта с общей отметкой об источнике.")
    return listings


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Сгенерировать CSV с учебными объектами.")
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "housing.csv")
    args = parser.parse_args()
    generate_csv(args.output, config, args.seed)
    print(args.output)


if __name__ == "__main__":
    main()
