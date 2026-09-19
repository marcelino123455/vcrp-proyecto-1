"""Fusion tardia de varias representaciones.

Combina los rankings de dos o mas representaciones (VLAD, BoVW, descriptores
globales) sumando sus puntuaciones normalizadas. Es la parte de "combine them
into a single representation" del enunciado, hecha a nivel de score en vez de
a nivel de vector.

Por que a nivel de score y no concatenando
-------------------------------------------
Concatenar VLAD (miles de dimensiones, L2-normalizado) con un histograma de
color (cientos de dimensiones, otra escala) obliga a elegir un peso relativo
implicito que depende de las dimensiones y las varianzas de cada bloque. Fusionar
scores es mas limpio: cada representacion produce su propio ranking con la
metrica que mejor le va, y recien despues se combinan.

Ademas permite algo que la concatenacion no: que cada representacion use una
metrica distinta. El color rinde mejor con Hellinger y VLAD con coseno, y aqui
cada uno puede usar la suya.

Las puntuaciones se normalizan min-max por query antes de sumar, porque un
coseno de VLAD y una distancia chi2 de HOG no viven en la misma escala.

Uso
---
    # barrido del peso entre dos representaciones
    python 5_fuse.py \\
        --embeddings data/features/vlad/vlad_k128_pca512.npz \\
                     data/features/global/combined_pca512_w0.5.npz \\
        --names vlad global --sweep

    # pesos fijos, tres representaciones, metricas distintas
    python 5_fuse.py \\
        --embeddings A.npz B.npz C.npz --names vlad bovw color \\
        --metrics cosine cosine hellinger --weights 0.6 0.3 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation import DATASETS, DEFAULT_KS, Query, evaluate, load_ground_truth
from retrieval import (
    METRICS,
    NORMALIZATIONS,
    alpha_query_expansion,
    database_side_augmentation,
    load_embeddings,
    normalize_features,
    rank,
    rank_normalize,
    similarity,
    split_ids,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GT_DIR = REPO_ROOT / "data" / "groundtruth"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--embeddings", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument(
        "--metrics", nargs="+", default=None, choices=METRICS,
        help="Una metrica por representacion (default: cosine para todas).",
    )
    parser.add_argument(
        "--normalizations", nargs="+", default=None, choices=NORMALIZATIONS,
        help="Una normalizacion por representacion (default: none).",
    )
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument(
        "--sweep", action="store_true",
        help="Solo con 2 representaciones: barre el peso de 0 a 1 en pasos de 0.1.",
    )
    parser.add_argument("--dba", type=int, default=0)
    parser.add_argument("--query-expansion", type=int, default=0)
    parser.add_argument("--qe-alpha", type=float, default=3.0)
    parser.add_argument("--datasets", nargs="+", default=["oxford"], choices=list(DATASETS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--name", default="fusion")
    return parser.parse_args()


def cargar_representaciones(args) -> list[dict]:
    n = len(args.embeddings)
    names = args.names or [Path(p).stem for p in args.embeddings]
    metrics = args.metrics or ["cosine"] * n
    normalizations = args.normalizations or ["none"] * n

    for nombre, valores in (("--names", names), ("--metrics", metrics),
                            ("--normalizations", normalizations)):
        if len(valores) != n:
            raise SystemExit(f"{nombre} debe tener {n} valores, uno por representacion")

    reps = []
    for path, name, metric, norm in zip(args.embeddings, names, metrics, normalizations):
        ids, X = load_embeddings(path)
        print(f"  {name:<12} {X.shape[0]:>6} x {X.shape[1]:<6} metrica={metric} norm={norm}")
        reps.append({
            "name": name, "path": str(path), "metric": metric,
            "normalization": norm, "ids": ids, "X": X,
        })
    return reps


def alinear(reps: list[dict]) -> np.ndarray:
    """image_ids comunes a todas las representaciones.

    Las representaciones no tienen por que cubrir las mismas imagenes: los
    descriptores globales descartan las corruptas, VLAD las conserva con
    vector nulo. Sin esta interseccion se estarian sumando scores de imagenes
    distintas, que es un error silencioso y devastador.
    """
    comunes = None
    for rep in reps:
        s = {str(x) for x in rep["ids"]}
        comunes = s if comunes is None else (comunes & s)
    assert comunes is not None

    ids = np.asarray(sorted(comunes), dtype=object)
    print(f"  imagenes en comun: {len(ids)}")

    for rep in reps:
        posicion = {str(x): i for i, x in enumerate(rep["ids"])}
        idx = np.asarray([posicion[str(i)] for i in ids])
        rep["X"] = normalize_features(rep["X"][idx], rep["normalization"])
        sobran = len(rep["ids"]) - len(ids)
        if sobran:
            print(f"    {rep['name']}: {sobran} imagenes descartadas por la interseccion")

    return ids


def sims_por_representacion(
    reps: list[dict], ids: np.ndarray, queries: list[Query], dataset: str, args
) -> tuple[list[np.ndarray], list[Query], np.ndarray]:
    datasets_arr, names = split_ids(ids)
    db_index = np.flatnonzero(datasets_arr == dataset)
    db_names = names[db_index]

    posicion = {str(names[g]): i for i, g in enumerate(db_index)}

    kept, filas = [], []
    for query in queries:
        if query.image_id in posicion:
            kept.append(query)
            filas.append(posicion[query.image_id])
    if not kept:
        raise SystemExit(f"Ninguna query de {dataset} esta en los embeddings")
    filas = np.asarray(filas)

    todas = []
    for rep in reps:
        db = rep["X"][db_index]
        if args.dba > 0:
            db = database_side_augmentation(db, n_dba=args.dba, alpha=args.qe_alpha)

        Q = db[filas]
        sims = similarity(Q, db, metric=rep["metric"])

        if args.query_expansion > 0 and rep["metric"] in ("cosine", "dot"):
            Q = alpha_query_expansion(
                Q, db, sims, n_qe=args.query_expansion, alpha=args.qe_alpha
            )
            sims = similarity(Q, db, metric=rep["metric"])

        todas.append(sims)

    return todas, kept, db_names


def evaluar_fusion(sims_list, pesos, kept, db_names, ks) -> float:
    total = np.zeros_like(np.atleast_2d(sims_list[0]), dtype=np.float32)
    for sims, peso in zip(sims_list, pesos):
        total += float(peso) * rank_normalize(sims)

    rankings = {q.query_id: rank(total[i], db_names) for i, q in enumerate(kept)}
    return evaluate(kept, rankings, ks=tuple(ks))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fusion tardia ===")
    reps = cargar_representaciones(args)
    ids = alinear(reps)

    if len(args.datasets) != 1:
        raise SystemExit("Por ahora la fusion evalua un dataset a la vez (--datasets oxford)")
    dataset = args.datasets[0]

    queries = load_ground_truth(args.gt_dir, dataset)
    print(f"  ground truth: {len(queries)} queries de {dataset}")

    sims_list, kept, db_names = sims_por_representacion(reps, ids, queries, dataset, args)
    print(f"  queries evaluables: {len(kept)}")

    filas: list[dict] = []

    if args.sweep:
        if len(reps) != 2:
            raise SystemExit("--sweep requiere exactamente 2 representaciones")
        combinaciones = [
            [round(w, 2), round(1 - w, 2)] for w in np.linspace(0.0, 1.0, 11)
        ]
    elif args.weights:
        if len(args.weights) != len(reps):
            raise SystemExit("--weights debe tener un valor por representacion")
        combinaciones = [list(args.weights)]
    else:
        combinaciones = [[1.0 / len(reps)] * len(reps)]

    print()
    mejor = None
    for pesos in combinaciones:
        resultado = evaluar_fusion(sims_list, pesos, kept, db_names, args.ks)
        etiqueta = "  ".join(
            f"{rep['name']}={p:.2f}" for rep, p in zip(reps, pesos)
        )
        print(f"  {etiqueta:<40} mAP={resultado.mean_ap:.4f}  "
              f"P@5={resultado.mean_precision_at(5):.4f}")

        fila = {f"w_{rep['name']}": p for rep, p in zip(reps, pesos)}
        fila.update({
            "mAP": round(resultado.mean_ap, 6),
            "mAP_simple": round(resultado.mean_ap_simple, 6),
            **{f"mP@{k}": round(resultado.mean_precision_at(k), 6) for k in args.ks},
        })
        filas.append(fila)

        if mejor is None or resultado.mean_ap > mejor[0]:
            mejor = (resultado.mean_ap, pesos, resultado)

    csv_path = results_dir / f"{args.name}_fusion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    assert mejor is not None
    mejor_map, mejor_pesos, mejor_resultado = mejor

    (results_dir / f"{args.name}_fusion.json").write_text(
        json.dumps({
            "representaciones": [
                {k: rep[k] for k in ("name", "path", "metric", "normalization")}
                for rep in reps
            ],
            "dba": args.dba,
            "query_expansion": args.query_expansion,
            "mejores_pesos": dict(zip([r["name"] for r in reps], mejor_pesos)),
            **mejor_resultado.summary(),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    etiqueta = "  ".join(f"{rep['name']}={p:.2f}" for rep, p in zip(reps, mejor_pesos))
    print(f"Mejor: {etiqueta}  ->  mAP = {mejor_map:.4f}")

    individuales = {
        rep["name"]: evaluar_fusion(
            [sims_list[i]], [1.0], kept, db_names, args.ks
        ).mean_ap
        for i, rep in enumerate(reps)
    }
    mejor_individual = max(individuales.values())
    print(f"Mejor representacion sola: {mejor_individual:.4f}")
    delta = mejor_map - mejor_individual
    if delta > 0:
        print(f"La fusion aporta +{delta:.4f} sobre la mejor individual")
    else:
        print(f"La fusion NO mejora ({delta:+.4f}); reportalo como resultado negativo")
    print("=" * 60)
    print(f"\nTabla: {csv_path}")


if __name__ == "__main__":
    main()
