"""Evaluacion de cualquier representacion de imagen sobre Oxford5k / Paris6k.

Este script es agnostico al descriptor: recibe un archivo de embeddings
(image_ids + matriz de features), corre las 55 queries oficiales de cada
dataset y reporta mAP, Precision@k, desglose por landmark y las peores queries
para el analisis cualitativo.

Sirve igual para el BoVW de `2_visual_vocabulary.py`, para los descriptores
globales de `3_feature_extraction_global.py` y para cualquier fusion posterior.
Que todos los experimentos pasen por el mismo codigo es lo que hace que la
tabla del reporte sea honesta.

Requisitos previos:
    ./data/download_groundtruth.sh
    python evaluation.py --check

Ejemplos:
    # BoVW tal como lo guarda 2_visual_vocabulary.py
    python 4_evaluate_retrieval.py \\
        --embeddings data/features/vocabulary/image_histograms.npz \\
        --name bovw_k500 --metric cosine --normalize l1

    # Barrido de metricas de distancia (una fila de la tabla del reporte)
    python 4_evaluate_retrieval.py \\
        --embeddings data/features/vocabulary/image_histograms.npz \\
        --name bovw_k500 --metric cosine chi2 hellinger l2 intersection

    # Con las imagenes del otro dataset como distractores
    python 4_evaluate_retrieval.py --embeddings ... --database all

    # Ademas, grids de top-5 para el analisis cualitativo
    python 4_evaluate_retrieval.py --embeddings ... --qualitative 6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

from evaluation import (
    DATASETS,
    DEFAULT_KS,
    Query,
    evaluate,
    load_ground_truth,
)
from retrieval import (
    METRICS,
    NORMALIZATIONS,
    load_embeddings,
    rank,
    similarity,
    split_ids,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GT_DIR = REPO_ROOT / "data" / "groundtruth"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Archivo .npz/.h5 con image_ids y la matriz de features.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Nombre del experimento (default: nombre del archivo de embeddings).",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=["cosine"],
        choices=METRICS,
        help="Metrica(s) de similitud a evaluar. Varias = barrido comparativo.",
    )
    parser.add_argument(
        "--normalize",
        nargs="+",
        default=["l2"],
        choices=NORMALIZATIONS,
        help="Normalizacion(es) aplicada(s) a los vectores antes de comparar.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Que conjuntos de queries evaluar.",
    )
    parser.add_argument(
        "--database",
        choices=("same", "all"),
        default="same",
        help=(
            "'same': las queries de Oxford se rankean solo contra imagenes de "
            "Oxford (protocolo estandar). 'all': la base es Oxford+Paris, o sea "
            "el otro dataset actua como distractores (mas dificil, y es el "
            "escenario que menciona el enunciado)."
        ),
    )
    parser.add_argument(
        "--exclude-query",
        action="store_true",
        help=(
            "Quita la propia imagen consulta del ranking. El benchmark oficial "
            "de VGG NO la quita; usalo solo si el grupo acuerda ese criterio, y "
            "de forma consistente en todos los experimentos."
        ),
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_KS),
        help="Valores de k para Precision@k.",
    )
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--save-rankings",
        action="store_true",
        help="Guarda el top-100 de cada query (reproducibilidad, la pide el enunciado).",
    )
    parser.add_argument(
        "--qualitative",
        type=int,
        default=0,
        help=(
            "Si > 0, genera grids de top-k para las N mejores y N peores queries "
            "(requiere las imagenes en data/oxford y data/paris)."
        ),
    )
    parser.add_argument(
        "--qualitative-topk",
        type=int,
        default=5,
        help="Cuantos resultados mostrar por query en los grids (default: 5).",
    )
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Solo imprime la linea resumen de cada configuracion.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Nucleo
# --------------------------------------------------------------------------- #


def build_rankings(
    queries: list[Query],
    names: np.ndarray,
    db_index: np.ndarray,
    X: np.ndarray,
    name_to_row: dict[tuple[str, str], int],
    dataset: str,
    metric: str,
    *,
    exclude_query: bool,
    verbose: bool = True,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray], list[str]]:
    """Rankea la base completa para cada query.

    Devuelve (rankings, similitudes_por_query, queries_no_encontradas).
    """
    missing: list[str] = []
    rows: list[int] = []
    kept: list[Query] = []

    for query in queries:
        row = name_to_row.get((dataset, query.image_id))
        if row is None:
            missing.append(query.query_id)
            continue
        rows.append(row)
        kept.append(query)

    if not kept:
        return {}, {}, missing

    Q = X[np.asarray(rows)]
    db = X[db_index]
    db_names = names[db_index]

    sims = similarity(Q, db, metric=metric)

    rankings: dict[str, list[str]] = {}
    sims_by_query: dict[str, np.ndarray] = {}

    for i, query in enumerate(kept):
        sim_row = sims[i].copy()
        if exclude_query:
            self_mask = db_names == query.image_id
            sim_row[self_mask] = -np.inf
        rankings[query.query_id] = rank(sim_row, db_names)
        sims_by_query[query.query_id] = sim_row

    if missing and verbose:
        print(
            f"  AVISO: {len(missing)} queries de {dataset} no estan en los "
            f"embeddings y se omiten: {missing[:5]}"
            + (" ..." if len(missing) > 5 else ""),
            file=sys.stderr,
        )

    return rankings, sims_by_query, missing


def warn_query_in_positives(queries: list[Query], verbose: bool = True) -> int:
    """Cuenta cuantas queries se listan a si mismas como positivo.

    Es un detalle del protocolo que cambia el mAP unos puntos: si la imagen
    consulta esta en su propio good/ok y no se excluye, siempre sale en el
    rango 1 y regala precision. Lo importante no es que criterio se elija,
    sino que todos los experimentos del grupo usen el mismo.
    """
    count = sum(1 for q in queries if q.image_id in q.positives)
    if count and verbose:
        print(
            f"  nota: {count}/{len(queries)} queries aparecen en su propio "
            "conjunto de positivos (ver --exclude-query)"
        )
    return count


def run_configuration(
    args: argparse.Namespace,
    gt: dict[str, list[Query]],
    image_ids: np.ndarray,
    datasets_arr: np.ndarray,
    names: np.ndarray,
    X_raw: np.ndarray,
    normalization: str,
    metric: str,
) -> dict:
    from retrieval import normalize_features

    X = normalize_features(X_raw, normalization)
    name_to_row = {
        (str(datasets_arr[i]), str(names[i])): i for i in range(len(names))
    }

    all_queries: list[Query] = []
    all_rankings: dict[str, list[str]] = {}

    verbose = not args.quiet

    for dataset in args.datasets:
        queries = gt[dataset]

        if args.database == "same":
            db_index = np.flatnonzero(datasets_arr == dataset)
        else:
            db_index = np.arange(len(names))

        if db_index.size == 0:
            print(f"  AVISO: no hay embeddings del dataset {dataset}", file=sys.stderr)
            continue

        if verbose:
            print(f"  {dataset}: {len(queries)} queries vs {db_index.size} imagenes")
            warn_query_in_positives(queries)

        rankings, _, _ = build_rankings(
            queries,
            names,
            db_index,
            X,
            name_to_row,
            dataset,
            metric,
            exclude_query=args.exclude_query,
            verbose=verbose,
        )

        all_queries.extend(queries)
        all_rankings.update({f"{dataset}::{k}": v for k, v in rankings.items()})

    # Los query_id pueden repetirse entre datasets (no ocurre con estos
    # landmarks, pero no queremos depender de eso), asi que se prefijan.
    prefixed_queries = []
    for q in all_queries:
        prefixed = Query(
            query_id=f"{q.dataset}::{q.query_id}",
            dataset=q.dataset,
            landmark=q.landmark,
            image_id=q.image_id,
            bbox=q.bbox,
            positives=q.positives,
            good=q.good,
            ok=q.ok,
            junk=q.junk,
        )
        prefixed_queries.append(prefixed)

    result = evaluate(prefixed_queries, all_rankings, ks=tuple(args.ks))
    return {"result": result, "rankings": all_rankings}


def main() -> None:
    args = parse_args()

    embeddings_path = Path(args.embeddings)
    run_name = args.name or embeddings_path.stem

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando embeddings: {embeddings_path}")
    image_ids, X_raw = load_embeddings(embeddings_path)
    datasets_arr, names = split_ids(image_ids)
    print(f"  {X_raw.shape[0]} imagenes, {X_raw.shape[1]} dimensiones")
    for dataset in DATASETS:
        count = int((datasets_arr == dataset).sum())
        if count:
            print(f"  {dataset}: {count} imagenes")

    print(f"Cargando ground truth: {args.gt_dir}")
    gt = {d: load_ground_truth(args.gt_dir, d) for d in args.datasets}
    for dataset, queries in gt.items():
        print(f"  {dataset}: {len(queries)} queries")

    configurations = list(product(args.normalize, args.metric))
    summary_rows: list[dict] = []
    best: tuple[float, str, dict] | None = None

    for normalization, metric in configurations:
        tag = f"{run_name}__{normalization}__{metric}"
        print()
        print(f"=== {tag} ===")

        payload = run_configuration(
            args, gt, image_ids, datasets_arr, names, X_raw, normalization, metric
        )
        result = payload["result"]

        if args.quiet:
            print(
                f"  mAP={result.mean_ap:.4f}  "
                + "  ".join(f"P@{k}={result.mean_precision_at(k):.4f}" for k in args.ks)
            )
        else:
            print(result.format_report())

        row = {
            "run": run_name,
            "normalization": normalization,
            "metric": metric,
            "database": args.database,
            "exclude_query": args.exclude_query,
            "num_queries": len(result.per_query),
            "mAP": round(result.mean_ap, 6),
            "mAP_simple": round(result.mean_ap_simple, 6),
        }
        for k in args.ks:
            row[f"mP@{k}"] = round(result.mean_precision_at(k), 6)
        summary_rows.append(row)

        per_query_path = results_dir / f"{tag}_per_query.csv"
        write_per_query_csv(per_query_path, result, tuple(args.ks))

        summary_path = results_dir / f"{tag}_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "run": run_name,
                    "embeddings": str(embeddings_path),
                    "normalization": normalization,
                    "metric": metric,
                    "database": args.database,
                    "exclude_query": args.exclude_query,
                    "num_images": int(X_raw.shape[0]),
                    "num_dimensions": int(X_raw.shape[1]),
                    **result.summary(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if args.save_rankings:
            write_rankings(results_dir / f"{tag}_rankings.txt", payload["rankings"], 100)

        if best is None or result.mean_ap > best[0]:
            best = (result.mean_ap, tag, payload)

    comparison_path = results_dir / f"{run_name}_comparison.csv"
    write_summary_csv(comparison_path, summary_rows)

    print()
    print("=" * 62)
    print("COMPARACION DE CONFIGURACIONES (ordenado por mAP)")
    print("=" * 62)
    for row in sorted(summary_rows, key=lambda r: -r["mAP"]):
        print(
            f"  {row['normalization']:<10} {row['metric']:<13} "
            f"mAP={row['mAP']:.4f}  mAP_simple={row['mAP_simple']:.4f}"
        )
    print()
    print(f"Resultados por query : {results_dir}")
    print(f"Tabla comparativa    : {comparison_path}")

    if args.qualitative > 0 and best is not None:
        _, tag, payload = best
        print()
        print(f"Generando analisis cualitativo de la mejor config ({tag})...")
        try:
            from qualitative import save_qualitative_grids

            out_dir = results_dir / "qualitative" / tag
            paths = save_qualitative_grids(
                result=payload["result"],
                rankings=payload["rankings"],
                gt=gt,
                image_root=Path(args.image_root),
                out_dir=out_dir,
                n_queries=args.qualitative,
                top_k=args.qualitative_topk,
            )
            print(f"  {len(paths)} figuras en {out_dir}")
        except Exception as exc:  # pragma: no cover
            print(f"  No se pudo generar el analisis cualitativo: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Salidas
# --------------------------------------------------------------------------- #


def write_per_query_csv(path: Path, result, ks: tuple[int, ...]) -> None:
    rows = [q.as_row(ks) for q in sorted(result.per_query, key=lambda q: -q.ap)]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rankings(path: Path, rankings: dict[str, list[str]], top_k: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for query_id in sorted(rankings):
            f.write(f"# {query_id}\n")
            for image_id in rankings[query_id][:top_k]:
                f.write(f"{image_id}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
