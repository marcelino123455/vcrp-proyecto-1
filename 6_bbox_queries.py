"""Consultas recortadas al bounding box del ground truth.

Experimento que cierra el analisis de fallos.

El problema
-----------
El ground truth de Oxford/Paris da, para cada consulta, el recuadro exacto del
objeto de interes. El protocolo publicado extrae los descriptores SOLO dentro
de ese recorte. Nosotros veniamos usando la imagen completa.

En las figuras cualitativas se ve el precio de esa simplificacion: en
magdalen_4 la torre ocupa un 10% del encuadre y el resto es jardin, y el
sistema recupera jardines. En christ_church_5 la torre esta detras de arboles
de invierno, y el sistema recupera arboles. SIFT dispara miles de keypoints
sobre hojas y ramas — que son textura de alta frecuencia — y solo unos cientos
sobre el edificio; VLAD los agrega a todos con el mismo derecho, asi que el
descriptor termina describiendo vegetacion.

Que hace este script
--------------------
Para cada consulta: recorta la imagen al bounding box, extrae SIFT del recorte,
lo codifica con el MISMO vocabulario VLAD y la MISMA PCA ya entrenados (no se
re-entrena nada), y rankea contra la base de datos sin tocar.

Despues compara, landmark por landmark, contra la linea base de imagen
completa. Si la hipotesis es correcta, Magdalen y Christ Church deben subir
mucho mas que el resto.

Requisitos
----------
Necesita el archivo de modelo que genera 3c_vlad.py:

    data/features/vlad/vlad_<tag>_model.npz

Si no existe, vuelve a correr 3c_vlad.py con esa configuracion (ahora guarda
el modelo automaticamente).

Uso
---
    python 6_bbox_queries.py --embeddings data/features/vlad/vlad_k256_pca512.npz
    python 6_bbox_queries.py --embeddings ... --padding 0.15
    python 6_bbox_queries.py --embeddings ... --dba 5 --diffusion 100
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from evaluation import DATASETS, DEFAULT_KS, evaluate, load_ground_truth
from retrieval import (
    alpha_query_expansion,
    build_knn_graph,
    database_side_augmentation,
    diffusion_rerank,
    load_embeddings,
    normalize_graph,
    rank,
    similarity,
    split_ids,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GT_DIR = REPO_ROOT / "data" / "groundtruth"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

EPS = 1e-12


def cargar_modulo_vlad():
    """Importa 3c_vlad.py por ruta (el nombre empieza con digito).

    Se hace asi para que la codificacion VLAD tenga una sola implementacion en
    todo el proyecto: si cambia alla, cambia aqui.
    """
    ruta = REPO_ROOT / "3c_vlad.py"
    if not ruta.exists():
        raise SystemExit(f"No se encontro {ruta}")
    spec = importlib.util.spec_from_file_location("vlad_mod", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)  # type: ignore[union-attr]
    return modulo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--embeddings", required=True, help="El .npz de la base de datos.")
    parser.add_argument(
        "--model", default=None,
        help="El .npz del modelo (default: <embeddings sin .npz>_model.npz).",
    )
    parser.add_argument("--datasets", nargs="+", default=["oxford"], choices=list(DATASETS))
    parser.add_argument(
        "--padding", type=float, default=0.0,
        help=(
            "Expande el bbox esta fraccion por lado. Un poco de contexto a "
            "veces ayuda: 0.1 agranda el recorte un 10%%."
        ),
    )
    parser.add_argument(
        "--min-side", type=int, default=32,
        help="Lado minimo del recorte en pixeles; por debajo se usa la imagen completa.",
    )
    parser.add_argument("--dba", type=int, default=0)
    parser.add_argument("--query-expansion", type=int, default=0)
    parser.add_argument("--qe-alpha", type=float, default=3.0)
    parser.add_argument("--diffusion", type=int, default=0)
    parser.add_argument("--diffusion-alpha", type=float, default=0.9)
    parser.add_argument("--diffusion-seeds", type=int, default=10)
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--name", default="bbox")
    parser.add_argument(
        "--save-crops", default=None,
        help="Si se pasa, guarda los recortes en ese directorio (para el reporte).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #


def recortar(img: np.ndarray, bbox, padding: float, min_side: int):
    """Recorta al bbox con padding opcional. Devuelve (recorte, fraccion_area)."""
    alto, ancho = img.shape[:2]
    x1, y1, x2, y2 = bbox

    if padding > 0:
        dx = (x2 - x1) * padding
        dy = (y2 - y1) * padding
        x1, x2 = x1 - dx, x2 + dx
        y1, y2 = y1 - dy, y2 + dy

    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(ancho, round(x2)))
    y2 = int(min(alto, round(y2)))

    if x2 - x1 < min_side or y2 - y1 < min_side:
        return img, 1.0

    fraccion = ((x2 - x1) * (y2 - y1)) / float(alto * ancho)
    return img[y1:y2, x1:x2], fraccion


def codificar_consulta(
    crop: np.ndarray, modelo: dict, config: dict, vlad_mod, sift
) -> np.ndarray:
    """SIFT sobre el recorte -> VLAD -> PCA, igual que la base de datos."""
    gris = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, descriptors = sift.detectAndCompute(gris, None)

    if descriptors is None or descriptors.size == 0:
        descriptors = np.empty((0, 128), dtype=np.float32)
    else:
        descriptors = descriptors.astype(np.float32)

    if config.get("rootsift", True):
        descriptors = vlad_mod.apply_rootsift(descriptors)

    centroids = modelo["centroids"]
    centroid_sq = np.einsum("ij,ij->i", centroids, centroids).astype(np.float32)

    vec = vlad_mod.vlad_encode(
        descriptors, centroids, centroid_sq,
        mass_norm=config.get("mass_norm", False),
        power=config.get("power", False),
        intra_norm=config.get("intra_norm", True),
    )

    if "pca_components" in modelo:
        centrado = vec - modelo["pca_mean"]
        vec = centrado @ modelo["pca_components"].T
        potencia = float(modelo.get("whiten_power", 0.0))
        if potencia > 0:
            escala = np.power(np.maximum(modelo["pca_variance"], EPS), potencia)
            vec = vec / escala

    norma = float(np.linalg.norm(vec))
    return (vec / norma if norma > EPS else vec).astype(np.float32)


# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = Path(args.embeddings)
    model_path = Path(args.model) if args.model else embeddings_path.with_name(
        embeddings_path.stem + "_model.npz"
    )
    if not model_path.exists():
        raise SystemExit(
            f"No existe el modelo {model_path}.\n"
            "Vuelve a correr 3c_vlad.py con esa configuracion; ahora guarda el "
            "modelo (centroides + PCA) automaticamente."
        )

    import json

    config_path = embeddings_path.with_suffix(".json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    print("=== Consultas recortadas al bounding box ===")
    print(f"  embeddings : {embeddings_path}")
    print(f"  modelo     : {model_path}")
    print(f"  padding    : {args.padding}")

    modelo = dict(np.load(model_path, allow_pickle=False))
    vlad_mod = cargar_modulo_vlad()
    sift = cv2.SIFT_create()

    image_ids, X = load_embeddings(embeddings_path)
    datasets_arr, names = split_ids(image_ids)
    print(f"  base       : {X.shape[0]} imagenes x {X.shape[1]} dims")

    image_root = Path(args.image_root)
    filas_csv: list[dict] = []
    resumen: dict[str, dict] = {}

    for dataset in args.datasets:
        queries = load_ground_truth(args.gt_dir, dataset)
        db_index = np.flatnonzero(datasets_arr == dataset)
        db = X[db_index]
        db_names = names[db_index]
        posicion = {str(db_names[i]): i for i in range(len(db_names))}

        if args.dba > 0:
            db = database_side_augmentation(db, n_dba=args.dba, alpha=args.qe_alpha)

        print(f"\n--- {dataset}: {len(queries)} queries vs {len(db_names)} imagenes ---")

        kept, Q_full, Q_crop, fracciones, sin_kp = [], [], [], [], []

        for query in queries:
            if query.image_id not in posicion:
                continue

            ruta = image_root / dataset / f"{query.image_id}.jpg"
            if not ruta.exists():
                continue
            img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
            if img is None:
                continue

            if query.bbox is None:
                crop, fraccion = img, 1.0
            else:
                crop, fraccion = recortar(img, query.bbox, args.padding, args.min_side)

            vec = codificar_consulta(crop, modelo, config, vlad_mod, sift)
            if not np.any(vec):
                sin_kp.append(query.query_id)
                continue

            if args.save_crops:
                destino = Path(args.save_crops) / dataset
                destino.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destino / f"{query.query_id}.jpg"), crop)

            kept.append(query)
            Q_full.append(db[posicion[query.image_id]])
            Q_crop.append(vec)
            fracciones.append(fraccion)

        if not kept:
            print("  ninguna query utilizable")
            continue

        if sin_kp:
            print(f"  {len(sin_kp)} recortes sin keypoints, omitidos: {sin_kp[:5]}")

        Q_full = np.vstack(Q_full).astype(np.float32)
        Q_crop = np.vstack(Q_crop).astype(np.float32)
        fracciones = np.asarray(fracciones)
        print(f"  area media del bbox: {fracciones.mean():.1%} de la imagen "
              f"(minima {fracciones.min():.1%})")

        resultados = {}
        for etiqueta, Q in (("completa", Q_full), ("recorte", Q_crop)):
            sims = similarity(Q, db, metric="cosine")

            if args.query_expansion > 0:
                Q_exp = alpha_query_expansion(
                    Q, db, sims, n_qe=args.query_expansion, alpha=args.qe_alpha
                )
                sims = similarity(Q_exp, db, metric="cosine")

            if args.diffusion > 0:
                graph = normalize_graph(build_knn_graph(db, k=args.diffusion))
                sims = diffusion_rerank(
                    graph, sims, k_seed=args.diffusion_seeds,
                    alpha=args.diffusion_alpha, iters=20,
                )

            rankings = {q.query_id: rank(sims[i], db_names) for i, q in enumerate(kept)}
            resultados[etiqueta] = evaluate(kept, rankings, ks=tuple(args.ks))

        base = resultados["completa"]
        recorte = resultados["recorte"]

        print(f"\n  mAP imagen completa : {base.mean_ap:.4f}")
        print(f"  mAP con recorte     : {recorte.mean_ap:.4f}"
              f"   ({recorte.mean_ap - base.mean_ap:+.4f})")

        # Lo importante: quien mejora y quien no.
        ap_base = defaultdict(list)
        ap_crop = defaultdict(list)
        for q in base.per_query:
            ap_base[q.landmark].append(q.ap)
        for q in recorte.per_query:
            ap_crop[q.landmark].append(q.ap)

        area_por_landmark = defaultdict(list)
        for q, fr in zip(kept, fracciones):
            area_por_landmark[q.landmark].append(fr)

        print(f"\n  {'landmark':<22} {'completa':>9} {'recorte':>9} {'delta':>8} {'area bbox':>10}")
        filas_landmark = []
        for landmark in sorted(ap_base):
            a = float(np.mean(ap_base[landmark]))
            b = float(np.mean(ap_crop[landmark]))
            area = float(np.mean(area_por_landmark[landmark]))
            filas_landmark.append((landmark, a, b, b - a, area))

        for landmark, a, b, delta, area in sorted(filas_landmark, key=lambda r: -r[3]):
            marca = "  <<<" if delta > 0.05 else ""
            print(f"  {landmark:<22} {a:>9.4f} {b:>9.4f} {delta:>+8.4f} {area:>9.1%}{marca}")
            filas_csv.append({
                "dataset": dataset, "landmark": landmark,
                "mAP_imagen_completa": round(a, 6),
                "mAP_recorte_bbox": round(b, 6),
                "delta": round(delta, 6),
                "area_bbox": round(area, 4),
            })

        resumen[dataset] = {
            "mAP_completa": round(base.mean_ap, 6),
            "mAP_recorte": round(recorte.mean_ap, 6),
            "delta": round(recorte.mean_ap - base.mean_ap, 6),
            "num_queries": len(kept),
        }

    if filas_csv:
        csv_path = results_dir / f"{args.name}_bbox_por_landmark.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(filas_csv[0].keys()))
            writer.writeheader()
            writer.writerows(filas_csv)
        print(f"\nTabla por landmark: {csv_path}")

    print()
    for dataset, datos in resumen.items():
        print(f"{dataset}: {datos['mAP_completa']:.4f} -> {datos['mAP_recorte']:.4f} "
              f"({datos['delta']:+.4f})")


if __name__ == "__main__":
    main()
