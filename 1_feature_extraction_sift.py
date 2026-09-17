"""Extraccion de descriptores SIFT para el dataset Oxford & Paris Buildings.

Para cada imagen en data/<dataset>/ calcula keypoints + descriptores SIFT
(cv2.SIFT) y los guarda en un unico archivo HDF5.

El calculo de SIFT (CPU-bound) se paraleliza entre varios procesos con
multiprocessing.Pool; la escritura al HDF5 la hace unicamente el proceso
principal (escritura concurrente al mismo archivo no es segura), de forma
incremental para no acumular todos los descriptores en memoria.

Salida:
    data/features/sift/sift_features.h5
        /<dataset>/<image_id>/descriptors   (N, 128) float32
        /<dataset>/<image_id>/keypoints     (N, 4)   float32  [x, y, size, angle]
    data/features/sift/sift_index.csv
        image_id, dataset, filename, num_keypoints

Uso:
    python 1_feature_extraction_sift.py
    python 1_feature_extraction_sift.py --datasets oxford
    python 1_feature_extraction_sift.py --overwrite
    python 1_feature_extraction_sift.py --workers 16
    python 1_feature_extraction_sift.py --limit 20  # prueba rapida, 20 img por dataset
"""

import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
FEATURES_DIR = DATA_DIR / "features" / "sift"
H5_PATH = FEATURES_DIR / "sift_features.h5"
INDEX_CSV_PATH = FEATURES_DIR / "sift_index.csv"

DATASETS = ["oxford", "paris"]

# Cada proceso worker crea su propio cv2.SIFT una sola vez (no es
# thread/process-safe compartirlo, y crearlo por imagen seria costoso).
_worker_sift: cv2.SIFT | None = None


def _init_worker() -> None:
    global _worker_sift
    _worker_sift = cv2.SIFT_create()


def keypoints_to_array(keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    return np.array(
        [[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints],
        dtype=np.float32,
    ).reshape(-1, 4)


def extract_sift(sift: cv2.SIFT, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    keypoints, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    else:
        descriptors = descriptors.astype(np.float32)

    return keypoints_to_array(keypoints), descriptors


def _process_one(task: tuple[str, str, str]) -> tuple[str, str, str, np.ndarray, np.ndarray]:
    """Worker: recibe (dataset, image_id, path_str) y retorna los arrays calculados."""
    dataset, image_id, path_str = task
    keypoints_arr, descriptors = extract_sift(_worker_sift, Path(path_str))
    return dataset, image_id, path_str, keypoints_arr, descriptors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Subconjuntos de data/ a procesar (default: todos).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Si se pasa, borra y recalcula imagenes ya presentes en el HDF5.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Numero de procesos en paralelo (default: todos los cores disponibles).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Si se pasa, procesa como maximo N imagenes por dataset "
            "(util para probar rapido que todo funciona antes de correr el dataset completo)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5_PATH, "a") as h5f:
        # 1) Decidir que imagenes hay que calcular vs. cuales ya estan en el h5.
        pending_tasks: list[tuple[str, str, str]] = []
        index_rows: list[tuple[str, str, str, int]] = []

        for dataset in args.datasets:
            image_dir = DATA_DIR / dataset
            image_paths = sorted(image_dir.glob("*.jpg"))
            if not image_paths:
                print(f"Aviso: no hay imagenes en {image_dir}, se omite.")
                continue
            if args.limit is not None:
                image_paths = image_paths[: args.limit]

            group = h5f.require_group(dataset)

            for image_path in image_paths:
                image_id = image_path.stem

                if image_id in group:
                    if not args.overwrite:
                        n_kp = group[image_id]["descriptors"].shape[0]
                        index_rows.append((image_id, dataset, image_path.name, n_kp))
                        continue
                    del group[image_id]

                pending_tasks.append((dataset, image_id, str(image_path)))

        print(
            f"{len(pending_tasks)} imagenes por procesar "
            f"({len(index_rows)} ya estaban en el HDF5), usando {args.workers} workers."
        )

        # 2) Calcular SIFT en paralelo, escribir al HDF5 desde el proceso principal
        #    a medida que van llegando resultados.
        if pending_tasks:
            with Pool(processes=args.workers, initializer=_init_worker) as pool:
                results = pool.imap_unordered(_process_one, pending_tasks, chunksize=8)

                for dataset, image_id, path_str, keypoints_arr, descriptors in tqdm(
                    results, total=len(pending_tasks), desc="SIFT"
                ):
                    group = h5f[dataset]
                    img_group = group.create_group(image_id)
                    img_group.create_dataset(
                        "descriptors", data=descriptors, compression="gzip"
                    )
                    img_group.create_dataset(
                        "keypoints", data=keypoints_arr, compression="gzip"
                    )
                    index_rows.append(
                        (image_id, dataset, Path(path_str).name, len(descriptors))
                    )

    with open(INDEX_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "dataset", "filename", "num_keypoints"])
        writer.writerows(index_rows)

    print(f"\nListo. {len(index_rows)} imagenes procesadas.")
    print(f"Features guardados en: {H5_PATH}")
    print(f"Indice guardado en: {INDEX_CSV_PATH}")


if __name__ == "__main__":
    main()
