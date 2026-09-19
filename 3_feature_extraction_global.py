"""Descriptores globales / estructurados para el dataset Oxford & Paris Buildings.

Esta es la segunda representacion que pide el enunciado ("Global or structured
representation: e.g., HOG descriptors or alternative global features"), a
contrastar con la representacion local basada en SIFT/BoVW.

Cada imagen se codifica como UN vector de largo fijo, calculado directamente
sobre los pixeles: nada de keypoints, nada de vocabulario.

Descriptores implementados
--------------------------

hog    Histogram of Oriented Gradients. Se calcula el gradiente, se vota la
       magnitud en histogramas de orientacion por celda, y se normalizan
       bloques solapados de celdas (L2-Hys). Captura la forma y la estructura
       de bordes. Es el que nombra explicitamente el enunciado.

color  Histograma conjunto en HSV con piramide espacial. Es el unico de la
       familia que usa color: BoVW trabaja sobre escala de grises y descarta
       esa informacion por completo, asi que es el que mas aporta en la fusion.

lbp    Local Binary Patterns uniformes e invariantes a rotacion. Describe
       textura: compara cada pixel con sus vecinos y cuenta los patrones.

gist   Energia de un banco de filtros de Gabor (varias escalas y
       orientaciones) promediada sobre una grilla. Es el descriptor clasico
       de "esencia de la escena", pensado justamente para reconocer lugares.

Todos admiten piramide espacial (`--pyramid`): el descriptor se calcula sobre
la imagen completa y sobre las celdas de una grilla 2x2, 4x4... y se concatena.
Eso es lo que convierte un descriptor "global" en uno "estructurado", que es el
termino que usa el enunciado, y da una nocion basica de donde esta cada cosa.

Sobre que esperar
-----------------
Estos descriptores van a rendir bastante peor que BoVW en este dataset, y eso
es un resultado, no un fracaso: asumen que la imagen esta globalmente alineada,
y Oxford/Paris tiene cambios severos de punto de vista, escala, oclusion y
recorte. El valor de esta parte esta en (1) cuantificar esa brecha con rigor,
(2) explicar POR QUE ocurre, y (3) aportar a la fusion tardia, donde el color
captura senal que BoVW literalmente no ve.

Salida (contrato que consume 4_evaluate_retrieval.py)
------------------------------------------------------
    data/features/global/<descriptor>.npz
        image_ids  : (N,)    "<dataset>/<image_id>"
        features   : (N, D)  float32
    data/features/global/extraction_config.json

Uso
---
    python 3_feature_extraction_global.py --limit 20            # prueba rapida
    python 3_feature_extraction_global.py --workers 32          # completo
    python 3_feature_extraction_global.py --descriptors hog color
    sbatch run_global_extraction.slurm

Despues:
    python 4_evaluate_retrieval.py --embeddings data/features/global/hog.npz --name hog
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "features" / "global"

DATASETS = ["oxford", "paris"]
DESCRIPTOR_NAMES = ("hog", "color", "lbp", "gist")

EPS = 1e-12

# Estado por worker: los kernels de Gabor se construyen una sola vez por
# proceso, no una vez por imagen.
_gabor_kernels: list[np.ndarray] | None = None
_config: dict | None = None


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > EPS else v


def l1_normalize(v: np.ndarray) -> np.ndarray:
    total = float(np.abs(v).sum())
    return v / total if total > EPS else v


def pyramid_regions(height: int, width: int, levels: int) -> list[tuple[int, int, int, int]]:
    """Regiones (y0, y1, x0, x1) de una piramide espacial.

    levels=1 -> solo la imagen completa (1 region)
    levels=2 -> completa + grilla 2x2            (5 regiones)
    levels=3 -> completa + 2x2 + 4x4             (21 regiones)
    """
    regions: list[tuple[int, int, int, int]] = []
    for level in range(levels):
        n = 2**level
        ys = np.linspace(0, height, n + 1).astype(int)
        xs = np.linspace(0, width, n + 1).astype(int)
        for i in range(n):
            for j in range(n):
                regions.append((ys[i], ys[i + 1], xs[j], xs[j + 1]))
    return regions


def load_image(path: Path, size: int) -> np.ndarray | None:
    """Lee la imagen en BGR y la lleva a `size` x `size`.

    Se fuerza un tamano cuadrado fijo a proposito: un descriptor global debe
    producir vectores de la misma dimension para todas las imagenes, y la
    relacion de aspecto original se pierde. Es una limitacion conocida del
    enfoque y hay que decirlo en el reporte.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Descriptores
# --------------------------------------------------------------------------- #


def describe_hog(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """HOG sobre la imagen en escala de grises.

    Se usa la implementacion de scikit-image, que hace el pipeline estandar:
    gradiente -> voto por orientacion en celdas -> normalizacion L2-Hys en
    bloques solapados. La normalizacion por bloque es lo que le da robustez
    frente a cambios de iluminacion y contraste local.
    """
    from skimage.feature import hog as skimage_hog

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    features = skimage_hog(
        gray,
        orientations=cfg["hog_orientations"],
        pixels_per_cell=(cfg["hog_cell"], cfg["hog_cell"]),
        cells_per_block=(cfg["hog_block"], cfg["hog_block"]),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return l2_normalize(features.astype(np.float32))


def describe_color(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Histograma conjunto HSV con piramide espacial.

    Conjunto y no por canal: un histograma 3D captura que un pixel sea
    "azul claro y saturado", mientras que tres histogramas marginales solo
    dicen que hay azul, que hay claro y que hay saturado, sin relacionarlos.

    El matiz (H) lleva mas bins que la saturacion y el valor porque es el canal
    que mejor discrimina material y menos sufre con los cambios de iluminacion.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bins = (cfg["color_bins_h"], cfg["color_bins_s"], cfg["color_bins_v"])
    h, w = hsv.shape[:2]

    parts: list[np.ndarray] = []
    for y0, y1, x0, x1 in pyramid_regions(h, w, cfg["pyramid"]):
        region = hsv[y0:y1, x0:x1]
        hist = cv2.calcHist(
            [region], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]
        ).flatten()
        # L1 por region: el descriptor no debe depender del area de la celda.
        parts.append(l1_normalize(hist.astype(np.float32)))

    return l2_normalize(np.concatenate(parts))


def describe_lbp(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Histogramas de Local Binary Patterns uniformes, con piramide espacial.

    El metodo "uniform" agrupa los patrones con a lo mas dos transiciones
    0->1, que son los que corresponden a estructuras reales (bordes, esquinas,
    manchas); el resto cae en un unico bin de "ruido". Eso reduce la dimension
    de 2^P a P+2 bins y ademas da invarianza a rotacion.
    """
    from skimage.feature import local_binary_pattern

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    points = cfg["lbp_points"]
    radius = cfg["lbp_radius"]
    n_bins = points + 2

    codes = local_binary_pattern(gray, points, radius, method="uniform")
    h, w = codes.shape

    parts: list[np.ndarray] = []
    for y0, y1, x0, x1 in pyramid_regions(h, w, cfg["pyramid"]):
        region = codes[y0:y1, x0:x1]
        hist = np.bincount(region.astype(np.int64).ravel(), minlength=n_bins)[:n_bins]
        parts.append(l1_normalize(hist.astype(np.float32)))

    return l2_normalize(np.concatenate(parts))


def build_gabor_kernels(cfg: dict) -> list[np.ndarray]:
    """Banco de filtros de Gabor: `gist_scales` escalas x `gist_orientations`."""
    kernels: list[np.ndarray] = []
    ksize = cfg["gist_ksize"]
    for scale in range(cfg["gist_scales"]):
        # Longitud de onda creciente: cada escala mira estructuras mas gruesas.
        wavelength = 4.0 * (2.0**scale)
        sigma = 0.56 * wavelength
        for orientation in range(cfg["gist_orientations"]):
            theta = np.pi * orientation / cfg["gist_orientations"]
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, wavelength, 0.5, 0, ktype=cv2.CV_32F
            )
            # Media cero: el filtro responde a estructura, no a brillo absoluto.
            kernel -= kernel.mean()
            kernels.append(kernel)
    return kernels


def describe_gist(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Descriptor tipo GIST: energia de Gabor promediada sobre una grilla.

    Para cada filtro se calcula |respuesta| y se promedia dentro de cada celda
    de una grilla fija. El resultado resume la distribucion espacial de
    orientaciones y frecuencias — la "forma general" de la escena — sin
    depender de ningun punto de interes concreto.
    """
    global _gabor_kernels
    kernels = _gabor_kernels if _gabor_kernels is not None else build_gabor_kernels(cfg)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Normalizacion de contraste local: quita el efecto de la exposicion.
    gray = (gray - gray.mean()) / max(float(gray.std()), EPS)

    grid = cfg["gist_grid"]
    h, w = gray.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)

    parts: list[np.ndarray] = []
    for kernel in kernels:
        response = np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel))
        cells = np.empty(grid * grid, dtype=np.float32)
        idx = 0
        for i in range(grid):
            for j in range(grid):
                cells[idx] = response[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].mean()
                idx += 1
        parts.append(cells)

    return l2_normalize(np.concatenate(parts))


DESCRIBERS = {
    "hog": describe_hog,
    "color": describe_color,
    "lbp": describe_lbp,
    "gist": describe_gist,
}


# --------------------------------------------------------------------------- #
# Paralelismo
# --------------------------------------------------------------------------- #


def _init_worker(cfg: dict) -> None:
    global _gabor_kernels, _config
    _config = cfg
    _gabor_kernels = build_gabor_kernels(cfg) if "gist" in cfg["descriptors"] else None
    # Evita que cada worker abra sus propios hilos de BLAS/OpenCV y se peleen
    # entre si: el paralelismo ya lo da el Pool.
    cv2.setNumThreads(1)


def _process_one(task: tuple[str, str, str]):
    dataset, image_id, path_str = task
    cfg = _config
    assert cfg is not None

    img = load_image(Path(path_str), cfg["image_size"])
    if img is None:
        return dataset, image_id, None, "no se pudo leer"

    vectors: dict[str, np.ndarray] = {}
    for name in cfg["descriptors"]:
        try:
            vec = DESCRIBERS[name](img, cfg)
        except Exception as exc:  # noqa: BLE001
            return dataset, image_id, None, f"{name} fallo: {exc}"

        if not np.all(np.isfinite(vec)):
            return dataset, image_id, None, f"{name} produjo valores no finitos"

        vectors[name] = vec.astype(np.float32)

    return dataset, image_id, vectors, None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument(
        "--descriptors", nargs="+", default=list(DESCRIPTOR_NAMES), choices=DESCRIPTOR_NAMES
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--limit", type=int, default=None, help="Procesa como maximo N imagenes por dataset."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--suffix", default="", help="Sufijo de salida (para barrer parametros).")

    g = parser.add_argument_group("parametros de los descriptores")
    g.add_argument("--image-size", type=int, default=128, help="Lado al que se reescalan las imagenes.")
    g.add_argument("--pyramid", type=int, default=2, help="Niveles de piramide espacial (1=sin piramide).")
    g.add_argument("--hog-orientations", type=int, default=9)
    g.add_argument("--hog-cell", type=int, default=16)
    g.add_argument("--hog-block", type=int, default=2)
    g.add_argument("--color-bins-h", type=int, default=8)
    g.add_argument("--color-bins-s", type=int, default=4)
    g.add_argument("--color-bins-v", type=int, default=4)
    g.add_argument("--lbp-points", type=int, default=8)
    g.add_argument("--lbp-radius", type=int, default=1)
    g.add_argument("--gist-scales", type=int, default=4)
    g.add_argument("--gist-orientations", type=int, default=8)
    g.add_argument("--gist-grid", type=int, default=4)
    g.add_argument("--gist-ksize", type=int, default=21)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    return {
        "descriptors": list(args.descriptors),
        "image_size": args.image_size,
        "pyramid": args.pyramid,
        "hog_orientations": args.hog_orientations,
        "hog_cell": args.hog_cell,
        "hog_block": args.hog_block,
        "color_bins_h": args.color_bins_h,
        "color_bins_s": args.color_bins_s,
        "color_bins_v": args.color_bins_v,
        "lbp_points": args.lbp_points,
        "lbp_radius": args.lbp_radius,
        "gist_scales": args.gist_scales,
        "gist_orientations": args.gist_orientations,
        "gist_grid": args.gist_grid,
        "gist_ksize": args.gist_ksize,
    }


def collect_tasks(data_dir: Path, datasets: list[str], limit: int | None):
    tasks: list[tuple[str, str, str]] = []
    for dataset in datasets:
        image_dir = data_dir / dataset
        if not image_dir.exists():
            print(f"  aviso: no existe {image_dir}, se omite")
            continue
        paths = sorted(image_dir.glob("*.jpg"))
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            tasks.append((dataset, path.stem, str(path)))
    return tasks


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.suffix}" if args.suffix else ""
    targets = {name: out_dir / f"{name}{tag}.npz" for name in args.descriptors}

    existentes = [p for p in targets.values() if p.exists()]
    if existentes and not args.overwrite:
        raise SystemExit(
            f"Ya existen {[p.name for p in existentes]}. Usa --overwrite o --suffix."
        )

    cfg = build_config(args)

    print("=== Extraccion de descriptores globales ===")
    print(f"  descriptores : {', '.join(args.descriptors)}")
    print(f"  imagen       : {args.image_size}x{args.image_size}, piramide={args.pyramid}")

    tasks = collect_tasks(data_dir, args.datasets, args.limit)
    if not tasks:
        raise SystemExit(
            f"No se encontraron imagenes en {data_dir}. "
            "Descarga el dataset con ./data/download_dataset.sh y ./data/unzip_dataset.sh"
        )
    print(f"  imagenes     : {len(tasks)}  |  workers: {args.workers}")
    print()

    image_ids: list[str] = []
    acumulado: dict[str, list[np.ndarray]] = {name: [] for name in args.descriptors}
    fallidas: list[tuple[str, str]] = []

    with Pool(processes=args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for dataset, image_id, vectors, error in tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=16),
            total=len(tasks),
            desc="Global",
            unit="img",
        ):
            if vectors is None:
                fallidas.append((f"{dataset}/{image_id}", error or "desconocido"))
                continue
            image_ids.append(f"{dataset}/{image_id}")
            for name, vec in vectors.items():
                acumulado[name].append(vec)

    if not image_ids:
        raise SystemExit("Ninguna imagen se pudo procesar.")

    # imap_unordered devuelve en orden arbitrario: se reordena para que la
    # salida sea deterministica y comparable entre corridas.
    orden = np.argsort(np.asarray(image_ids, dtype=object), kind="stable")
    ids_ordenados = np.asarray(image_ids, dtype=object)[orden]

    print()
    for name in args.descriptors:
        matrix = np.vstack(acumulado[name]).astype(np.float32)[orden]
        np.savez_compressed(targets[name], image_ids=ids_ordenados, features=matrix)
        print(f"  {name:<6} {matrix.shape[0]} x {matrix.shape[1]:<6} -> {targets[name]}")

    config_path = out_dir / f"extraction_config{tag}.json"
    config_path.write_text(
        json.dumps(
            {
                **cfg,
                "datasets": list(args.datasets),
                "num_images": len(ids_ordenados),
                "num_failed": len(fallidas),
                "dimensions": {
                    name: int(np.vstack(acumulado[name]).shape[1]) for name in args.descriptors
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if fallidas:
        print(f"\n  {len(fallidas)} imagenes omitidas:")
        for image_id, motivo in fallidas[:10]:
            print(f"    {image_id}: {motivo}")
        if len(fallidas) > 10:
            print(f"    ... y {len(fallidas) - 10} mas")
        print("  (son las corruptas conocidas del dataset; ver README)")

    print(f"\n  config -> {config_path}")
    print("\nSiguiente paso:")
    for name in args.descriptors:
        print(f"  python 4_evaluate_retrieval.py --embeddings {targets[name]} --name {name}")
    print(f"  python 3b_combine_global.py --descriptors {' '.join(args.descriptors)}")


if __name__ == "__main__":
    main()
