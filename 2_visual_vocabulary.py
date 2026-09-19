"""Vocabulario visual (Bag of Visual Words) a partir de los descriptores SIFT.

Toma `data/features/sift/sift_features.h5` (generado por
1_feature_extraction_sift.py, o bajado con data/download_sift_features.sh),
construye un vocabulario visual con K-means y representa cada imagen como un
histograma de palabras visuales de largo fijo K.

Salida (contrato que consume 4_evaluate_retrieval.py):
    data/features/vocabulary/image_histograms.npz
        image_ids  : (N,)    "<dataset>/<image_id>"
        histograms : (N, K)  float32
    data/features/vocabulary/visual_vocabulary.npz
        centroids  : (K, 128) float32  + la configuracion usada
    data/features/vocabulary/vocabulary_config.json

Uso tipico:
    python 2_visual_vocabulary.py --vocab-size 1000 --rootsift
    python 2_visual_vocabulary.py --vocab-size 20000 --rootsift --tfidf
    python 2_visual_vocabulary.py --query data/oxford/all_souls_000013.jpg --top-k 5

Despues:
    python 4_evaluate_retrieval.py \\
        --embeddings data/features/vocabulary/image_histograms.npz --name bovw


Notas de implementacion
-----------------------

**Memoria.** El HDF5 completo son ~30 millones de descriptores, o sea unos
15 GB en float32. Nunca se cargan todos: el muestreo para entrenar el
vocabulario se hace *durante* el recorrido del archivo, tomando como maximo
`--sample-per-image` descriptores de cada imagen. El pico de memoria queda
acotado por `--sample-size` (por defecto 1M x 128 x 4 B = 512 MB) y no depende
del tamano del dataset.

Muestrear por imagen ademas es mas correcto estadisticamente que muestrear
uniformemente sobre el conjunto de todos los descriptores: lo segundo sesga el
vocabulario hacia las imagenes texturadas, que aportan muchisimos mas keypoints.

**RootSIFT** (`--rootsift`). Normalizar el descriptor en L1 y sacarle raiz
cuadrada convierte la distancia euclidiana en la distancia de Hellinger sobre
el descriptor original. Son dos lineas y en la literatura de retrieval de
landmarks vale varios puntos de mAP. Se aplica tanto al entrenar como al
asignar, y queda registrado en el config para no mezclar representaciones.

**tf-idf** (`--tfidf`). Pondera cada palabra visual por log(N / df): las
palabras que aparecen en casi todas las imagenes no discriminan nada. Es la
adaptacion directa del modelo de recuperacion de texto en el que se inspira BoVW.

**MiniBatchKMeans** por defecto. Con K grande (que es donde este modelo
funciona bien) el K-means clasico con n_init=10 es inviable. `--full-kmeans`
fuerza el clasico si se quiere comparar en el reporte.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Falta scikit-learn. Instalalo con: pip install scikit-learn"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
SIFT_H5_PATH = DATA_DIR / "features" / "sift" / "sift_features.h5"
VOCAB_DIR = DATA_DIR / "features" / "vocabulary"
VOCAB_PATH = VOCAB_DIR / "visual_vocabulary.npz"
HISTOGRAMS_PATH = VOCAB_DIR / "image_histograms.npz"
CONFIG_PATH = VOCAB_DIR / "vocabulary_config.json"

DATASETS = ["oxford", "paris"]
SIFT_DIM = 128
EPS = 1e-12


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--h5", default=str(SIFT_H5_PATH), help="HDF5 con descriptores SIFT.")
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, choices=DATASETS,
        help="Subconjuntos a procesar.",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=1000,
        help=(
            "Numero de palabras visuales (clusters). Para retrieval de landmarks "
            "mas grande es mejor: la literatura usa decenas o cientos de miles. "
            "Default 1000 para una primera corrida rapida."
        ),
    )
    parser.add_argument(
        "--sample-size", type=int, default=1_000_000,
        help="Maximo de descriptores usados para entrenar K-means (acota la RAM).",
    )
    parser.add_argument(
        "--sample-per-image", type=int, default=200,
        help="Maximo de descriptores tomados de cada imagen para el muestreo.",
    )
    parser.add_argument(
        "--rootsift", action="store_true",
        help="Aplica RootSIFT (L1 + sqrt) a los descriptores. Recomendado.",
    )
    parser.add_argument(
        "--tfidf", action="store_true",
        help="Pondera las palabras visuales por idf = log(N / df).",
    )
    parser.add_argument(
        "--normalize", default="l1", choices=("none", "l1", "l2", "power"),
        help="Normalizacion final del histograma (default: l1).",
    )
    parser.add_argument(
        "--full-kmeans", action="store_true",
        help="Usa KMeans clasico en vez de MiniBatchKMeans (mucho mas lento).",
    )
    parser.add_argument("--batch-size", type=int, default=10_000, help="Batch de MiniBatchKMeans.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recalcula aunque ya existan vocabulario e histogramas.",
    )
    parser.add_argument("--out-dir", default=str(VOCAB_DIR))
    parser.add_argument(
        "--suffix", default="",
        help="Sufijo para los archivos de salida (util para barrer K sin pisar nada).",
    )
    parser.add_argument("--query", default=None, help="Imagen consulta para un retrieval rapido.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Descriptores
# --------------------------------------------------------------------------- #


def apply_rootsift(descriptors: np.ndarray) -> np.ndarray:
    """RootSIFT: normaliza en L1 y saca raiz cuadrada.

    Hace que la distancia euclidiana entre descriptores transformados equivalga
    a la distancia de Hellinger entre los originales, que se comporta mucho
    mejor para histogramas como SIFT.
    """
    if descriptors.size == 0:
        return descriptors
    norms = np.abs(descriptors).sum(axis=1, keepdims=True)
    return np.sqrt(descriptors / np.maximum(norms, EPS)).astype(np.float32)


def list_images(h5f: h5py.File, datasets: list[str]) -> list[tuple[str, str]]:
    """Lista (dataset, image_id) sin leer ningun descriptor."""
    entries: list[tuple[str, str]] = []
    for dataset in datasets:
        if dataset not in h5f:
            print(f"  aviso: '{dataset}' no esta en el HDF5, se omite")
            continue
        for image_id in sorted(h5f[dataset].keys()):
            entries.append((dataset, image_id))
    return entries


def read_descriptors(h5f: h5py.File, dataset: str, image_id: str, rootsift: bool) -> np.ndarray:
    group = h5f[dataset][image_id]
    if "descriptors" not in group:
        return np.empty((0, SIFT_DIM), dtype=np.float32)
    descriptors = np.asarray(group["descriptors"][()], dtype=np.float32)
    if descriptors.size == 0:
        return np.empty((0, SIFT_DIM), dtype=np.float32)
    return apply_rootsift(descriptors) if rootsift else descriptors


def sample_descriptors(
    h5_path: Path,
    datasets: list[str],
    sample_size: int,
    per_image: int,
    rootsift: bool,
    seed: int,
) -> np.ndarray:
    """Muestrea descriptores recorriendo el HDF5, sin cargarlo entero.

    El presupuesto total (`sample_size`) se reparte entre las imagenes, asi que
    el pico de memoria es ~sample_size x 128 x 4 bytes sin importar cuanto pese
    el archivo.
    """
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as h5f:
        entries = list_images(h5f, datasets)
        if not entries:
            raise ValueError(f"No hay imagenes en {h5_path} para {datasets}")

        budget = max(1, sample_size // len(entries))
        per_image = max(1, min(per_image, budget))

        print(
            f"  {len(entries)} imagenes; hasta {per_image} descriptores por imagen "
            f"(~{len(entries) * per_image:,} en total)"
        )

        chunks: list[np.ndarray] = []
        total = 0
        vacias = 0

        for dataset, image_id in tqdm(entries, desc="Muestreando", unit="img"):
            descriptors = read_descriptors(h5f, dataset, image_id, rootsift)
            if descriptors.shape[0] == 0:
                vacias += 1
                continue

            if descriptors.shape[0] > per_image:
                idx = rng.choice(descriptors.shape[0], size=per_image, replace=False)
                descriptors = descriptors[idx]

            chunks.append(descriptors)
            total += descriptors.shape[0]

            if total >= sample_size:
                break

    if not chunks:
        raise ValueError(f"No se encontraron descriptores validos en {h5_path}")

    X = np.vstack(chunks)
    if X.shape[0] > sample_size:
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X = X[idx]

    if vacias:
        print(f"  {vacias} imagenes sin descriptores (se omiten del muestreo)")
    print(f"  muestra final: {X.shape[0]:,} x {X.shape[1]} ({X.nbytes / 2**20:.0f} MB)")
    return np.ascontiguousarray(X, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Vocabulario
# --------------------------------------------------------------------------- #


def train_vocabulary(
    X: np.ndarray, vocab_size: int, full_kmeans: bool, batch_size: int, seed: int
) -> np.ndarray:
    if vocab_size > X.shape[0]:
        raise SystemExit(
            f"--vocab-size ({vocab_size}) no puede superar el numero de "
            f"descriptores muestreados ({X.shape[0]}). Sube --sample-size."
        )

    por_cluster = X.shape[0] / vocab_size
    if por_cluster < 20:
        print(
            f"  AVISO: solo ~{por_cluster:.0f} descriptores por cluster. "
            "Sube --sample-size para un vocabulario mas estable."
        )

    start = time.time()
    if full_kmeans:
        print(f"  KMeans clasico, K={vocab_size} (esto puede tardar muchisimo)")
        model = KMeans(n_clusters=vocab_size, n_init=3, random_state=seed, max_iter=300)
    else:
        print(f"  MiniBatchKMeans, K={vocab_size}, batch={batch_size}")
        model = MiniBatchKMeans(
            n_clusters=vocab_size,
            batch_size=batch_size,
            n_init=3,
            max_iter=200,
            max_no_improvement=50,
            random_state=seed,
        )

    model.fit(X)
    print(f"  entrenado en {time.time() - start:.1f} s")
    return np.ascontiguousarray(model.cluster_centers_, dtype=np.float32)


def assign_words(descriptors: np.ndarray, centroids: np.ndarray, centroid_sq: np.ndarray) -> np.ndarray:
    """Asigna cada descriptor a su centroide mas cercano (L2).

    Usa ||d - c||^2 = ||d||^2 - 2 d.c + ||c||^2 y descarta ||d||^2, que es
    constante por fila y no cambia el argmin. Se procesa por bloques para que
    la matriz intermedia (bloque x K) no se dispare con K grande.
    """
    if descriptors.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)

    k = centroids.shape[0]
    # Techo de ~64 MB para la matriz de distancias parcial.
    block = max(1, min(descriptors.shape[0], (64 << 20) // max(1, k * 4)))

    labels = np.empty(descriptors.shape[0], dtype=np.int32)
    for start in range(0, descriptors.shape[0], block):
        chunk = descriptors[start : start + block]
        scores = chunk @ centroids.T
        scores *= 2.0
        scores -= centroid_sq[None, :]
        labels[start : start + chunk.shape[0]] = np.argmax(scores, axis=1)

    return labels


def build_histograms(
    h5_path: Path,
    centroids: np.ndarray,
    datasets: list[str],
    rootsift: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Histograma de conteos crudos por imagen. Devuelve (image_ids, counts)."""
    k = centroids.shape[0]
    centroid_sq = np.einsum("ij,ij->i", centroids, centroids).astype(np.float32)

    with h5py.File(h5_path, "r") as h5f:
        entries = list_images(h5f, datasets)

        nbytes = len(entries) * k * 4
        print(f"  matriz de histogramas: {len(entries)} x {k} ({nbytes / 2**20:.0f} MB)")

        image_ids = np.empty(len(entries), dtype=object)
        counts = np.zeros((len(entries), k), dtype=np.float32)
        sin_descriptores = 0

        for i, (dataset, image_id) in enumerate(
            tqdm(entries, desc="Histogramas", unit="img")
        ):
            image_ids[i] = f"{dataset}/{image_id}"

            descriptors = read_descriptors(h5f, dataset, image_id, rootsift)
            if descriptors.shape[0] == 0:
                sin_descriptores += 1
                continue  # queda como histograma de ceros, a proposito

            labels = assign_words(descriptors, centroids, centroid_sq)
            counts[i] = np.bincount(labels, minlength=k).astype(np.float32)

    if sin_descriptores:
        print(
            f"  {sin_descriptores} imagenes sin descriptores quedan con histograma nulo "
            "(se mantienen en la matriz para no romper el indice)"
        )

    return image_ids, counts


def apply_tfidf(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pondera por idf = log(N / df). Devuelve (histogramas, idf)."""
    n_images = counts.shape[0]
    df = (counts > 0).sum(axis=0).astype(np.float32)
    idf = np.log(n_images / np.maximum(df, 1.0)).astype(np.float32)

    muertas = int((df == 0).sum())
    if muertas:
        print(f"  {muertas} palabras visuales no aparecen en ninguna imagen")

    return counts * idf[None, :], idf


def normalize_histograms(H: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return H
    if mode == "l1":
        return H / np.maximum(np.abs(H).sum(axis=1, keepdims=True), EPS)
    if mode == "l2":
        return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), EPS)
    # power: signed sqrt + L2
    Y = np.sign(H) * np.sqrt(np.abs(H))
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), EPS)


# --------------------------------------------------------------------------- #
# Retrieval rapido (para inspeccion; la evaluacion formal va en 4_...)
# --------------------------------------------------------------------------- #


def run_retrieval(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    import cv2

    vocab = np.load(paths["vocab"], allow_pickle=True)
    centroids = vocab["centroids"].astype(np.float32)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))

    data = np.load(paths["histograms"], allow_pickle=True)
    image_ids = data["image_ids"]
    histograms = data["histograms"].astype(np.float32)

    img = cv2.imread(args.query, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen consulta: {args.query}")

    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img, None)
    descriptors = (
        np.empty((0, SIFT_DIM), dtype=np.float32)
        if descriptors is None
        else descriptors.astype(np.float32)
    )
    if config.get("rootsift"):
        descriptors = apply_rootsift(descriptors)

    centroid_sq = np.einsum("ij,ij->i", centroids, centroids).astype(np.float32)
    labels = assign_words(descriptors, centroids, centroid_sq)
    hist = np.bincount(labels, minlength=centroids.shape[0]).astype(np.float32)[None, :]

    if config.get("tfidf") and "idf" in vocab:
        hist = hist * vocab["idf"][None, :]
    hist = normalize_histograms(hist, config.get("normalize", "l1"))

    a = hist / max(float(np.linalg.norm(hist)), EPS)
    b = histograms / np.maximum(np.linalg.norm(histograms, axis=1, keepdims=True), EPS)
    scores = (b @ a[0]).astype(np.float32)

    order = np.argsort(-scores)[: args.top_k]
    print(f"\nConsulta: {args.query}")
    print("Top resultados:")
    for rank_i, idx in enumerate(order, 1):
        print(f"{rank_i:>2}. {image_ids[idx]}  score={scores[idx]:.6f}")

    print("\nPara la evaluacion formal (mAP, P@k) usa 4_evaluate_retrieval.py")


# --------------------------------------------------------------------------- #


def output_paths(out_dir: Path, suffix: str) -> dict[str, Path]:
    tag = f"_{suffix}" if suffix else ""
    return {
        "vocab": out_dir / f"visual_vocabulary{tag}.npz",
        "histograms": out_dir / f"image_histograms{tag}.npz",
        "config": out_dir / f"vocabulary_config{tag}.json",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir, args.suffix)

    if args.query is not None:
        faltan = [p for p in ("vocab", "histograms", "config") if not paths[p].exists()]
        if faltan:
            raise SystemExit(
                "Aun no hay vocabulario entrenado. Corre el script sin --query primero."
            )
        run_retrieval(args, paths)
        return

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise SystemExit(
            f"No existe {h5_path}.\n"
            "Genera los features con 1_feature_extraction_sift.py, o bajalos con:\n"
            "  ./data/download_sift_features.sh"
        )

    if paths["histograms"].exists() and not args.overwrite:
        raise SystemExit(
            f"Ya existe {paths['histograms']}. Usa --overwrite, o --suffix para "
            "guardar en archivos aparte (util al barrer --vocab-size)."
        )

    print(f"HDF5: {h5_path} ({h5_path.stat().st_size / 2**30:.2f} GB)")
    print(f"K={args.vocab_size}  rootsift={args.rootsift}  tfidf={args.tfidf}  "
          f"normalize={args.normalize}")

    print("\n=== 1/3 Muestreo de descriptores ===")
    X = sample_descriptors(
        h5_path, args.datasets, args.sample_size, args.sample_per_image,
        args.rootsift, args.seed,
    )

    print("\n=== 2/3 Entrenamiento del vocabulario ===")
    centroids = train_vocabulary(
        X, args.vocab_size, args.full_kmeans, args.batch_size, args.seed
    )
    del X

    print("\n=== 3/3 Histogramas por imagen ===")
    image_ids, counts = build_histograms(h5_path, centroids, args.datasets, args.rootsift)

    idf = None
    H = counts
    if args.tfidf:
        print("  aplicando tf-idf")
        H, idf = apply_tfidf(counts)

    H = normalize_histograms(H, args.normalize).astype(np.float32)

    vacios = int((np.abs(H).sum(axis=1) == 0).sum())

    vocab_payload = {"centroids": centroids}
    if idf is not None:
        vocab_payload["idf"] = idf
    np.savez_compressed(paths["vocab"], **vocab_payload)
    np.savez_compressed(paths["histograms"], image_ids=image_ids, histograms=H)

    config = {
        "vocab_size": int(args.vocab_size),
        "sample_size": int(args.sample_size),
        "sample_per_image": int(args.sample_per_image),
        "rootsift": bool(args.rootsift),
        "tfidf": bool(args.tfidf),
        "normalize": args.normalize,
        "kmeans": "full" if args.full_kmeans else "minibatch",
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "datasets": list(args.datasets),
        "num_images": int(len(image_ids)),
        "num_empty_histograms": vacios,
        "source_h5": str(h5_path),
    }
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Vocabulario  : {paths['vocab']}")
    print(f"Histogramas  : {paths['histograms']}  ({H.shape[0]} x {H.shape[1]})")
    print(f"Config       : {paths['config']}")
    if vacios:
        print(f"Histogramas nulos: {vacios}")
    print("=" * 60)
    print("\nSiguiente paso:")
    print(f"  python 4_evaluate_retrieval.py --embeddings {paths['histograms']} \\")
    print(f"      --name bovw_k{args.vocab_size}")


if __name__ == "__main__":
    main()
