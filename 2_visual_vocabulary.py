"""Visual Vocabulary (Bag of Visual Words) para retrieval de imágenes.

Este script toma los descriptores SIFT generados por 1_feature_extraction_sift.py,
construye un vocabulario visual con K-means y genera histogramas BoVW para cada
imagen de Oxford/Paris.

La pipeline implementada es:
    1) cargar los descriptores SIFT del HDF5
    2) muestrear un subconjunto para entrenar el vocabulario
    3) calcular centroides con K-means
    4) representar cada imagen por un histograma de palabras visuales
    5) opcionalmente realizar retrieval por similitud frente a la base

Uso esperado:
    python 2_visual_vocabulary.py --vocab-size 500
    python 2_visual_vocabulary.py --vocab-size 1000 --sample-size 20000
    python 2_visual_vocabulary.py --query data/oxford/oxford_000001.jpg --top-k 5
"""

import argparse
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError as exc:  # pragma: no cover - manejo de uso
    raise SystemExit(
        "Falta scikit-learn. Instálalo con: pip install scikit-learn"
    ) from exc

DATA_DIR = Path(__file__).resolve().parent / "data"
SIFT_H5_PATH = DATA_DIR / "features" / "sift" / "sift_features.h5"
VOCAB_DIR = DATA_DIR / "features" / "vocabulary"
VOCAB_PATH = VOCAB_DIR / "visual_vocabulary.pkl"
HISTOGRAMS_PATH = VOCAB_DIR / "image_histograms.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=str,
        default=str(SIFT_H5_PATH),
        help="Ruta al archivo HDF5 con descriptores SIFT.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Tamaño del vocabulario visual (número de clusters).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Máximo de descriptores usados para entrenar K-means.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Cantidad de resultados devueltos por retrieval.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ruta de una imagen consulta para ejecutar retrieval.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Si se pasa, vuelve a calcular vocabulario e histogramas aunque ya existan.",
    )
    return parser.parse_args()


def collect_descriptors(h5_path: Path, max_descriptors: int = 50000):
    """Recoge todos los descriptores SIFT del HDF5 en una sola matriz.

    Devuelve:
        X: array (N, 128)
        image_list: lista de strings con formato 'dataset/image_id'
    """
    descriptors_list = []
    image_list = []

    if not h5_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo HDF5 de SIFT: {h5_path}. "
            "Primero ejecuta 1_feature_extraction_sift.py"
        )

    with h5py.File(h5_path, "r") as h5f:
        for dataset_name in sorted(h5f.keys()):
            dataset_group = h5f[dataset_name]
            for image_id in sorted(dataset_group.keys()):
                image_group = dataset_group[image_id]
                if "descriptors" not in image_group:
                    continue

                descriptors = image_group["descriptors"][()]
                if descriptors.size == 0:
                    continue

                descriptors_list.append(descriptors.astype(np.float32))
                image_list.append(f"{dataset_name}/{image_id}")

    if not descriptors_list:
        raise ValueError(f"No se encontraron descriptores SIFT válidos en {h5_path}")

    X = np.vstack(descriptors_list)
    if X.shape[0] > max_descriptors:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_descriptors, replace=False)
        X = X[idx]

    return X, image_list


def train_visual_vocabulary(descriptors: np.ndarray, vocab_size: int) -> KMeans:
    """Entrena K-means sobre los descriptores para construir el vocabulario visual."""
    model = KMeans(
        n_clusters=vocab_size,
        n_init=10,
        random_state=42,
        max_iter=300,
    )
    model.fit(descriptors)
    return model


def descriptor_histogram(descriptors: np.ndarray, model: KMeans) -> np.ndarray:
    """Asigna cada descriptor al cluster más cercano y devuelve un histograma."""
    if descriptors.size == 0:
        return np.zeros(model.n_clusters, dtype=np.float32)

    labels = model.predict(descriptors)
    hist = np.bincount(labels, minlength=model.n_clusters).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def build_histograms_from_h5(h5_path: Path, model: KMeans, image_ids: list[str]):
    """Genera el histograma BoVW para cada imagen del HDF5."""
    histograms = []
    valid_ids = []

    with h5py.File(h5_path, "r") as h5f:
        for dataset_name in sorted(h5f.keys()):
            dataset_group = h5f[dataset_name]
            for image_id in sorted(dataset_group.keys()):
                image_group = dataset_group[image_id]
                if "descriptors" not in image_group:
                    continue

                descriptors = image_group["descriptors"][()].astype(np.float32)
                hist = descriptor_histogram(descriptors, model)
                histograms.append(hist)
                valid_ids.append(f"{dataset_name}/{image_id}")

    if not histograms:
        raise ValueError(f"No se pudo construir ningún histograma desde {h5_path}")

    # Mantener el mismo orden que se usa en la lista de imágenes y permitir una
    # recuperación directa cuando se guarda en disco.
    if len(valid_ids) != len(image_ids):
        # Si la lista externa ya se generó a partir del mismo recorrido, esto se
        # usa como validación simple. No aborta para mantener compatibilidad.
        pass

    return np.vstack(histograms), np.array(valid_ids, dtype=object)


def save_model_and_histograms(model: KMeans, image_ids: np.ndarray, histograms: np.ndarray):
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(model, f)

    np.savez_compressed(
        HISTOGRAMS_PATH,
        image_ids=image_ids,
        histograms=histograms.astype(np.float32),
    )

    print(f"Vocabulario guardado en: {VOCAB_PATH}")
    print(f"Histogramas guardadas en: {HISTOGRAMS_PATH}")


def load_model_and_histograms():
    with open(VOCAB_PATH, "rb") as f:
        model = pickle.load(f)

    data = np.load(HISTOGRAMS_PATH, allow_pickle=True)
    image_ids = data["image_ids"]
    histograms = data["histograms"].astype(np.float32)
    return model, image_ids, histograms


def compute_cosine_similarity(query_hist: np.ndarray, histograms: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_hist)
    hist_norms = np.linalg.norm(histograms, axis=1)
    similarities = np.divide(
        histograms.dot(query_hist),
        np.clip(hist_norms * query_norm, a_min=1e-12, a_max=None),
        out=np.zeros_like(hist_norms, dtype=np.float32),
        where=(hist_norms * query_norm) > 0,
    )
    return similarities.astype(np.float32)


def query_image_histogram(image_path: str, model: KMeans):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen consulta: {image_path}")

    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    else:
        descriptors = descriptors.astype(np.float32)

    return descriptor_histogram(descriptors, model)


def run_retrieval(query_image: str, top_k: int):
    model, image_ids, histograms = load_model_and_histograms()
    query_hist = query_image_histogram(query_image, model)
    similarities = compute_cosine_similarity(query_hist, histograms)
    ranked_idx = np.argsort(similarities)[::-1]

    print(f"\nConsulta: {query_image}")
    print("Top resultados:")
    for rank, idx in enumerate(ranked_idx[:top_k], 1):
        print(f"{rank:>2}. {image_ids[idx]}  score={similarities[idx]:.6f}")


def main() -> None:
    args = parse_args()
    h5_path = Path(args.h5)
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)

    if args.query is not None:
        if not (VOCAB_PATH.exists() and HISTOGRAMS_PATH.exists()):
            raise FileNotFoundError(
                "Aún no existe vocabulario entrenado. "
                "Primero ejecuta el script sin --query para construirlo."
            )
        run_retrieval(args.query, args.top_k)
        return

    descriptors, image_list = collect_descriptors(h5_path, max_descriptors=args.sample_size)
    model = train_visual_vocabulary(descriptors, vocab_size=args.vocab_size)
    histograms, image_ids = build_histograms_from_h5(h5_path, model, image_list)
    save_model_and_histograms(model, image_ids, histograms)

    print(f"\nVocabulario visual construido con K={args.vocab_size} clusters.")
    print(f"Se procesaron {len(image_ids)} imágenes en la base.")
    print("Para ejecutar retrieval, usa: python 2_visual_vocabulary.py --query <ruta_a_imagen> --top-k 5")


if __name__ == "__main__":
    main()
