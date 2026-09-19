"""VLAD — Vector of Locally Aggregated Descriptors.

Representacion GLOBAL de largo fijo construida agregando descriptores locales.
Es la pieza que faltaba entre los dos extremos que teniamos: BoVW (local,
cuenta palabras) y HOG/GIST/color (global, sobre pixeles).

Referencias
-----------
[1] H. Jegou, M. Douze, C. Schmid, P. Perez. "Aggregating local descriptors
    into a compact image representation". CVPR 2010.
[2] R. Arandjelovic, A. Zisserman. "All about VLAD". CVPR 2013.
    -> intra-normalizacion, adaptacion de vocabulario, MultiVLAD.
[3] R. Arandjelovic, A. Zisserman. "Three things everyone should know to
    improve object retrieval". CVPR 2012.  -> RootSIFT.

Que es
------
Se entrena un vocabulario GRUESO (K pequeno: 64, 128, 256 — no miles como en
BoVW). Para cada imagen, cada descriptor se asigna a su centroide mas cercano,
y en vez de contar cuantos cayeron en cada celda, se acumula el RESIDUO:

    v_k = sum_{x : NN(x) = c_k} (x - c_k)

El descriptor final es la concatenacion de los K residuos, de dimension K x 128.

La diferencia con BoVW es la clave: BoVW guarda *cuantos* descriptores cayeron
en cada celda y tira la posicion dentro de la celda. VLAD guarda *hacia donde*
se desvian, o sea conserva informacion de primer orden sobre la distribucion
dentro de cada celda. Por eso con K=64 VLAD compite con un BoVW de miles de
palabras: cada celda carga 128 numeros en vez de uno.

(Fisher Vectors, que el enunciado tambien menciona, extienden esta idea a
estadisticos de segundo orden sobre un GMM. VLAD es su version dura y
simplificada, y en la practica rinde parecido con mucho menos codigo.)

Normalizaciones
---------------
Es donde se juega buena parte del rendimiento, y el orden importa:

1. mass norm (`--mass-norm`): dividir cada v_k por el numero de descriptores
   asignados a esa celda. Evita que una celda muy poblada domine.
2. signed square root (`--power`): x -> sign(x)*sqrt(|x|). Atenua los
   componentes grandes.
3. intra-normalizacion (`--intra-norm`, activa por defecto): L2-normalizar
   CADA bloque v_k por separado antes de concatenar. Es la contribucion
   principal de [2]. Ataca la "burstiness": si una textura repetitiva de la
   imagen (una reja, una fila de ventanas) dispara muchisimas veces la misma
   palabra visual, ese bloque crece sin limite y domina la distancia,
   aunque no sea lo que distingue al edificio. Normalizar por bloque le da a
   cada palabra visual el mismo peso pase lo que pase.
4. L2 global: siempre al final, para que el producto punto sea coseno.

Salida (contrato de 4_evaluate_retrieval.py)
---------------------------------------------
    data/features/vlad/vlad_k<K>.npz
        image_ids : (N,)    "<dataset>/<image_id>"
        features  : (N, D)  float32

Uso
---
    python 3c_vlad.py --vocab-size 64
    python 3c_vlad.py --vocab-size 128 --pca 512 --whiten-power 0.25
    python 3c_vlad.py --vocab-size 64 --no-intra-norm     # ablacion

Despues:
    python 4_evaluate_retrieval.py --embeddings data/features/vlad/vlad_k64.npz \\
        --name vlad_k64 --normalize none --metric cosine
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Falta scikit-learn. Instalalo con: pip install scikit-learn") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
SIFT_H5_PATH = DATA_DIR / "features" / "sift" / "sift_features.h5"
OUTPUT_DIR = DATA_DIR / "features" / "vlad"

DATASETS = ["oxford", "paris"]
SIFT_DIM = 128
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Lectura del HDF5
# --------------------------------------------------------------------------- #


def apply_rootsift(descriptors: np.ndarray) -> np.ndarray:
    """RootSIFT [3]: L1 + raiz cuadrada.

    Convierte la distancia euclidiana entre descriptores transformados en la
    distancia de Hellinger entre los originales, que se comporta mucho mejor
    para histogramas como SIFT. Es de las mejoras mas baratas que existen.
    """
    if descriptors.size == 0:
        return descriptors
    norms = np.abs(descriptors).sum(axis=1, keepdims=True)
    return np.sqrt(descriptors / np.maximum(norms, EPS)).astype(np.float32)


def list_images(h5f: h5py.File, datasets: list[str]) -> list[tuple[str, str]]:
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
    h5_path: Path, datasets: list[str], sample_size: int, per_image: int,
    rootsift: bool, seed: int,
) -> np.ndarray:
    """Muestreo acotado en memoria: nunca carga el HDF5 completo."""
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as h5f:
        entries = list_images(h5f, datasets)
        if not entries:
            raise SystemExit(f"No hay imagenes en {h5_path} para {datasets}")

        budget = max(1, sample_size // len(entries))
        per_image = max(1, min(per_image, budget))
        print(f"  {len(entries)} imagenes; hasta {per_image} descriptores por imagen")

        chunks: list[np.ndarray] = []
        total = 0
        for dataset, image_id in tqdm(entries, desc="Muestreando", unit="img"):
            descriptors = read_descriptors(h5f, dataset, image_id, rootsift)
            if descriptors.shape[0] == 0:
                continue
            if descriptors.shape[0] > per_image:
                idx = rng.choice(descriptors.shape[0], size=per_image, replace=False)
                descriptors = descriptors[idx]
            chunks.append(descriptors)
            total += descriptors.shape[0]
            if total >= sample_size:
                break

    X = np.vstack(chunks)
    if X.shape[0] > sample_size:
        X = X[rng.choice(X.shape[0], size=sample_size, replace=False)]
    print(f"  muestra: {X.shape[0]:,} x {X.shape[1]} ({X.nbytes / 2**20:.0f} MB)")
    return np.ascontiguousarray(X, dtype=np.float32)


# --------------------------------------------------------------------------- #
# VLAD
# --------------------------------------------------------------------------- #


def assign(descriptors: np.ndarray, centroids: np.ndarray, centroid_sq: np.ndarray) -> np.ndarray:
    """Centroide mas cercano por descriptor (argmin L2, via argmax del score)."""
    if descriptors.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)
    scores = 2.0 * (descriptors @ centroids.T) - centroid_sq[None, :]
    return np.argmax(scores, axis=1).astype(np.int32)


def vlad_encode(
    descriptors: np.ndarray,
    centroids: np.ndarray,
    centroid_sq: np.ndarray,
    *,
    mass_norm: bool,
    power: bool,
    intra_norm: bool,
) -> np.ndarray:
    """Codifica una imagen como vector VLAD de K*128 dimensiones."""
    k, dim = centroids.shape
    V = np.zeros((k, dim), dtype=np.float32)

    if descriptors.shape[0] == 0:
        return V.ravel()  # imagen sin keypoints -> vector nulo

    labels = assign(descriptors, centroids, centroid_sq)
    counts = np.bincount(labels, minlength=k)

    for cluster in np.flatnonzero(counts):
        miembros = descriptors[labels == cluster]
        # suma de (x - c) = suma(x) - n*c
        V[cluster] = miembros.sum(axis=0) - counts[cluster] * centroids[cluster]

    # 1) mass norm: quita el efecto de cuantos descriptores cayeron en la celda
    if mass_norm:
        divisor = np.maximum(counts, 1).astype(np.float32)[:, None]
        V = V / divisor

    # 2) signed square root
    if power:
        V = np.sign(V) * np.sqrt(np.abs(V))

    # 3) intra-normalizacion: L2 por bloque (la contribucion de "All about VLAD")
    if intra_norm:
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.maximum(norms, EPS)

    flat = V.ravel()

    # 4) L2 global
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > EPS else flat


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h5", default=str(SIFT_H5_PATH))
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--vocab-size", type=int, default=64,
        help="K del vocabulario GRUESO. Tipico 64/128/256. La dimension final es K*128.",
    )
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--sample-per-image", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--rootsift", action=argparse.BooleanOptionalAction, default=True,
        help="Aplica RootSIFT a los descriptores (activo por defecto).",
    )
    parser.add_argument(
        "--intra-norm", action=argparse.BooleanOptionalAction, default=True,
        help="Intra-normalizacion por bloque (activa por defecto).",
    )
    parser.add_argument("--mass-norm", action="store_true", help="Divide cada bloque por su conteo.")
    parser.add_argument("--power", action="store_true", help="Signed square root antes de normalizar.")
    parser.add_argument("--pca", type=int, default=0, help="Si > 0, reduce a esa dimension.")
    parser.add_argument("--whiten-power", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.suffix or f"k{args.vocab_size}"
    out_path = out_dir / f"vlad_{tag}.npz"

    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Ya existe {out_path}. Usa --overwrite o --suffix.")

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise SystemExit(
            f"No existe {h5_path}.\nBajalo con ./data/download_sift_features.sh"
        )

    dim_final = args.vocab_size * SIFT_DIM
    print("=== VLAD ===")
    print(f"  K={args.vocab_size} -> {dim_final} dimensiones")
    print(f"  rootsift={args.rootsift} intra_norm={args.intra_norm} "
          f"mass_norm={args.mass_norm} power={args.power}")

    print("\n=== 1/3 Muestreo ===")
    X = sample_descriptors(
        h5_path, args.datasets, args.sample_size, args.sample_per_image,
        args.rootsift, args.seed,
    )

    print("\n=== 2/3 Vocabulario grueso ===")
    start = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=args.vocab_size, batch_size=args.batch_size,
        n_init=5, max_iter=300, max_no_improvement=50, random_state=args.seed,
    )
    kmeans.fit(X)
    centroids = np.ascontiguousarray(kmeans.cluster_centers_, dtype=np.float32)
    centroid_sq = np.einsum("ij,ij->i", centroids, centroids).astype(np.float32)
    print(f"  entrenado en {time.time() - start:.1f} s")
    del X

    print("\n=== 3/3 Codificacion ===")
    with h5py.File(h5_path, "r") as h5f:
        entries = list_images(h5f, args.datasets)

        estimado = len(entries) * dim_final * 4 / 2**20
        print(f"  matriz: {len(entries)} x {dim_final} ({estimado:.0f} MB)")
        if estimado > 4096:
            print("  AVISO: mas de 4 GB. Considera bajar --vocab-size o usar --pca.")

        image_ids = np.empty(len(entries), dtype=object)
        features = np.zeros((len(entries), dim_final), dtype=np.float32)
        vacias = 0

        for i, (dataset, image_id) in enumerate(
            tqdm(entries, desc="VLAD", unit="img")
        ):
            image_ids[i] = f"{dataset}/{image_id}"
            descriptors = read_descriptors(h5f, dataset, image_id, args.rootsift)
            if descriptors.shape[0] == 0:
                vacias += 1
                continue
            features[i] = vlad_encode(
                descriptors, centroids, centroid_sq,
                mass_norm=args.mass_norm, power=args.power, intra_norm=args.intra_norm,
            )

    if vacias:
        print(f"  {vacias} imagenes sin descriptores quedan con vector nulo")

    pca_info = None
    # Modelo: lo que hace falta para codificar una imagen NUEVA con este mismo
    # vocabulario, sin re-entrenar nada. Se usa, por ejemplo, para codificar
    # consultas recortadas al bounding box (ver 6_bbox_queries.py).
    modelo: dict[str, np.ndarray] = {"centroids": centroids}

    if args.pca > 0:
        from sklearn.decomposition import PCA

        n_components = min(args.pca, features.shape[0], features.shape[1])
        power = float(np.clip(args.whiten_power, 0.0, 0.5))

        pca = PCA(n_components=n_components, whiten=False, random_state=args.seed)
        pca.fit(features)
        features = pca.transform(features).astype(np.float32)
        if power > 0:
            escala = np.power(np.maximum(pca.explained_variance_, EPS), power).astype(np.float32)
            features = features / escala[None, :]

        modelo.update(
            pca_mean=pca.mean_.astype(np.float32),
            pca_components=pca.components_.astype(np.float32),
            pca_variance=pca.explained_variance_.astype(np.float32),
            whiten_power=np.float32(power),
        )

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = (features / np.maximum(norms, EPS)).astype(np.float32)

        pca_info = {
            "components": int(n_components),
            "whiten_power": power,
            "explained_variance_ratio": round(float(pca.explained_variance_ratio_.sum()), 4),
        }
        print(f"  PCA -> {features.shape[1]} dims "
              f"({pca_info['explained_variance_ratio']:.1%} de la varianza)")

    if not np.all(np.isfinite(features)):
        raise SystemExit("VLAD produjo valores no finitos; revisa las normalizaciones.")

    np.savez_compressed(out_path, image_ids=image_ids, features=features)

    model_path = out_dir / f"vlad_{tag}_model.npz"
    np.savez_compressed(model_path, **modelo)
    print(f"  modelo: {model_path} ({model_path.stat().st_size / 2**20:.0f} MB)")

    config = {
        "vocab_size": int(args.vocab_size),
        "dimensions": int(features.shape[1]),
        "rootsift": bool(args.rootsift),
        "intra_norm": bool(args.intra_norm),
        "mass_norm": bool(args.mass_norm),
        "power": bool(args.power),
        "pca": pca_info,
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "num_images": int(len(image_ids)),
        "num_empty": int(vacias),
        "datasets": list(args.datasets),
    }
    (out_dir / f"vlad_{tag}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\n  salida: {out_path}  ({features.shape[0]} x {features.shape[1]})")
    print("\nSiguiente paso:")
    print(f"  python 4_evaluate_retrieval.py --embeddings {out_path} \\")
    print(f"      --name vlad_{tag} --normalize none --metric cosine")
    print("  (VLAD ya sale L2-normalizado; --normalize none evita normalizar dos veces)")


if __name__ == "__main__":
    main()
