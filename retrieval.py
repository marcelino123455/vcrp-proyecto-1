"""Utilidades de retrieval: carga de embeddings, normalizaciones y metricas.

Este modulo es agnostico al descriptor. Cualquier representacion del proyecto
(BoVW, HOG, GIST, color, LBP, VLAD, fusion) se evalua pasando por aqui, de modo
que la unica variable entre experimentos sea el descriptor y no el codigo de
ranking.

Contrato de embeddings
----------------------
Un archivo de embeddings es un `.npz` (o `.h5`) que contiene:

    image_ids : array de strings, formato "<dataset>/<image_id>"
                p.ej. "oxford/all_souls_000013"
    features  : array (N, D) float32, misma fila que image_ids

Por compatibilidad se aceptan tambien los nombres `ids`, `histograms`,
`embeddings` y `descriptors` para la matriz de features, que es lo que ya
genera `2_visual_vocabulary.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EPS = 1e-12

_ID_KEYS = ("image_ids", "ids", "names", "image_id")
_FEATURE_KEYS = ("features", "histograms", "embeddings", "descriptors", "X")

NORMALIZATIONS = ("none", "l1", "l2", "power", "hellinger")
METRICS = ("cosine", "dot", "l2", "l1", "chi2", "intersection", "hellinger")


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #


def _pick(container, keys: tuple[str, ...], what: str, available):
    for key in keys:
        if key in container:
            return container[key]
    raise KeyError(
        f"No se encontro la clave de {what} en el archivo de embeddings. "
        f"Se buscaron {keys}, disponibles: {sorted(available)}"
    )


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga un archivo de embeddings y devuelve (image_ids, features).

    `image_ids` es un array de strings de largo N, `features` es (N, D) float32.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de embeddings: {path}")

    if path.suffix in (".npz", ".npy"):
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys()) if hasattr(data, "keys") else []
        ids = _pick(data, _ID_KEYS, "image_ids", keys)
        features = _pick(data, _FEATURE_KEYS, "features", keys)
    elif path.suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as h5f:
            keys = list(h5f.keys())
            ids = _pick(h5f, _ID_KEYS, "image_ids", keys)[()]
            features = _pick(h5f, _FEATURE_KEYS, "features", keys)[()]
    else:
        raise ValueError(f"Extension no soportada: {path.suffix} (usa .npz o .h5)")

    ids = np.asarray([_as_str(x) for x in np.asarray(ids).ravel()], dtype=object)
    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"features debe ser 2D (N, D), se recibio {features.shape}")
    if len(ids) != features.shape[0]:
        raise ValueError(
            f"image_ids ({len(ids)}) y features ({features.shape[0]}) no coinciden en largo"
        )

    return ids, features


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def split_ids(image_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separa "<dataset>/<image_id>" en dos arrays (datasets, names).

    Si un id viene sin prefijo de dataset, se infiere: los nombres de Paris
    empiezan con "paris_"; el resto se asume Oxford.
    """
    datasets = []
    names = []
    for raw in image_ids:
        text = _as_str(raw)
        if "/" in text:
            dataset, name = text.split("/", 1)
        else:
            name = text
            dataset = "paris" if name.startswith("paris_") else "oxford"
        datasets.append(dataset)
        names.append(name)
    return np.asarray(datasets, dtype=object), np.asarray(names, dtype=object)


# --------------------------------------------------------------------------- #
# Normalizacion
# --------------------------------------------------------------------------- #


def normalize_features(X: np.ndarray, mode: str = "l2") -> np.ndarray:
    """Normaliza las filas de X.

    Modos:
        none      : sin cambios.
        l1        : cada fila suma 1. Lo natural para histogramas de conteo.
        l2        : cada fila tiene norma 1. Convierte el producto punto en coseno.
        power     : signed square root (x -> sign(x)*sqrt(|x|)) + L2. Reduce el
                    efecto de "burstiness" (palabras visuales que se repiten
                    muchisimo en una sola imagen y dominan el histograma).
        hellinger : L1 + sqrt + L2. El producto punto resultante equivale al
                    coeficiente de Bhattacharyya, es decir, permite usar una
                    metrica tipo chi-cuadrado con el costo de un producto punto.
    """
    if mode not in NORMALIZATIONS:
        raise ValueError(f"normalizacion desconocida: {mode!r} (opciones: {NORMALIZATIONS})")

    X = np.asarray(X, dtype=np.float32)

    if mode == "none":
        return X

    if mode == "l1":
        return _l1(X)

    if mode == "l2":
        return _l2(X)

    if mode == "power":
        Y = np.sign(X) * np.sqrt(np.abs(X))
        return _l2(Y)

    # hellinger
    Y = _l1(np.abs(X))
    Y = np.sqrt(Y)
    return _l2(Y)


def _l1(X: np.ndarray) -> np.ndarray:
    norms = np.abs(X).sum(axis=1, keepdims=True)
    return X / np.maximum(norms, EPS)


def _l2(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, EPS)


# --------------------------------------------------------------------------- #
# Similitud
# --------------------------------------------------------------------------- #


def similarity(
    Q: np.ndarray,
    X: np.ndarray,
    metric: str = "cosine",
    chunk_size: int = 2048,
) -> np.ndarray:
    """Matriz de similitud (nq, n). Siempre "mayor = mas parecido".

    Las distancias se devuelven negadas para que el criterio de orden sea
    uniforme y `argsort(-sim)` funcione con cualquier metrica.
    """
    if metric not in METRICS:
        raise ValueError(f"metrica desconocida: {metric!r} (opciones: {METRICS})")

    Q = np.atleast_2d(np.asarray(Q, dtype=np.float32))
    X = np.asarray(X, dtype=np.float32)

    if metric == "cosine":
        return _l2(Q) @ _l2(X).T

    if metric == "dot":
        return Q @ X.T

    if metric == "hellinger":
        Qh = _l2(np.sqrt(_l1(np.abs(Q))))
        Xh = _l2(np.sqrt(_l1(np.abs(X))))
        return Qh @ Xh.T

    if metric == "l2":
        # ||q - x||^2 = ||q||^2 - 2 q.x + ||x||^2  (evita materializar N x D x nq)
        q_sq = (Q * Q).sum(axis=1, keepdims=True)
        x_sq = (X * X).sum(axis=1, keepdims=True).T
        d2 = np.maximum(q_sq - 2.0 * (Q @ X.T) + x_sq, 0.0)
        return -np.sqrt(d2)

    # l1, chi2 e intersection no se factorizan como producto de matrices.
    # Se calculan query por query y en bloques de la base, para que el tensor
    # intermedio sea (chunk_size, D) y no (nq, n, D) — que para D grande
    # (HOG con piramide facilmente pasa de 8000 dims) serian decenas de GB.
    n, d = X.shape
    chunk_size = max(1, min(chunk_size, max(1, (64 << 20) // max(1, d * 4))))

    out = np.empty((Q.shape[0], n), dtype=np.float32)
    for qi in range(Q.shape[0]):
        q = Q[qi]  # (D,)
        for start in range(0, n, chunk_size):
            block = X[start : start + chunk_size]  # (b, D)
            diff = q[None, :] - block

            if metric == "l1":
                values = -np.abs(diff).sum(axis=1)
            elif metric == "chi2":
                summed = q[None, :] + block
                values = -0.5 * ((diff * diff) / np.maximum(summed, EPS)).sum(axis=1)
            else:  # intersection
                values = np.minimum(q[None, :], block).sum(axis=1)

            out[qi, start : start + block.shape[0]] = values

    return out


def rank(sim_row: np.ndarray, names: np.ndarray) -> list[str]:
    """Convierte una fila de similitudes en una lista de nombres ordenada."""
    order = np.argsort(-sim_row, kind="stable")
    return [str(names[i]) for i in order]


# --------------------------------------------------------------------------- #
# Fusion tardia
# --------------------------------------------------------------------------- #


def rank_normalize(sim: np.ndarray) -> np.ndarray:
    """Normaliza similitudes a [0, 1] por query (min-max).

    Necesario antes de fusionar scores de representaciones distintas: un
    coseno de BoVW y una distancia chi2 de HOG no viven en la misma escala,
    y sumarlas crudas hace que una domine arbitrariamente.
    """
    sim = np.atleast_2d(np.asarray(sim, dtype=np.float32))
    lo = sim.min(axis=1, keepdims=True)
    hi = sim.max(axis=1, keepdims=True)
    return (sim - lo) / np.maximum(hi - lo, EPS)


def fuse(sims: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Fusion tardia por suma ponderada de scores normalizados.

    Cada matriz de `sims` debe tener la misma forma (nq, n) y las filas y
    columnas deben referirse a las mismas queries e imagenes, en el mismo orden.
    """
    if not sims:
        raise ValueError("se requiere al menos una matriz de similitud")

    shapes = {s.shape for s in sims}
    if len(shapes) != 1:
        raise ValueError(f"las matrices de similitud no tienen la misma forma: {shapes}")

    if weights is None:
        weights = [1.0 / len(sims)] * len(sims)
    if len(weights) != len(sims):
        raise ValueError("weights debe tener el mismo largo que sims")

    total = np.zeros_like(np.atleast_2d(sims[0]), dtype=np.float32)
    for sim, weight in zip(sims, weights):
        total += float(weight) * rank_normalize(sim)
    return total
