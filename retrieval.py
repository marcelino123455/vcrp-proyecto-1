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
# Query expansion
# --------------------------------------------------------------------------- #

QE_COMPATIBLE_METRICS = ("cosine", "dot")


def alpha_query_expansion(
    Q: np.ndarray,
    X: np.ndarray,
    sims: np.ndarray,
    *,
    n_qe: int = 10,
    alpha: float = 3.0,
    include_query: bool = True,
) -> np.ndarray:
    """alpha-weighted Query Expansion (alphaQE).

    Idea, de Chum et al. "Total Recall" (ICCV 2007): la imagen consulta es UNA
    vista del objeto. Los primeros resultados, si son correctos, son otras
    vistas del mismo objeto. Promediarlos produce una consulta mas rica, que
    recupera imagenes que la original no alcanzaba — sobre todo las tomadas
    desde angulos muy distintos, que son justo las que mas cuestan.

    La version alpha (Radenovic et al., "Fine-tuning CNN Image Retrieval with
    No Human Annotation") pesa cada vecino por su similitud elevada a alpha:

        q' = normalizar( q + sum_i (q . x_i)^alpha * x_i )

    Con alpha = 0 todos los vecinos del top-n pesan igual y se recupera la
    Average QE clasica. Subir alpha concentra el peso en los vecinos de los que
    el sistema esta mas seguro, lo que hace el metodo mucho menos sensible a
    cuantos vecinos se tomen: un falso positivo en el puesto 8 entra con peso
    casi nulo en vez de arruinar la consulta. Los autores usan alpha = 3.

    Importante para el reporte: esto NO usa el ground truth en ningun momento,
    solo los vecinos que el propio sistema devuelve. Es una tecnica legitima de
    retrieval, no una fuga de informacion.

    Limitacion honesta: la QE clasica aplica verificacion espacial a los
    vecinos antes de promediarlos, para no expandir con falsos positivos. Aqui
    se usa el top-n crudo, asi que si la busqueda inicial es mala la expansion
    puede empeorarla ("query drift"). Por eso conviene barrer n_qe y reportarlo.

    Args:
        Q: (nq, D) vectores consulta.
        X: (n, D) base de datos.
        sims: (nq, n) similitudes ya calculadas (coseno o producto punto).
        n_qe: cuantos vecinos usar.
        alpha: exponente del peso.
        include_query: si suma tambien la consulta original.

    Returns:
        (nq, D) nuevos vectores consulta, L2-normalizados.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float32))
    X = np.asarray(X, dtype=np.float32)
    sims = np.atleast_2d(np.asarray(sims, dtype=np.float32))

    if n_qe <= 0:
        return _l2(Q)
    if sims.shape[0] != Q.shape[0] or sims.shape[1] != X.shape[0]:
        raise ValueError(
            f"sims {sims.shape} no concuerda con Q {Q.shape} y X {X.shape}"
        )

    n_qe = min(n_qe, X.shape[0])
    expanded = np.zeros_like(Q, dtype=np.float32)

    for i in range(Q.shape[0]):
        row = sims[i]
        # argpartition evita ordenar toda la base solo para sacar el top-n.
        top = np.argpartition(-row, n_qe - 1)[:n_qe]
        top = top[np.argsort(-row[top])]

        # Las similitudes negativas no deben aportar: un vecino "opuesto" no es
        # evidencia a favor. Se recortan a 0 antes de elevar a alpha, que
        # ademas evita potencias de numeros negativos.
        weights = np.power(np.clip(row[top], 0.0, None), alpha).astype(np.float32)

        aggregated = (weights[:, None] * X[top]).sum(axis=0)
        if include_query:
            aggregated = aggregated + Q[i]
        expanded[i] = aggregated

    return _l2(expanded)


# --------------------------------------------------------------------------- #
# Database-side augmentation
# --------------------------------------------------------------------------- #


def database_side_augmentation(
    X: np.ndarray, *, n_dba: int = 5, alpha: float = 3.0, chunk_size: int = 512
) -> np.ndarray:
    """Reemplaza cada vector de la base por una mezcla con sus vecinos.

    Es la QE aplicada del otro lado: si expandir la consulta con sus vecinos
    ayuda, expandir tambien cada imagen de la base ayuda por el mismo motivo.
    Cada imagen queda representada por un promedio ponderado de si misma y de
    las vistas parecidas del mismo objeto, lo que la hace alcanzable desde
    puntos de vista mas variados.

    Se hace UNA VEZ, sin conocer las consultas, asi que no cuesta nada en
    tiempo de busqueda. Se combina bien con `alpha_query_expansion`.

    Riesgo: si la base tiene muchas imagenes ambiguas, propaga el error. Por
    eso n_dba suele ser mas chico que el n de la query expansion.
    """
    Xn = _l2(np.asarray(X, dtype=np.float32))
    n = Xn.shape[0]
    if n_dba <= 0 or n <= 1:
        return Xn

    n_dba = min(n_dba, n - 1)
    out = np.empty_like(Xn)

    for start in range(0, n, chunk_size):
        block = Xn[start : start + chunk_size]
        sims = block @ Xn.T  # (b, n)

        for i in range(block.shape[0]):
            row = sims[i]
            row[start + i] = -np.inf  # el propio vector se suma aparte
            top = np.argpartition(-row, n_dba - 1)[:n_dba]
            weights = np.power(np.clip(row[top], 0.0, None), alpha).astype(np.float32)
            out[start + i] = block[i] + (weights[:, None] * Xn[top]).sum(axis=0)

    return _l2(out)


# --------------------------------------------------------------------------- #
# Difusion en el grafo de similitud
# --------------------------------------------------------------------------- #


def build_knn_graph(
    X: np.ndarray, *, k: int = 50, alpha: float = 3.0, chunk_size: int = 512
):
    """Grafo k-NN disperso y simetrico sobre la base de datos.

    Cada nodo se conecta a sus k vecinos mas parecidos, con peso
    similitud^alpha. Elevar a alpha hace que las aristas debiles casi
    desaparezcan, que es lo que evita que la difusion se escape a otro
    landmark por una conexion espuria.

    Se simetriza con el maximo, que es una aproximacion barata al grafo
    k-reciproco: una arista sobrevive si al menos uno de los dos nodos
    considera vecino al otro.
    """
    from scipy import sparse

    Xn = _l2(np.asarray(X, dtype=np.float32))
    n = Xn.shape[0]
    k = max(1, min(k, n - 1))

    rows = np.empty(n * k, dtype=np.int32)
    cols = np.empty(n * k, dtype=np.int32)
    vals = np.empty(n * k, dtype=np.float32)
    pos = 0

    for start in range(0, n, chunk_size):
        block = Xn[start : start + chunk_size]
        sims = block @ Xn.T  # (b, n)

        for i in range(block.shape[0]):
            gi = start + i
            row = sims[i]
            row[gi] = -np.inf  # sin lazos
            top = np.argpartition(-row, k - 1)[:k]

            rows[pos : pos + k] = gi
            cols[pos : pos + k] = top
            vals[pos : pos + k] = np.power(np.clip(row[top], 0.0, None), alpha)
            pos += k

    W = sparse.csr_matrix((vals[:pos], (rows[:pos], cols[:pos])), shape=(n, n))
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def normalize_graph(W):
    """Normalizacion simetrica S = D^-1/2 W D^-1/2.

    Impide que los nodos muy conectados (los "hubs", tipicos en retrieval de
    landmarks: una foto generica de cielo se parece un poco a todo) acaparen
    la masa que se propaga.
    """
    from scipy import sparse

    degrees = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    nonzero = degrees > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    D = sparse.diags(inv_sqrt)
    return (D @ W @ D).astype(np.float32)


def diffusion_rerank(
    S,
    sims: np.ndarray,
    *,
    k_seed: int = 10,
    alpha: float = 0.9,
    iters: int = 20,
) -> np.ndarray:
    """Re-rankea propagando la similitud por el grafo de la base de datos.

    Idea, de Iscen et al. "Efficient Diffusion on Region Manifolds" (CVPR
    2017): las imagenes de un mismo landmark no forman una bola en el espacio
    de descriptores, forman una VARIEDAD alargada — una cadena de vistas donde
    cada una se parece a la siguiente, pero los extremos no se parecen entre
    si. La distancia directa no puede recorrer esa cadena; la difusion si,
    porque propaga la similitud paso a paso por el grafo de vecinos.

    Es justo lo que arregla el fallo tipico de este dataset: la foto tomada
    desde el lado opuesto del edificio, que ningun descriptor global va a
    emparejar directamente pero que si esta conectada a traves de vistas
    intermedias.

    Se resuelve (I - alpha*S) f = y iterativamente:

        f_{t+1} = alpha * S * f_t + (1 - alpha) * y

    que converge porque los autovalores de S estan en [-1, 1] y alpha < 1.
    Con alpha -> 0 se recupera el ranking original; alpha alto propaga mas
    lejos y arriesga fugarse a otro landmark.

    Args:
        S: grafo normalizado (salida de `normalize_graph`).
        sims: (nq, n) similitudes iniciales.
        k_seed: cuantos vecinos iniciales siembran la difusion. Sembrar con
            TODA la fila deja entrar mucho ruido.
        alpha: cuanto se propaga (tipico 0.9 - 0.99).
        iters: iteraciones de punto fijo.

    Returns:
        (nq, n) nuevas puntuaciones, mayor = mas parecido.
    """
    sims = np.atleast_2d(np.asarray(sims, dtype=np.float32))
    nq, n = sims.shape
    if S.shape[0] != n:
        raise ValueError(f"el grafo tiene {S.shape[0]} nodos y sims tiene {n} columnas")

    alpha = float(np.clip(alpha, 0.0, 0.999))
    k_seed = max(1, min(k_seed, n))

    # Semillas: solo los k_seed mejores de cada consulta, recortados a >= 0.
    Y = np.zeros((nq, n), dtype=np.float32)
    for i in range(nq):
        row = sims[i]
        top = np.argpartition(-row, k_seed - 1)[:k_seed]
        Y[i, top] = np.clip(row[top], 0.0, None)

    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    Y = Y / np.maximum(norms, EPS)

    F = Y.copy()
    for _ in range(iters):
        # (S @ F.T).T en vez de F @ S: scipy multiplica disperso por denso de
        # forma eficiente en ese orden y devuelve un ndarray limpio.
        F = alpha * np.asarray((S @ F.T).T) + (1.0 - alpha) * Y

    return F.astype(np.float32)


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
