"""Tests del harness de evaluacion.

El mAP es el 8 de la nota y se compara contra otros grupos, asi que el codigo
que lo calcula no puede ser "seguramente esta bien". Estos tests verifican el
AP contra valores calculados a mano y contra propiedades que deben cumplirse
siempre (ranking perfecto = 1.0, junk no afecta, etc.).

Correr:
    python test_evaluation.py        # sin dependencias extra
    pytest test_evaluation.py -q     # si tienes pytest
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from evaluation import (
    Query,
    average_precision,
    average_precision_simple,
    evaluate,
    normalize_image_id,
    precision_at_k,
    recall_at_k,
)
from retrieval import (
    alpha_query_expansion,
    build_knn_graph,
    database_side_augmentation,
    diffusion_rerank,
    fuse,
    normalize_graph,
    load_embeddings,
    normalize_features,
    rank,
    rank_normalize,
    similarity,
    split_ids,
)

TOL = 1e-9


def approx(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


# --------------------------------------------------------------------------- #
# normalize_image_id
# --------------------------------------------------------------------------- #


def test_normalize_image_id():
    assert normalize_image_id("oxc1_all_souls_000013") == "all_souls_000013"
    assert normalize_image_id("all_souls_000013.jpg") == "all_souls_000013"
    assert normalize_image_id("  paris_defense_000331  ") == "paris_defense_000331"
    assert normalize_image_id("data/oxford/hertford_000015.jpg") == "hertford_000015"
    assert normalize_image_id("") == ""
    # "oxc1_" solo se quita al inicio
    assert normalize_image_id("x_oxc1_abc") == "x_oxc1_abc"


# --------------------------------------------------------------------------- #
# average_precision (protocolo oficial)
# --------------------------------------------------------------------------- #


def test_ap_ranking_perfecto():
    # Todos los positivos arriba -> AP = 1.0 exacto
    assert approx(average_precision(["a", "b", "c", "d"], {"a", "b"}), 1.0)


def test_ap_calculado_a_mano():
    # positivos = {b}, ranking = [a, b]
    #   k=1 (a): recall 0.0, precision 0.0 -> ap += (0-0)*((1+0)/2) = 0
    #   k=2 (b): recall 1.0, precision 0.5 -> ap += (1-0)*((0+0.5)/2) = 0.25
    assert approx(average_precision(["a", "b"], {"b"}), 0.25)

    # positivos = {a, c}, ranking = [a, b, c]
    #   k=1 (a): r=0.5 p=1.0   -> ap += 0.5 * (1.0+1.0)/2 = 0.5
    #   k=2 (b): r=0.5 p=0.5   -> ap += 0
    #   k=3 (c): r=1.0 p=2/3   -> ap += 0.5 * (0.5 + 2/3)/2 = 0.2916666...
    esperado = 0.5 + 0.5 * ((0.5 + 2.0 / 3.0) / 2.0)
    assert approx(average_precision(["a", "b", "c"], {"a", "c"}), esperado)


def test_ap_junk_se_ignora():
    # La junk no ocupa lugar en el ranking: [j, b] se evalua como [b]
    assert approx(average_precision(["j", "b"], {"b"}, {"j"}), 1.0)
    # Y no cuenta como acierto aunque estuviera en positivos por error
    assert approx(average_precision(["j", "b"], {"b", "j"}, {"j"}), 0.5)


def test_ap_sin_positivos_es_cero():
    assert approx(average_precision(["a", "b"], set()), 0.0)


def test_ap_ranking_vacio_es_cero():
    assert approx(average_precision([], {"a"}), 0.0)


def test_ap_positivo_faltante_baja_el_recall():
    # Solo 1 de 2 positivos aparece -> el recall se queda en 0.5
    ap = average_precision(["a", "x", "y"], {"a", "z"})
    assert approx(ap, 0.5)


def test_ap_es_monotono_al_subir_un_positivo():
    positivos = {"p1", "p2"}
    peor = average_precision(["n1", "n2", "p1", "p2"], positivos)
    mejor = average_precision(["n1", "p1", "n2", "p2"], positivos)
    optimo = average_precision(["p1", "p2", "n1", "n2"], positivos)
    assert peor < mejor < optimo
    assert approx(optimo, 1.0)


def test_ap_acotado_en_cero_uno():
    rng = np.random.default_rng(0)
    universo = [f"img{i}" for i in range(50)]
    for _ in range(200):
        positivos = set(rng.choice(universo, size=5, replace=False).tolist())
        junk = set(rng.choice(universo, size=3, replace=False).tolist()) - positivos
        ranked = list(rng.permutation(universo))
        ap = average_precision(ranked, positivos, junk)
        assert -TOL <= ap <= 1.0 + TOL, ap


# --------------------------------------------------------------------------- #
# average_precision_simple (formula del enunciado)
# --------------------------------------------------------------------------- #


def test_ap_simple_calculado_a_mano():
    # positivos = {b}, ranking = [a, b] -> P(2) = 0.5, N = 1 -> AP = 0.5
    assert approx(average_precision_simple(["a", "b"], {"b"}), 0.5)
    # ranking perfecto -> 1.0
    assert approx(average_precision_simple(["a", "b", "c"], {"a", "b"}), 1.0)


def test_ap_simple_difiere_del_oficial():
    # Esta diferencia es esperada y hay que explicarla en el reporte:
    # la version oficial interpola, la del enunciado no.
    oficial = average_precision(["a", "b"], {"b"})
    simple = average_precision_simple(["a", "b"], {"b"})
    assert not approx(oficial, simple)
    assert approx(oficial, 0.25)
    assert approx(simple, 0.5)


def test_ap_simple_ignora_junk():
    assert approx(average_precision_simple(["j", "b"], {"b"}, {"j"}), 1.0)


# --------------------------------------------------------------------------- #
# precision_at_k / recall_at_k
# --------------------------------------------------------------------------- #


def test_precision_at_k():
    assert approx(precision_at_k(["a", "b", "c"], {"a", "b"}, 1), 1.0)
    assert approx(precision_at_k(["a", "b", "c"], {"a", "b"}, 2), 1.0)
    assert approx(precision_at_k(["a", "b", "c"], {"a", "b"}, 3), 2.0 / 3.0)
    assert approx(precision_at_k(["x", "a", "b"], {"a", "b"}, 2), 0.5)


def test_precision_at_k_salta_junk():
    # [j, a, b] con j junk y k=2 -> se evaluan a y b -> 1.0
    assert approx(precision_at_k(["j", "a", "b"], {"a", "b"}, 2, {"j"}), 1.0)


def test_precision_at_k_con_lista_corta():
    # Menos de k elementos no-junk: se divide por los que haya, no por k
    assert approx(precision_at_k(["a"], {"a"}, 5), 1.0)


def test_recall_at_k():
    assert approx(recall_at_k(["a", "b", "c"], {"a", "b", "z"}, 2), 2.0 / 3.0)
    assert approx(recall_at_k(["a"], set(), 5), 0.0)


# --------------------------------------------------------------------------- #
# evaluate()
# --------------------------------------------------------------------------- #


def _make_query(query_id: str, positives: set[str], junk: set[str] | None = None) -> Query:
    return Query(
        query_id=query_id,
        dataset="oxford",
        landmark=query_id.rsplit("_", 1)[0],
        image_id=f"{query_id}_img",
        bbox=None,
        positives=positives,
        good=positives,
        ok=set(),
        junk=junk or set(),
    )


def test_evaluate_map_promedia_las_queries():
    queries = [
        _make_query("all_souls_1", {"a"}),
        _make_query("all_souls_2", {"b"}),
    ]
    rankings = {
        "all_souls_1": ["a", "x"],  # AP = 1.0
        "all_souls_2": ["x", "b"],  # AP = 0.25
    }
    result = evaluate(queries, rankings)
    assert len(result.per_query) == 2
    assert approx(result.mean_ap, (1.0 + 0.25) / 2.0)


def test_evaluate_omite_queries_sin_ranking():
    queries = [_make_query("all_souls_1", {"a"}), _make_query("all_souls_2", {"b"})]
    result = evaluate(queries, {"all_souls_1": ["a"]})
    assert len(result.per_query) == 1

    estricto = evaluate(queries, {"all_souls_1": ["a"]}, skip_missing=False)
    assert len(estricto.per_query) == 2
    assert approx(estricto.mean_ap, 0.5)  # la faltante cuenta como 0


def test_evaluate_desglose_por_landmark():
    queries = [
        _make_query("all_souls_1", {"a"}),
        _make_query("hertford_1", {"b"}),
    ]
    rankings = {"all_souls_1": ["a"], "hertford_1": ["x", "b"]}
    result = evaluate(queries, rankings)
    por_landmark = result.by_landmark()
    assert approx(por_landmark["oxford/all_souls"], 1.0)
    assert approx(por_landmark["oxford/hertford"], 0.25)


def test_evaluate_peores_y_mejores():
    queries = [_make_query(f"lm_{i}", {"a"}) for i in range(3)]
    rankings = {
        "lm_0": ["a"],
        "lm_1": ["x", "a"],
        "lm_2": ["x", "y", "a"],
    }
    result = evaluate(queries, rankings)
    assert result.best_queries(1)[0].query_id == "lm_0"
    assert result.worst_queries(1)[0].query_id == "lm_2"


# --------------------------------------------------------------------------- #
# retrieval: normalizaciones
# --------------------------------------------------------------------------- #


def test_normalizaciones():
    X = np.array([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]], dtype=np.float32)

    assert np.allclose(normalize_features(X, "none"), X)

    l1 = normalize_features(X, "l1")
    assert approx(float(l1[0].sum()), 1.0, 1e-6)
    assert approx(float(l1[2].sum()), 1.0, 1e-6)
    assert approx(float(l1[1].sum()), 0.0, 1e-6)  # fila nula no explota

    l2 = normalize_features(X, "l2")
    assert approx(float(np.linalg.norm(l2[0])), 1.0, 1e-6)
    assert approx(float(np.linalg.norm(l2[1])), 0.0, 1e-6)

    for mode in ("power", "hellinger"):
        Y = normalize_features(X, mode)
        assert Y.shape == X.shape
        assert np.isfinite(Y).all()


def test_normalizacion_desconocida_falla():
    try:
        normalize_features(np.zeros((2, 2), dtype=np.float32), "loquesea")
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError")


# --------------------------------------------------------------------------- #
# retrieval: metricas
# --------------------------------------------------------------------------- #


def test_metricas_ponen_primero_al_identico():
    """Toda metrica de distancia real debe rankear #1 a la imagen identica.

    `dot` queda fuera a proposito: el producto punto crudo depende de la norma,
    asi que un vector mas largo puede ganarle al identico. Ese es justamente el
    motivo por el que los histogramas se normalizan antes de comparar; ver
    `test_dot_crudo_es_sensible_a_la_norma`.
    """
    rng = np.random.default_rng(7)
    X = np.abs(rng.random((40, 12))).astype(np.float32) + 0.01
    names = np.asarray([f"img{i}" for i in range(40)], dtype=object)

    for metric in ("cosine", "l2", "l1", "chi2", "intersection", "hellinger"):
        sim = similarity(X[3:4], X, metric=metric)
        assert sim.shape == (1, 40)
        assert np.isfinite(sim).all(), metric
        assert rank(sim[0], names)[0] == "img3", metric


def test_dot_crudo_es_sensible_a_la_norma():
    # v y 10*v apuntan en la misma direccion, pero el producto punto premia al
    # mas largo: la "mas parecida" a v resulta ser 10*v, no v.
    X = np.array([[1.0, 1.0], [10.0, 10.0]], dtype=np.float32)
    names = np.asarray(["v", "10v"], dtype=object)

    sim_dot = similarity(X[0:1], X, metric="dot")
    assert rank(sim_dot[0], names)[0] == "10v"

    # Con coseno el empate direccional es exacto y v vuelve a estar primero
    # (desempate estable por indice).
    sim_cos = similarity(X[0:1], X, metric="cosine")
    assert approx(float(sim_cos[0, 0]), float(sim_cos[0, 1]), 1e-5)
    assert rank(sim_cos[0], names)[0] == "v"


def test_cosine_equivale_a_dot_tras_l2():
    rng = np.random.default_rng(11)
    X = rng.random((25, 9)).astype(np.float32)
    a = similarity(X[:3], X, metric="cosine")
    b = similarity(normalize_features(X[:3], "l2"), normalize_features(X, "l2"), metric="dot")
    assert np.allclose(a, b, atol=1e-5)


def test_l2_es_distancia_negada():
    a = np.array([[0.0, 0.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0]], dtype=np.float32)
    assert approx(float(similarity(a, b, metric="l2")[0, 0]), -5.0, 1e-5)


def test_chi2_es_cero_para_vectores_identicos():
    v = np.array([[0.2, 0.5, 0.3]], dtype=np.float32)
    assert approx(float(similarity(v, v, metric="chi2")[0, 0]), 0.0, 1e-6)


def test_metrica_desconocida_falla():
    try:
        similarity(np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32), "xyz")
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError")


# --------------------------------------------------------------------------- #
# retrieval: ids, carga y fusion
# --------------------------------------------------------------------------- #


def test_split_ids():
    ids = np.asarray(
        ["oxford/all_souls_000013", "paris/paris_eiffel_000001", "hertford_000015", "paris_louvre_000002"],
        dtype=object,
    )
    datasets, names = split_ids(ids)
    assert list(datasets) == ["oxford", "paris", "oxford", "paris"]
    assert list(names) == [
        "all_souls_000013",
        "paris_eiffel_000001",
        "hertford_000015",
        "paris_louvre_000002",
    ]


def test_load_embeddings_npz_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "emb.npz"
        ids = np.asarray(["oxford/a", "oxford/b"], dtype=object)
        feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # Se usan los nombres que ya escribe 2_visual_vocabulary.py
        np.savez_compressed(path, image_ids=ids, histograms=feats)

        loaded_ids, loaded_feats = load_embeddings(path)
        assert list(loaded_ids) == ["oxford/a", "oxford/b"]
        assert np.allclose(loaded_feats, feats)


def test_rank_normalize_lleva_a_cero_uno():
    sim = np.array([[1.0, 3.0, 5.0], [-2.0, 0.0, 2.0]], dtype=np.float32)
    norm = rank_normalize(sim)
    assert np.allclose(norm.min(axis=1), 0.0)
    assert np.allclose(norm.max(axis=1), 1.0)


def test_fuse_respeta_los_pesos():
    a = np.array([[1.0, 0.0]], dtype=np.float32)  # prefiere la imagen 0
    b = np.array([[0.0, 1.0]], dtype=np.float32)  # prefiere la imagen 1

    solo_a = fuse([a, b], [1.0, 0.0])
    assert solo_a[0, 0] > solo_a[0, 1]

    solo_b = fuse([a, b], [0.0, 1.0])
    assert solo_b[0, 1] > solo_b[0, 0]

    empate = fuse([a, b], [0.5, 0.5])
    assert approx(float(empate[0, 0]), float(empate[0, 1]), 1e-6)


# --------------------------------------------------------------------------- #
# Query expansion
# --------------------------------------------------------------------------- #


def test_qe_devuelve_vectores_normalizados():
    rng = np.random.default_rng(3)
    X = _l2_rows(rng.random((30, 8)).astype(np.float32))
    Q = X[:4]
    sims = similarity(Q, X, metric="cosine")

    out = alpha_query_expansion(Q, X, sims, n_qe=5, alpha=3.0)
    assert out.shape == Q.shape
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_qe_con_n_cero_no_cambia_nada():
    rng = np.random.default_rng(4)
    X = _l2_rows(rng.random((10, 5)).astype(np.float32))
    Q = X[:2]
    sims = similarity(Q, X, metric="cosine")
    out = alpha_query_expansion(Q, X, sims, n_qe=0)
    assert np.allclose(out, Q, atol=1e-5)


def test_qe_acerca_la_query_a_sus_vecinos():
    """Con vecinos coherentes, la consulta expandida debe parecerse mas al
    centro del grupo que la consulta original."""
    cluster = np.array(
        [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.9, 0.1, 0.0], [0.92, 0.08, 0.0]],
        dtype=np.float32,
    )
    lejos = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    X = _l2_rows(np.vstack([cluster, lejos]))
    Q = X[0:1]

    sims = similarity(Q, X, metric="cosine")
    Q2 = alpha_query_expansion(Q, X, sims, n_qe=4, alpha=3.0)

    centro = _l2_rows(cluster.mean(axis=0, keepdims=True))
    antes = float((Q @ centro.T)[0, 0])
    despues = float((Q2 @ centro.T)[0, 0])
    assert despues > antes


def test_qe_ignora_similitudes_negativas():
    # Un vecino "opuesto" no debe restar: su peso se recorta a 0.
    X = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    Q = X[0:1]
    sims = similarity(Q, X, metric="cosine")
    out = alpha_query_expansion(Q, X, sims, n_qe=2, alpha=3.0)
    # Debe seguir apuntando en la direccion original, no cancelarse.
    assert float((out @ Q.T)[0, 0]) > 0.99


def test_qe_alpha_cero_es_average_qe():
    rng = np.random.default_rng(5)
    X = _l2_rows(rng.random((12, 6)).astype(np.float32))
    Q = X[0:1]
    sims = similarity(Q, X, metric="cosine")

    out = alpha_query_expansion(Q, X, sims, n_qe=3, alpha=0.0)

    order = np.argsort(-sims[0])[:3]
    manual = X[order].sum(axis=0) + Q[0]
    manual = manual / np.linalg.norm(manual)
    assert np.allclose(out[0], manual, atol=1e-5)


def test_qe_rechaza_sims_con_forma_incorrecta():
    X = np.zeros((5, 3), dtype=np.float32)
    Q = np.zeros((2, 3), dtype=np.float32)
    try:
        alpha_query_expansion(Q, X, np.zeros((2, 4), dtype=np.float32), n_qe=2)
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError")


def _l2_rows(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------- #
# DBA y difusion
# --------------------------------------------------------------------------- #


def test_dba_devuelve_vectores_normalizados():
    rng = np.random.default_rng(6)
    X = rng.random((20, 7)).astype(np.float32)
    out = database_side_augmentation(X, n_dba=3, alpha=3.0)
    assert out.shape == X.shape
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_dba_con_cero_vecinos_solo_normaliza():
    rng = np.random.default_rng(7)
    X = rng.random((8, 4)).astype(np.float32)
    assert np.allclose(database_side_augmentation(X, n_dba=0), _l2_rows(X), atol=1e-6)


def test_grafo_knn_es_simetrico_y_sin_lazos():
    rng = np.random.default_rng(8)
    X = _l2_rows(rng.random((25, 6)).astype(np.float32))
    W = build_knn_graph(X, k=4)

    assert W.shape == (25, 25)
    assert abs((W - W.T)).max() < 1e-6, "el grafo debe ser simetrico"
    assert np.allclose(W.diagonal(), 0.0), "no debe haber lazos"
    assert W.nnz > 0


def test_grafo_normalizado_no_tiene_nan_con_nodos_aislados():
    # Un nodo sin aristas tendria grado 0: la normalizacion no debe dividir
    # por cero.
    from scipy import sparse

    W = sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32))
    S = normalize_graph(W)
    assert np.all(np.isfinite(S.toarray()))


def test_difusion_recupera_el_extremo_de_la_variedad():
    """La prueba de fuego de la difusion.

    Cada clase se construye como una CADENA de vistas: los extremos son
    ortogonales entre si (similitud 0) pero estan unidos por vistas
    intermedias. El coseno directo no puede cruzar la cadena; la difusion si,
    porque propaga paso a paso por el grafo de vecinos.

    Es exactamente el fallo que importa en Oxford: la foto del lado opuesto
    del edificio.
    """
    rng = np.random.default_rng(0)
    dim, n_clases, largo = 64, 6, 20

    vectores, etiquetas = [], []
    for c in range(n_clases):
        a = rng.normal(size=dim)
        a /= np.linalg.norm(a)
        b = rng.normal(size=dim)
        b -= (b @ a) * a
        b /= np.linalg.norm(b)
        for t in np.linspace(0, np.pi / 2, largo):
            v = np.cos(t) * a + np.sin(t) * b + rng.normal(0, 0.01, dim)
            vectores.append(v / np.linalg.norm(v))
            etiquetas.append(c)

    X = np.asarray(vectores, dtype=np.float32)
    etiquetas = np.asarray(etiquetas)
    nombres = np.asarray(
        [f"c{e}_{i % largo:02d}" for i, e in enumerate(etiquetas)], dtype=object
    )

    idx_q = [c * largo for c in range(n_clases)]
    sims = similarity(X[idx_q], X, metric="cosine")

    S = normalize_graph(build_knn_graph(X, k=5))
    difundido = diffusion_rerank(S, sims, k_seed=5, alpha=0.95, iters=30)

    def calcular_map(scores):
        aps = []
        for i, c in enumerate(range(n_clases)):
            ranked = [str(nombres[j]) for j in np.argsort(-scores[i])]
            positivos = {str(nombres[j]) for j in np.flatnonzero(etiquetas == c)}
            aps.append(average_precision(ranked, positivos))
        return float(np.mean(aps))

    base = calcular_map(sims)
    con_difusion = calcular_map(difundido)

    assert con_difusion > base, f"difusion={con_difusion} no mejoro sobre {base}"
    assert con_difusion > 0.99, f"difusion deberia resolver este caso, dio {con_difusion}"


def test_difusion_rechaza_grafo_de_tamano_incorrecto():
    rng = np.random.default_rng(9)
    X = _l2_rows(rng.random((12, 5)).astype(np.float32))
    S = normalize_graph(build_knn_graph(X, k=3))
    try:
        diffusion_rerank(S, np.zeros((2, 7), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError")


# --------------------------------------------------------------------------- #
# Test end-to-end sintetico
# --------------------------------------------------------------------------- #


def test_end_to_end_descriptor_perfecto_vs_aleatorio():
    """Con un descriptor que separa perfectamente las clases, mAP debe ser 1.0;
    con vectores aleatorios debe ser bajo. Si esto no se cumple, hay un bug en
    el pipeline de ranking, no en el descriptor."""
    rng = np.random.default_rng(42)

    n_landmarks, por_landmark, dim = 5, 8, 16
    names = []
    etiquetas = []
    for lm in range(n_landmarks):
        for j in range(por_landmark):
            names.append(f"lm{lm}_{j:03d}")
            etiquetas.append(lm)
    names_arr = np.asarray(names, dtype=object)
    etiquetas = np.asarray(etiquetas)

    # Descriptor perfecto: one-hot del landmark + ruido minusculo
    perfecto = np.zeros((len(names), dim), dtype=np.float32)
    perfecto[np.arange(len(names)), etiquetas] = 1.0
    perfecto += rng.normal(0, 1e-4, perfecto.shape).astype(np.float32)

    aleatorio = rng.random((len(names), dim)).astype(np.float32)

    for X, esperado_alto in ((perfecto, True), (aleatorio, False)):
        queries = []
        rankings = {}
        for lm in range(n_landmarks):
            qname = f"lm{lm}_000"
            qidx = names.index(qname)
            positives = {n for n, e in zip(names, etiquetas) if e == lm}

            queries.append(
                Query(
                    query_id=f"lm{lm}_1",
                    dataset="oxford",
                    landmark=f"lm{lm}",
                    image_id=qname,
                    bbox=None,
                    positives=positives,
                    good=positives,
                    ok=set(),
                    junk=set(),
                )
            )
            sim = similarity(X[qidx : qidx + 1], X, metric="cosine")
            rankings[f"lm{lm}_1"] = rank(sim[0], names_arr)

        mAP = evaluate(queries, rankings).mean_ap
        if esperado_alto:
            assert mAP > 0.99, f"descriptor perfecto dio mAP={mAP}"
        else:
            assert mAP < 0.75, f"descriptor aleatorio dio mAP={mAP} (sospechoso)"


# --------------------------------------------------------------------------- #
# Runner sin pytest
# --------------------------------------------------------------------------- #


def main() -> None:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]

    fallos = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            fallos.append((name, exc))
            print(f"  FALLO {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            fallos.append((name, exc))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")

    print()
    print(f"{len(tests) - len(fallos)}/{len(tests)} tests pasaron")
    raise SystemExit(1 if fallos else 0)


if __name__ == "__main__":
    main()
