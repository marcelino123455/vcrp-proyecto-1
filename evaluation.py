"""Protocolo de evaluacion para image retrieval en Oxford5k / Paris6k.

Este modulo es la *unica* fuente de verdad para las metricas del proyecto.
Cualquier representacion (BoVW, HOG, GIST, VLAD, fusion...) debe evaluarse
llamando a estas funciones, para que todos los numeros del reporte sean
comparables entre si y con la literatura.

Protocolo estandar (Philbin et al., CVPR 2007)
----------------------------------------------
Cada dataset trae 55 queries (11 landmarks x 5 queries). Para cada query hay:

    <query_id>_query.txt  -> nombre de la imagen consulta + bounding box
    <query_id>_good.txt   -> el landmark es claramente visible
    <query_id>_ok.txt     -> el landmark es visible en mas del 25%
    <query_id>_junk.txt   -> menos del 25% visible, u oclusion severa

El conjunto positivo es good + ok. Las imagenes *junk* NO son negativas:
se eliminan del ranking antes de calcular la precision. Esa es la parte que
mas facilmente se implementa mal y la que hace que un mAP no sea comparable
con el de otro grupo.

La funcion `average_precision` es un port linea por linea de `compute_ap.cpp`
distribuido por VGG (y de su equivalente en Python), no una reimplementacion
"equivalente". Se usa interpolacion trapezoidal sobre la curva
precision-recall, que da valores ligeramente distintos a la suma simple
`1/N * sum(P(k) * rel(k))` del enunciado del curso.

Ambas estan implementadas:
    - ap_mode="official" -> compute_ap de VGG (comparable con papers)
    - ap_mode="simple"   -> formula del PDF del curso

Uso como libreria
-----------------
    from evaluation import load_ground_truth, evaluate

    queries = load_ground_truth("data/groundtruth", "oxford")
    results = evaluate(queries, rankings)   # rankings: query_id -> [image_id, ...]
    print(results.mean_ap)

Uso como script (sanity check del ground truth)
------------------------------------------------
    python evaluation.py --check
    python evaluation.py --check --gt-dir data/groundtruth
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GT_DIR = Path(__file__).resolve().parent / "data" / "groundtruth"

DATASETS = ("oxford", "paris")

# Los 11 landmarks de cada dataset. Se usan solo para validar la descarga y
# para agrupar resultados por landmark en el reporte.
OXFORD_LANDMARKS = (
    "all_souls",
    "ashmolean",
    "balliol",
    "bodleian",
    "christ_church",
    "cornmarket",
    "hertford",
    "keble",
    "magdalen",
    "pitt_rivers",
    "radcliffe_camera",
)

PARIS_LANDMARKS = (
    "defense",
    "eiffel",
    "invalides",
    "louvre",
    "moulinrouge",
    "museedorsay",
    "notredame",
    "pantheon",
    "pompidou",
    "sacrecoeur",
    "triomphe",
)

LANDMARKS_BY_DATASET = {
    "oxford": OXFORD_LANDMARKS,
    "paris": PARIS_LANDMARKS,
}

# El ground truth de Oxford nombra la imagen consulta con el prefijo "oxc1_",
# que NO aparece en los nombres de archivo reales ni en good/ok/junk.
_OXC1_PREFIX = re.compile(r"^oxc1_")

DEFAULT_KS = (1, 5, 10, 20)


# --------------------------------------------------------------------------- #
# Parseo del ground truth
# --------------------------------------------------------------------------- #


def normalize_image_id(raw: str) -> str:
    """Normaliza un nombre de imagen del ground truth a nuestro `image_id`.

    Quita el prefijo `oxc1_` de Oxford, la extension `.jpg` si viene, y
    cualquier ruta previa. El resultado coincide con `Path(archivo).stem`,
    que es exactamente la clave usada en `sift_features.h5`.
    """
    name = raw.strip()
    if not name:
        return ""
    name = name.replace("\\", "/").split("/")[-1]
    name = _OXC1_PREFIX.sub("", name)
    if name.lower().endswith(".jpg"):
        name = name[:-4]
    return name


def _load_id_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return {nid for nid in (normalize_image_id(line) for line in f) if nid}


@dataclass
class Query:
    """Una query del protocolo, con su imagen, su bbox y sus etiquetas."""

    query_id: str  # "all_souls_1"
    dataset: str  # "oxford" | "paris"
    landmark: str  # "all_souls"
    image_id: str  # "all_souls_000013"
    bbox: tuple[float, float, float, float] | None  # (x1, y1, x2, y2)
    positives: set[str] = field(default_factory=set)  # good + ok
    good: set[str] = field(default_factory=set)
    ok: set[str] = field(default_factory=set)
    junk: set[str] = field(default_factory=set)

    @property
    def num_positives(self) -> int:
        return len(self.positives)


def _parse_query_file(path: Path) -> tuple[str, tuple[float, float, float, float] | None]:
    """Lee `<query_id>_query.txt` y devuelve (image_id, bbox_o_None)."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        line = ""
        for candidate in f:
            if candidate.strip():
                line = candidate.strip()
                break

    if not line:
        raise ValueError(f"Archivo de query vacio: {path}")

    tokens = line.split()
    image_id = normalize_image_id(tokens[0])

    bbox: tuple[float, float, float, float] | None = None
    if len(tokens) >= 5:
        try:
            x1, y1, x2, y2 = (float(t) for t in tokens[1:5])
            bbox = (x1, y1, x2, y2)
        except ValueError:
            bbox = None

    return image_id, bbox


def load_ground_truth(
    gt_dir: str | Path = DEFAULT_GT_DIR,
    dataset: str = "oxford",
    *,
    strict: bool = False,
) -> list[Query]:
    """Carga las queries de `dataset` desde `<gt_dir>/<dataset>/`.

    Busca recursivamente los `*_query.txt`, asi que funciona tanto si el
    tarball se extrajo plano como si dejo un subdirectorio.

    Con `strict=True` levanta error si falta algun archivo good/ok/junk;
    por defecto solo avisa (Paris, por ejemplo, tiene queries sin junk).
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset debe ser uno de {DATASETS}, se recibio {dataset!r}")

    root = Path(gt_dir) / dataset
    if not root.exists():
        raise FileNotFoundError(
            f"No existe {root}. Ejecuta primero ./data/download_groundtruth.sh"
        )

    query_files = sorted(root.rglob("*_query.txt"))
    if not query_files:
        raise FileNotFoundError(
            f"No se encontro ningun *_query.txt bajo {root}. "
            "Revisa que el tarball del ground truth se haya extraido bien."
        )

    queries: list[Query] = []
    for query_file in query_files:
        query_id = query_file.name[: -len("_query.txt")]
        image_id, bbox = _parse_query_file(query_file)

        base = query_file.parent / query_id
        good = _load_id_list(Path(f"{base}_good.txt"))
        ok = _load_id_list(Path(f"{base}_ok.txt"))
        junk = _load_id_list(Path(f"{base}_junk.txt"))

        if strict and not (good or ok):
            raise ValueError(f"Query sin positivos: {query_id} ({dataset})")

        # El query_id es "<landmark>_<n>"; el landmark puede llevar guiones
        # bajos (christ_church_1 -> christ_church).
        landmark = query_id.rsplit("_", 1)[0]

        queries.append(
            Query(
                query_id=query_id,
                dataset=dataset,
                landmark=landmark,
                image_id=image_id,
                bbox=bbox,
                positives=good | ok,
                good=good,
                ok=ok,
                junk=junk,
            )
        )

    return queries


def load_all_ground_truth(
    gt_dir: str | Path = DEFAULT_GT_DIR,
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, list[Query]]:
    return {d: load_ground_truth(gt_dir, d) for d in datasets}


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #


def average_precision(
    ranked_list: list[str],
    positives: set[str],
    junk: set[str] | None = None,
) -> float:
    """Average Precision segun `compute_ap.cpp` de VGG (protocolo oficial).

    Port linea por linea del benchmark original. Usa interpolacion trapezoidal
    sobre la curva precision-recall:

        AP = sum_k (r_k - r_{k-1}) * (p_{k-1} + p_k) / 2

    Las imagenes en `junk` se saltan por completo: no avanzan el contador de
    rango ni cuentan como fallo.

    Notas de borde:
        - Si `positives` esta vacio el AP es 0.0 (query degenerada).
        - Si el ranking no contiene todos los positivos, el AP baja
          proporcionalmente: el recall nunca llega a 1. Por eso el ranking
          que se pasa debe ser la base de datos COMPLETA, no solo el top-k.
    """
    if not positives:
        return 0.0

    junk = junk or set()

    intersect_size = 0.0
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    j = 0.0

    n_positives = float(len(positives))

    for image_id in ranked_list:
        if image_id in junk:
            continue

        j += 1.0
        if image_id in positives:
            intersect_size += 1.0

        recall = intersect_size / n_positives
        precision = intersect_size / j

        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

        old_recall = recall
        old_precision = precision

    return ap


def average_precision_simple(
    ranked_list: list[str],
    positives: set[str],
    junk: set[str] | None = None,
) -> float:
    """AP segun la formula del enunciado del curso:

        AP = (1 / N) * sum_{k=1}^{M} P(k) * rel(k)

    donde N es el numero total de imagenes relevantes de la query.
    Se mantiene el descarte de las junk para ser coherentes con el protocolo.

    Da valores ligeramente distintos a `average_precision` (sin interpolacion).
    Se reporta como metrica secundaria; la oficial es la que se compara entre
    grupos y con los papers.
    """
    if not positives:
        return 0.0

    junk = junk or set()

    hits = 0
    rank = 0
    total = 0.0

    for image_id in ranked_list:
        if image_id in junk:
            continue
        rank += 1
        if image_id in positives:
            hits += 1
            total += hits / rank

    return total / len(positives)


def precision_at_k(
    ranked_list: list[str],
    positives: set[str],
    k: int,
    junk: set[str] | None = None,
) -> float:
    """Precision@k, descartando las junk antes de tomar el top-k."""
    if k <= 0:
        raise ValueError("k debe ser positivo")

    junk = junk or set()

    hits = 0
    seen = 0
    for image_id in ranked_list:
        if image_id in junk:
            continue
        seen += 1
        if image_id in positives:
            hits += 1
        if seen == k:
            break

    # Si la base tiene menos de k imagenes no-junk, se divide por lo que haya.
    denominator = seen if seen > 0 else k
    return hits / denominator


def recall_at_k(
    ranked_list: list[str],
    positives: set[str],
    k: int,
    junk: set[str] | None = None,
) -> float:
    if not positives:
        return 0.0

    junk = junk or set()

    hits = 0
    seen = 0
    for image_id in ranked_list:
        if image_id in junk:
            continue
        seen += 1
        if image_id in positives:
            hits += 1
        if seen == k:
            break

    return hits / len(positives)


# --------------------------------------------------------------------------- #
# Agregacion
# --------------------------------------------------------------------------- #


@dataclass
class QueryResult:
    query_id: str
    dataset: str
    landmark: str
    image_id: str
    num_positives: int
    num_junk: int
    num_retrieved: int
    ap: float
    ap_simple: float
    precision_at: dict[int, float]

    @property
    def short_query_id(self) -> str:
        """query_id sin el prefijo "<dataset>::" que usa el runner internamente."""
        return self.query_id.split("::", 1)[-1]

    def as_row(self, ks: tuple[int, ...] = DEFAULT_KS) -> dict:
        row = {
            "dataset": self.dataset,
            "query_id": self.short_query_id,
            "landmark": self.landmark,
            "query_image": self.image_id,
            "num_positives": self.num_positives,
            "num_junk": self.num_junk,
            "num_retrieved": self.num_retrieved,
            "ap": round(self.ap, 6),
            "ap_simple": round(self.ap_simple, 6),
        }
        for k in ks:
            row[f"p@{k}"] = round(self.precision_at.get(k, 0.0), 6)
        return row


@dataclass
class EvaluationResult:
    per_query: list[QueryResult]
    ks: tuple[int, ...] = DEFAULT_KS

    @property
    def mean_ap(self) -> float:
        if not self.per_query:
            return 0.0
        return sum(q.ap for q in self.per_query) / len(self.per_query)

    @property
    def mean_ap_simple(self) -> float:
        if not self.per_query:
            return 0.0
        return sum(q.ap_simple for q in self.per_query) / len(self.per_query)

    def mean_precision_at(self, k: int) -> float:
        if not self.per_query:
            return 0.0
        return sum(q.precision_at.get(k, 0.0) for q in self.per_query) / len(self.per_query)

    def by_landmark(self) -> dict[str, float]:
        """mAP agrupado por landmark: donde se gana y se pierde."""
        grouped: dict[str, list[float]] = defaultdict(list)
        for q in self.per_query:
            grouped[f"{q.dataset}/{q.landmark}"].append(q.ap)
        return {k: sum(v) / len(v) for k, v in sorted(grouped.items())}

    def by_dataset(self) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for q in self.per_query:
            grouped[q.dataset].append(q.ap)
        return {k: sum(v) / len(v) for k, v in sorted(grouped.items())}

    def worst_queries(self, n: int = 5) -> list[QueryResult]:
        """Las peores queries: material directo para el analisis cualitativo."""
        return sorted(self.per_query, key=lambda q: q.ap)[:n]

    def best_queries(self, n: int = 5) -> list[QueryResult]:
        return sorted(self.per_query, key=lambda q: q.ap, reverse=True)[:n]

    def summary(self) -> dict:
        return {
            "num_queries": len(self.per_query),
            "mAP": round(self.mean_ap, 6),
            "mAP_simple": round(self.mean_ap_simple, 6),
            **{f"mP@{k}": round(self.mean_precision_at(k), 6) for k in self.ks},
            "by_dataset": {k: round(v, 6) for k, v in self.by_dataset().items()},
            "by_landmark": {k: round(v, 6) for k, v in self.by_landmark().items()},
        }

    def format_report(self) -> str:
        lines = [
            "=" * 62,
            f"{'RESULTADOS DE RETRIEVAL':^62}",
            "=" * 62,
            f"Queries evaluadas : {len(self.per_query)}",
            f"mAP (oficial VGG) : {self.mean_ap:.4f}",
            f"mAP (formula PDF) : {self.mean_ap_simple:.4f}",
        ]
        for k in self.ks:
            lines.append(f"Precision@{k:<7}: {self.mean_precision_at(k):.4f}")

        by_dataset = self.by_dataset()
        if len(by_dataset) > 1:
            lines.append("")
            lines.append("mAP por dataset:")
            for name, value in by_dataset.items():
                lines.append(f"  {name:<20} {value:.4f}")

        lines.append("")
        lines.append("mAP por landmark:")
        for name, value in self.by_landmark().items():
            lines.append(f"  {name:<28} {value:.4f}")

        lines.append("")
        lines.append("Peores queries (candidatas para el analisis de fallos):")
        for q in self.worst_queries(5):
            label = f"{q.dataset}/{q.short_query_id}"
            lines.append(
                f"  {label:<30} AP={q.ap:.4f}  ({q.num_positives} positivos)"
            )

        lines.append("=" * 62)
        return "\n".join(lines)


def evaluate(
    queries: list[Query],
    rankings: dict[str, list[str]],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    skip_missing: bool = True,
) -> EvaluationResult:
    """Evalua un conjunto de rankings contra el ground truth.

    Args:
        queries: salida de `load_ground_truth`.
        rankings: `query_id -> lista de image_id ordenada por similitud
            decreciente`. Debe contener TODA la base de datos, no solo el
            top-k, o el recall se trunca y el AP sale artificialmente bajo.
        ks: valores de k para Precision@k.
        skip_missing: si una query no tiene ranking, se omite en vez de
            contarla como AP=0. Ponlo en False para penalizar queries
            faltantes (util para detectar bugs).
    """
    results: list[QueryResult] = []

    for query in queries:
        ranked = rankings.get(query.query_id)
        if ranked is None:
            if skip_missing:
                continue
            ranked = []

        results.append(
            QueryResult(
                query_id=query.query_id,
                dataset=query.dataset,
                landmark=query.landmark,
                image_id=query.image_id,
                num_positives=query.num_positives,
                num_junk=len(query.junk),
                num_retrieved=len(ranked),
                ap=average_precision(ranked, query.positives, query.junk),
                ap_simple=average_precision_simple(ranked, query.positives, query.junk),
                precision_at={
                    k: precision_at_k(ranked, query.positives, k, query.junk) for k in ks
                },
            )
        )

    return EvaluationResult(per_query=results, ks=ks)


# --------------------------------------------------------------------------- #
# Sanity check del ground truth
# --------------------------------------------------------------------------- #


def check_ground_truth(
    gt_dir: str | Path = DEFAULT_GT_DIR,
    datasets: tuple[str, ...] = DATASETS,
) -> bool:
    """Valida que el ground truth se haya descargado y parseado bien."""
    all_ok = True

    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        try:
            queries = load_ground_truth(gt_dir, dataset)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ERROR: {exc}")
            all_ok = False
            continue

        expected_landmarks = set(LANDMARKS_BY_DATASET[dataset])
        found_landmarks = {q.landmark for q in queries}

        print(f"  queries            : {len(queries)} (esperadas: 55)")
        print(f"  landmarks          : {len(found_landmarks)} (esperados: 11)")

        missing = expected_landmarks - found_landmarks
        extra = found_landmarks - expected_landmarks
        if missing:
            print(f"  ERROR faltan       : {sorted(missing)}")
            all_ok = False
        if extra:
            print(f"  AVISO inesperados  : {sorted(extra)}")

        if len(queries) != 55:
            all_ok = False

        positives = [q.num_positives for q in queries]
        junks = [len(q.junk) for q in queries]
        with_bbox = sum(1 for q in queries if q.bbox is not None)

        if positives:
            print(
                f"  positivos/query    : min={min(positives)} "
                f"max={max(positives)} media={sum(positives) / len(positives):.1f}"
            )
            print(
                f"  junk/query         : min={min(junks)} "
                f"max={max(junks)} media={sum(junks) / len(junks):.1f}"
            )
        print(f"  queries con bbox   : {with_bbox}/{len(queries)}")

        sin_positivos = [q.query_id for q in queries if q.num_positives == 0]
        if sin_positivos:
            print(f"  ERROR sin positivos: {sin_positivos}")
            all_ok = False

        ejemplo = queries[0]
        print(
            f"  ejemplo            : {ejemplo.query_id} -> imagen "
            f"'{ejemplo.image_id}', bbox={ejemplo.bbox}, "
            f"{ejemplo.num_positives} positivos, {len(ejemplo.junk)} junk"
        )

    print()
    print("Ground truth OK." if all_ok else "Ground truth CON PROBLEMAS (ver arriba).")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Valida que el ground truth este descargado y se parsee bien.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=str(DEFAULT_GT_DIR),
        help="Directorio con el ground truth (default: data/groundtruth).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Que datasets validar (util si Paris aun no se pudo descargar).",
    )
    args = parser.parse_args()

    if args.check:
        ok = check_ground_truth(args.gt_dir, tuple(args.datasets))
        raise SystemExit(0 if ok else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
