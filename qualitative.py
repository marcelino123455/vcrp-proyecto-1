"""Analisis cualitativo: grids de top-k con aciertos y fallos marcados.

El enunciado pide explicitamente visualizar los top-5 de un subconjunto de
queries y discutir casos de exito y de fallo. Este modulo genera esas figuras
automaticamente a partir de los rankings ya calculados, eligiendo las N mejores
y las N peores queries — que son justamente las que dan material para la
discusion.

Codigo de colores del borde:
    verde  -> positivo (good u ok)
    gris   -> junk (el protocolo la ignora al calcular AP)
    rojo   -> falso positivo
    azul   -> la propia imagen consulta
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display: necesario en nodos de computo

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BORDER_POSITIVE = "#2e7d32"
BORDER_JUNK = "#9e9e9e"
BORDER_NEGATIVE = "#c62828"
BORDER_QUERY = "#1565c0"
BORDER_WIDTH = 5

SEARCH_DATASETS = ("oxford", "paris")


def resolve_image_path(image_root: Path, image_id: str, dataset_hint: str | None = None) -> Path | None:
    """Encuentra el .jpg de una imagen probando en los directorios conocidos."""
    candidates = []
    if dataset_hint:
        candidates.append(image_root / dataset_hint / f"{image_id}.jpg")
    for dataset in SEARCH_DATASETS:
        candidates.append(image_root / dataset / f"{image_id}.jpg")
    # Por si el equipo movio las corruptas con repair_dataset.py
    for dataset in SEARCH_DATASETS:
        candidates.append(image_root / "bad" / dataset / f"{image_id}.jpg")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_rgb(path: Path, max_side: int = 320) -> np.ndarray | None:
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _placeholder(text: str = "sin imagen") -> np.ndarray:
    img = np.full((240, 320, 3), 240, dtype=np.uint8)
    return img


def _classify(image_id: str, query) -> tuple[str, str]:
    if image_id == query.image_id:
        return BORDER_QUERY, "query"
    if image_id in query.junk:
        return BORDER_JUNK, "junk"
    if image_id in query.positives:
        return BORDER_POSITIVE, "ok"
    return BORDER_NEGATIVE, "fallo"


def save_topk_grid(
    query,
    ranked: list[str],
    image_root: Path,
    out_path: Path,
    top_k: int = 5,
    ap: float | None = None,
    skip_junk: bool = True,
) -> Path:
    """Guarda una figura: imagen consulta + top-k recuperadas.

    Con `skip_junk=True` se muestran los top-k *tal como los evalua el
    protocolo*, es decir, saltando las junk. Ponlo en False para ver el
    ranking crudo, que a veces explica mejor un fallo.
    """
    shown: list[str] = []
    for image_id in ranked:
        if skip_junk and image_id in query.junk:
            continue
        shown.append(image_id)
        if len(shown) == top_k:
            break

    n_cols = len(shown) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.6))
    if n_cols == 1:
        axes = [axes]

    # Panel de la query
    query_path = resolve_image_path(image_root, query.image_id, query.dataset)
    query_img = _load_rgb(query_path) if query_path else None
    axes[0].imshow(query_img if query_img is not None else _placeholder())
    axes[0].set_title(f"QUERY\n{query.image_id}", fontsize=9, color=BORDER_QUERY)
    _style_axis(axes[0], BORDER_QUERY)

    # Si el ground truth trae bounding box, dibujarlo: explica muchos fallos
    # (la query suele ser un recorte del landmark, no la foto completa).
    if query.bbox is not None and query_img is not None and query_path is not None:
        _draw_bbox(axes[0], query, query_path, query_img)

    for i, image_id in enumerate(shown, start=1):
        color, label = _classify(image_id, query)
        path = resolve_image_path(image_root, image_id, query.dataset)
        img = _load_rgb(path) if path else None
        axes[i].imshow(img if img is not None else _placeholder())
        axes[i].set_title(f"#{i} {label}\n{image_id}", fontsize=8, color=color)
        _style_axis(axes[i], color)

    title = f"{query.dataset} / {query.query_id}"
    if ap is not None:
        title += f"   —   AP = {ap:.4f}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _style_axis(ax, color: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(BORDER_WIDTH)


def _draw_bbox(ax, query, query_path: Path, shown_img: np.ndarray) -> None:
    """Dibuja el bbox del ground truth, reescalado al tamano mostrado."""
    try:
        import cv2
        from matplotlib.patches import Rectangle

        original = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
        if original is None:
            return
        oh, ow = original.shape[:2]
        sh, sw = shown_img.shape[:2]
        sx, sy = sw / ow, sh / oh

        x1, y1, x2, y2 = query.bbox
        ax.add_patch(
            Rectangle(
                (x1 * sx, y1 * sy),
                (x2 - x1) * sx,
                (y2 - y1) * sy,
                fill=False,
                edgecolor="#fdd835",
                linewidth=2.0,
                linestyle="--",
            )
        )
    except Exception:
        return


def save_qualitative_grids(
    result,
    rankings: dict[str, list[str]],
    gt: dict[str, list],
    image_root: Path,
    out_dir: Path,
    n_queries: int = 5,
    top_k: int = 5,
) -> list[Path]:
    """Genera grids para las N mejores y N peores queries de un experimento."""
    by_id = {}
    for dataset, queries in gt.items():
        for query in queries:
            by_id[f"{dataset}::{query.query_id}"] = query

    selected: list[tuple[str, object, float]] = []
    seen: set[str] = set()

    for tag, items in (("peor", result.worst_queries(n_queries)), ("mejor", result.best_queries(n_queries))):
        for qr in items:
            if qr.query_id in seen:
                continue
            seen.add(qr.query_id)
            selected.append((tag, qr, qr.ap))

    paths: list[Path] = []
    for tag, qr, ap in selected:
        query = by_id.get(qr.query_id)
        ranked = rankings.get(qr.query_id)
        if query is None or ranked is None:
            continue

        safe = qr.query_id.replace("::", "_")
        out_path = out_dir / f"{tag}_{ap:.3f}_{safe}.png"
        paths.append(
            save_topk_grid(
                query=query,
                ranked=ranked,
                image_root=image_root,
                out_path=out_path,
                top_k=top_k,
                ap=ap,
            )
        )

    return paths
