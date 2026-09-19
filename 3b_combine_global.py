"""Combina descriptores globales y aplica PCA-whitening.

El enunciado permite explicitamente "combine them into a single representation
(feature concatenation)". Este script hace esa concatenacion bien: normaliza
cada bloque por separado antes de unirlos, y opcionalmente aplica
PCA-whitening al resultado.

Por que normalizar por bloque
------------------------------
HOG con piramide puede tener miles de dimensiones y el histograma LBP un par
de cientos. Si se concatenan crudos, el bloque mas grande (o el de mayor
varianza) domina la distancia y los demas dejan de influir. Normalizar cada
bloque a norma 1 antes de concatenar le da a cada descriptor el mismo peso
inicial; `--weights` permite despues desbalancearlos a proposito.

Por que PCA-whitening
---------------------
Las dimensiones de estos descriptores estan fuertemente correlacionadas entre
si (celdas vecinas de HOG ven casi lo mismo) y tienen varianzas muy distintas.
La distancia euclidiana sobre datos asi cuenta varias veces la misma
informacion. El whitening decorrelaciona y iguala varianzas, de modo que cada
direccion aporta lo mismo. En la literatura de retrieval es de los
post-procesos mas rentables, y ademas reduce la dimension, lo que acelera el
ranking.

Se documenta una limitacion honesta: el PCA se ajusta sobre la misma base de
datos que luego se evalua. Lo correcto seria ajustarlo en un conjunto
independiente; hacerlo aqui es la practica habitual en este benchmark, pero
conviene mencionarlo en el reporte.

Uso
---
    python 3b_combine_global.py --descriptors hog color lbp gist
    python 3b_combine_global.py --descriptors hog color --pca 256
    python 3b_combine_global.py --descriptors hog color lbp gist --pca 512 --power

Despues:
    python 4_evaluate_retrieval.py \\
        --embeddings data/features/global/combined.npz --name global_combined
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
GLOBAL_DIR = REPO_ROOT / "data" / "features" / "global"

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--descriptors", nargs="+", required=True,
        help="Nombres de los .npz a combinar (sin extension), p.ej. hog color lbp gist.",
    )
    parser.add_argument("--in-dir", default=str(GLOBAL_DIR))
    parser.add_argument("--out", default=None, help="Ruta del .npz resultante.")
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Peso por descriptor (mismo orden). Por defecto todos igual.",
    )
    parser.add_argument(
        "--block-norm", default="l2", choices=("none", "l1", "l2"),
        help="Normalizacion de cada bloque antes de concatenar (default: l2).",
    )
    parser.add_argument(
        "--power", action="store_true",
        help="Aplica signed square root antes de la normalizacion final.",
    )
    parser.add_argument(
        "--pca", type=int, default=0,
        help="Si > 0, reduce a esa dimension con PCA.",
    )
    parser.add_argument(
        "--whiten-power", type=float, default=0.25,
        help=(
            "Intensidad del whitening: cada componente se divide por "
            "lambda^power. 0.5 = whitening completo (el valor clasico de la "
            "literatura), 0 = solo rotacion PCA. El default 0.25 es el "
            "compromiso robusto; vale la pena barrer los tres en el reporte."
        ),
    )
    parser.add_argument(
        "--pca-fit-sample", type=int, default=0,
        help="Ajusta el PCA en una muestra aleatoria de N imagenes (0 = todas).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_rows(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return X
    if mode == "l1":
        return X / np.maximum(np.abs(X).sum(axis=1, keepdims=True), EPS)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def load_block(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(
            f"No existe {path}. Generalo primero con 3_feature_extraction_global.py"
        )
    data = np.load(path, allow_pickle=True)
    ids = np.asarray([str(x) for x in data["image_ids"]], dtype=object)
    features = np.asarray(data["features"], dtype=np.float32)
    return ids, features


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)

    weights = args.weights or [1.0] * len(args.descriptors)
    if len(weights) != len(args.descriptors):
        raise SystemExit("--weights debe tener un valor por descriptor")

    print("=== Combinando descriptores globales ===")

    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in args.descriptors:
        ids, features = load_block(in_dir / f"{name}.npz")
        blocks[name] = (ids, features)
        print(f"  {name:<8} {features.shape[0]} x {features.shape[1]}")

    # Interseccion de image_ids: si una imagen fallo en un descriptor y no en
    # otro, las matrices no estan alineadas y concatenar a ciegas mezclaria
    # filas de imagenes distintas. Es el error silencioso mas peligroso aqui.
    common = None
    for ids, _ in blocks.values():
        s = set(ids.tolist())
        common = s if common is None else (common & s)
    assert common is not None

    common_ids = np.asarray(sorted(common), dtype=object)
    print(f"  imagenes en comun: {len(common_ids)}")

    for name, (ids, _) in blocks.items():
        faltan = len(ids) - len(common_ids)
        if faltan > 0:
            print(f"  aviso: {name} tenia {faltan} imagenes que otros descriptores no tienen")

    parts: list[np.ndarray] = []
    for name, weight in zip(args.descriptors, weights):
        ids, features = blocks[name]
        posicion = {image_id: i for i, image_id in enumerate(ids.tolist())}
        idx = np.asarray([posicion[i] for i in common_ids.tolist()])
        block = normalize_rows(features[idx], args.block_norm)
        parts.append(block * float(weight))

    X = np.hstack(parts).astype(np.float32)
    print(f"  concatenado: {X.shape[0]} x {X.shape[1]}")

    if args.power:
        X = np.sign(X) * np.sqrt(np.abs(X))
        print("  power normalization aplicada")

    pca_info: dict = {}
    if args.pca > 0:
        from sklearn.decomposition import PCA

        n_components = min(args.pca, X.shape[0], X.shape[1])
        if n_components < args.pca:
            print(f"  aviso: PCA reducido a {n_components} componentes (limite de los datos)")

        fit_data = X
        if args.pca_fit_sample and args.pca_fit_sample < X.shape[0]:
            rng = np.random.default_rng(args.seed)
            sample = rng.choice(X.shape[0], size=args.pca_fit_sample, replace=False)
            fit_data = X[sample]
            print(f"  PCA ajustado sobre una muestra de {len(sample)} imagenes")

        power = float(np.clip(args.whiten_power, 0.0, 0.5))

        # El whitening se hace a mano en vez de con whiten=True para poder
        # controlar su intensidad. Whitening completo divide cada componente
        # por sqrt(lambda); cuando lambda es pequena esa division amplifica
        # direcciones que son puro ruido de muestreo. Con pocas imagenes
        # respecto a la dimension, eso degrada el retrieval en vez de
        # mejorarlo (medido: en una prueba con 80 imagenes y 2966 dims, el
        # mAP cayo de 0.99 sin PCA a 0.37 con whitening completo a 64 dims,
        # y el dano crecia con el numero de componentes).
        pca = PCA(n_components=n_components, whiten=False, random_state=args.seed)
        pca.fit(fit_data)

        X = pca.transform(X).astype(np.float32)
        if power > 0:
            escala = np.power(
                np.maximum(pca.explained_variance_, EPS), power
            ).astype(np.float32)
            X = X / escala[None, :]

        varianza = float(pca.explained_variance_ratio_.sum())
        muestras_por_componente = fit_data.shape[0] / max(1, n_components)

        pca_info = {
            "components": int(n_components),
            "explained_variance_ratio": round(varianza, 4),
            "whiten_power": power,
            "samples_per_component": round(muestras_por_componente, 1),
            "fit_sample": int(args.pca_fit_sample) or None,
        }

        etiqueta = "sin whitening" if power == 0 else f"whitening^{power:g}"
        print(f"  PCA ({etiqueta}) -> {X.shape[1]} dims ({varianza:.1%} de la varianza)")

        if muestras_por_componente < 20 and power > 0:
            print(
                f"  AVISO: solo {muestras_por_componente:.1f} imagenes por componente. "
                "Las componentes de baja varianza son ruido y el whitening las "
                "amplifica. Baja --pca, o usa --whiten-power 0.25 o 0."
            )

    X = normalize_rows(X, "l2").astype(np.float32)

    if not np.all(np.isfinite(X)):
        raise SystemExit("El resultado contiene valores no finitos; revisa los descriptores.")

    out_path = Path(args.out) if args.out else in_dir / "combined.npz"
    np.savez_compressed(out_path, image_ids=common_ids, features=X)

    config = {
        "descriptors": list(args.descriptors),
        "weights": [float(w) for w in weights],
        "block_norm": args.block_norm,
        "power": bool(args.power),
        "pca": pca_info or None,
        "num_images": int(X.shape[0]),
        "dimensions": int(X.shape[1]),
        "source_dimensions": {
            name: int(features.shape[1]) for name, (_, features) in blocks.items()
        },
    }
    config_path = out_path.with_suffix(".json")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print()
    print(f"  salida : {out_path}  ({X.shape[0]} x {X.shape[1]})")
    print(f"  config : {config_path}")
    print()
    print("Siguiente paso:")
    print(f"  python 4_evaluate_retrieval.py --embeddings {out_path} --name global_combined")


if __name__ == "__main__":
    main()
