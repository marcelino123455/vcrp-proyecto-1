#!/usr/bin/env bash
set -uo pipefail

# Descarga el ground truth oficial de Oxford5k y Paris6k y lo deja en
# data/groundtruth/{oxford,paris}/.
#
# Estos archivos son los que definen el protocolo de evaluacion estandar:
# para cada una de las 55 queries por dataset hay 4 archivos:
#
#   <landmark>_<n>_query.txt   -> imagen consulta + bounding box del objeto
#   <landmark>_<n>_good.txt    -> imagenes claramente del mismo landmark
#   <landmark>_<n>_ok.txt      -> imagenes del landmark, parcialmente visible
#   <landmark>_<n>_junk.txt    -> imagenes ambiguas: se IGNORAN al calcular AP
#
# El conjunto positivo es good + ok. Las junk no cuentan ni como acierto ni
# como error: se eliminan del ranking antes de calcular la precision.
#
# El servidor de Oxford no siempre es alcanzable (en algunas redes y en nodos
# de computo sin salida directa devuelve 403). Por eso Oxford tiene un mirror
# incluido en el repo: data/oxford_gt_mirror.tgz, que ya viene verificado
# (55 queries, 11 landmarks). Paris todavia depende de la descarga.
#
# Uso:
#   ./data/download_groundtruth.sh

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
GT_DIR="$BASE_DIR/groundtruth"
ARCHIVES="$GT_DIR/_archives"

OXFORD_URLS=(
    "https://thor.robots.ox.ac.uk/datasets/oxford-buildings/gt_files_170407.tgz"
    "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz"
)
PARIS_URLS=(
    "https://thor.robots.ox.ac.uk/datasets/paris-buildings/paris_120310.tgz"
    "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz"
)

# Mirror de Oxford incluido en el repo (ver cabecera).
OXFORD_LOCAL_MIRROR="$BASE_DIR/oxford_gt_mirror.tgz"
# Ultimo recurso para Oxford: repositorio publico que versiona los mismos .txt.
OXFORD_GIT_MIRROR="https://github.com/oddconcepts/oxford-benchmark"

mkdir -p "$GT_DIR/oxford" "$GT_DIR/paris" "$ARCHIVES"

try_urls() {
    local out="$1"
    shift

    if [[ -s "$out" ]]; then
        echo "  ya existe, se reutiliza: $(basename "$out")"
        return 0
    fi

    local url
    for url in "$@"; do
        echo "  intentando $url"
        if curl -fsSL --retry 2 --retry-delay 2 --max-time 180 -o "$out" "$url"; then
            echo "  ok"
            return 0
        fi
        rm -f "$out"
    done

    return 1
}

flatten() {
    local dir="$1"
    find "$dir" -mindepth 2 -type f -name "*.txt" -exec mv -t "$dir" {} + 2>/dev/null || true
    find "$dir" -mindepth 1 -depth -type d -empty -delete 2>/dev/null || true
    rm -f "$dir/readme.md" "$dir/README.md" 2>/dev/null || true
}

# --------------------------------------------------------------------------- #
echo "=== Oxford ground truth ==="
OXFORD_OK=0

if try_urls "$ARCHIVES/gt_files_170407.tgz" "${OXFORD_URLS[@]}"; then
    tar -xzf "$ARCHIVES/gt_files_170407.tgz" -C "$GT_DIR/oxford" && OXFORD_OK=1
fi

if [[ "$OXFORD_OK" -eq 0 && -f "$OXFORD_LOCAL_MIRROR" ]]; then
    # El tarball ya trae la carpeta oxford/ en su raiz.
    echo "  descarga fallida; usando el mirror incluido en el repo"
    if tar -xzf "$OXFORD_LOCAL_MIRROR" -C "$GT_DIR"; then
        OXFORD_OK=1
    fi
fi

if [[ "$OXFORD_OK" -eq 0 ]] && command -v git >/dev/null 2>&1; then
    echo "  probando mirror en GitHub"
    TMP_CLONE="$(mktemp -d)"
    if git clone -q --depth 1 "$OXFORD_GIT_MIRROR" "$TMP_CLONE/repo" 2>/dev/null; then
        cp "$TMP_CLONE/repo"/*.txt "$GT_DIR/oxford/" 2>/dev/null && OXFORD_OK=1
    fi
    rm -rf "$TMP_CLONE"
fi

flatten "$GT_DIR/oxford"

# --------------------------------------------------------------------------- #
echo "=== Paris ground truth ==="
PARIS_OK=0

if try_urls "$ARCHIVES/paris_120310.tgz" "${PARIS_URLS[@]}"; then
    tar -xzf "$ARCHIVES/paris_120310.tgz" -C "$GT_DIR/paris" && PARIS_OK=1
fi

flatten "$GT_DIR/paris"

# --------------------------------------------------------------------------- #
OXFORD_QUERIES=$(find "$GT_DIR/oxford" -name "*_query.txt" 2>/dev/null | wc -l)
PARIS_QUERIES=$(find "$GT_DIR/paris" -name "*_query.txt" 2>/dev/null | wc -l)

echo ""
echo "Oxford: $OXFORD_QUERIES queries en $GT_DIR/oxford"
echo "Paris:  $PARIS_QUERIES queries en $GT_DIR/paris"
echo ""

STATUS=0

if [[ "$OXFORD_QUERIES" -ne 55 ]]; then
    echo "AVISO: Oxford deberia tener 55 queries." >&2
    STATUS=1
fi

if [[ "$PARIS_QUERIES" -ne 55 ]]; then
    echo "AVISO: Paris deberia tener 55 queries y no se pudo descargar." >&2
    echo "  Bajalo a mano desde:" >&2
    echo "    https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/" >&2
    echo "  y descomprime paris_120310.tgz en $GT_DIR/paris/" >&2
    echo "  Mientras tanto puedes evaluar solo Oxford:" >&2
    echo "    python 4_evaluate_retrieval.py --embeddings ... --datasets oxford" >&2
    STATUS=1
fi

echo "Verifica el parseo con:"
echo "  python evaluation.py --check"

exit "$STATUS"
