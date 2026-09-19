#!/usr/bin/env bash
set -uo pipefail

# Descarga los descriptores SIFT ya extraidos desde el dataset publico de Kaggle
# y los deja en data/features/sift/.
#
#   https://www.kaggle.com/datasets/marcelinomaitavargas/vcrp-sift-features
#
# Evita repetir la extraccion completa (horas de CPU). Son ~7.3 GB comprimidos,
# asi que corre esto desde el NODO DE ACCESO de Khipu: los nodos de computo
# normalmente no tienen salida a internet.
#
# Credenciales: exporta KAGGLE_USERNAME y KAGGLE_KEY, o coloca tu token en
# ~/.kaggle/kaggle.json (Kaggle > Account > Create New Token).
#
# Uso:
#   ./data/download_sift_features.sh
#   KEEP_ZIP=1 ./data/download_sift_features.sh    # conserva el zip descargado

DATASET="marcelinomaitavargas/vcrp-sift-features"

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$BASE_DIR/features/sift"
TMP_DIR="${TMPDIR:-/tmp}"
ZIP_FILE="$TMP_DIR/vcrp-sift-features.zip"

H5_FILE="$DEST_DIR/sift_features.h5"
CSV_FILE="$DEST_DIR/sift_index.csv"

# El zip pesa ~7.3 GB y al descomprimir ocupa otro tanto.
REQUIRED_GB=17

# --------------------------------------------------------------------------- #
if [[ -f "$H5_FILE" && -f "$CSV_FILE" ]]; then
    echo "Los features ya estan en $DEST_DIR:"
    ls -lh "$H5_FILE" "$CSV_FILE"
    echo ""
    echo "Si quieres volver a bajarlos, borra esos archivos primero."
    exit 0
fi

# --------------------------------------------------------------------------- #
# Credenciales
if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    KAGGLE_JSON="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
    if [[ -f "$KAGGLE_JSON" ]]; then
        if command -v jq >/dev/null 2>&1; then
            KAGGLE_USERNAME=$(jq -r '.username' "$KAGGLE_JSON")
            KAGGLE_KEY=$(jq -r '.key' "$KAGGLE_JSON")
        else
            # Fallback sin jq (no siempre esta instalado en un cluster).
            KAGGLE_USERNAME=$(sed -n 's/.*"username"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$KAGGLE_JSON")
            KAGGLE_KEY=$(sed -n 's/.*"key"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$KAGGLE_JSON")
        fi
    fi
fi

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    echo "Error: no se encontraron credenciales de Kaggle." >&2
    echo "  export KAGGLE_USERNAME=tu_usuario" >&2
    echo "  export KAGGLE_KEY=tu_api_key" >&2
    echo "o coloca el token en ~/.kaggle/kaggle.json" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Espacio en disco
available_gb() {
    df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9' \
        || df -g "$1" 2>/dev/null | tail -1 | awk '{print $4}'
}

for dir in "$TMP_DIR" "$BASE_DIR"; do
    avail=$(available_gb "$dir")
    if [[ -n "$avail" && "$avail" -lt "$REQUIRED_GB" ]]; then
        echo "AVISO: solo ${avail} GB libres en $dir (se recomiendan ${REQUIRED_GB} GB)." >&2
        echo "  Puedes apuntar el temporal a otro disco:  TMPDIR=/ruta/con/espacio $0" >&2
    fi
done

mkdir -p "$DEST_DIR"

# --------------------------------------------------------------------------- #
echo "=== Descargando $DATASET (~7.3 GB) ==="
echo "    destino temporal: $ZIP_FILE"
echo "    (curl reanuda si la descarga se corta; vuelve a correr el script)"
echo ""

# -C - reanuda una descarga parcial; --retry reintenta cortes de red.
if ! curl -L --fail --retry 5 --retry-delay 5 --retry-all-errors -C - \
        -u "${KAGGLE_USERNAME}:${KAGGLE_KEY}" \
        -o "$ZIP_FILE" \
        "https://www.kaggle.com/api/v1/datasets/download/${DATASET}"; then
    echo "" >&2
    echo "Error: fallo la descarga." >&2
    echo "  - Si el error es 403, revisa que el token de Kaggle sea valido." >&2
    echo "  - Si es 404, revisa que el dataset siga siendo publico." >&2
    echo "  - El archivo parcial quedo en $ZIP_FILE; vuelve a correr para reanudar." >&2
    exit 1
fi

# Kaggle a veces responde una pagina HTML de error con codigo 200.
if ! unzip -tq "$ZIP_FILE" >/dev/null 2>&1; then
    echo "Error: lo descargado no es un zip valido (probablemente una pagina de error)." >&2
    echo "  Borra $ZIP_FILE y revisa tus credenciales." >&2
    head -c 200 "$ZIP_FILE" >&2
    exit 1
fi

echo ""
echo "=== Descomprimiendo en $DEST_DIR ==="
unzip -o "$ZIP_FILE" -d "$DEST_DIR"

if [[ "${KEEP_ZIP:-0}" != "1" ]]; then
    rm -f "$ZIP_FILE"
    echo "    zip temporal eliminado (usa KEEP_ZIP=1 para conservarlo)"
fi

# --------------------------------------------------------------------------- #
echo ""
if [[ -f "$H5_FILE" && -f "$CSV_FILE" ]]; then
    ls -lh "$H5_FILE" "$CSV_FILE"
    echo ""
    echo "Imagenes en el indice: $(( $(wc -l < "$CSV_FILE") - 1 ))"
    echo ""
    echo "Siguiente paso:"
    echo "  python 2_visual_vocabulary.py --vocab-size 1000 --rootsift"
else
    echo "AVISO: no se encontraron los archivos esperados en $DEST_DIR" >&2
    ls -la "$DEST_DIR" >&2
    exit 1
fi
