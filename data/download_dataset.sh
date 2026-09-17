#!/usr/bin/env bash
set -euo pipefail

# Descarga el dataset "skylord/oxbuildings" desde Kaggle usando la API REST,
# sin depender de Python/kagglehub.
#
# Credenciales: exporta KAGGLE_USERNAME y KAGGLE_KEY, o crea ~/.kaggle/kaggle.json
# (Kaggle > Account > Create New Token) con formato:
#   {"username":"tu_usuario","key":"tu_api_key"}

DATASET="skylord/oxbuildings"
DEST_DIR="$(dirname "$0")/oxbuildings"
ZIP_FILE="/tmp/oxbuildings.zip"

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    KAGGLE_JSON="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
    if [[ -f "$KAGGLE_JSON" ]]; then
        KAGGLE_USERNAME=$(jq -r '.username' "$KAGGLE_JSON")
        KAGGLE_KEY=$(jq -r '.key' "$KAGGLE_JSON")
    else
        echo "Error: no se encontraron credenciales de Kaggle." >&2
        echo "Exporta KAGGLE_USERNAME y KAGGLE_KEY, o coloca tu token en $KAGGLE_JSON" >&2
        exit 1
    fi
fi

mkdir -p "$DEST_DIR"

echo "Descargando dataset $DATASET..."
curl -L -u "${KAGGLE_USERNAME}:${KAGGLE_KEY}" \
    -o "$ZIP_FILE" \
    "https://www.kaggle.com/api/v1/datasets/download/${DATASET}"

echo "Descomprimiendo en $DEST_DIR..."
unzip -o "$ZIP_FILE" -d "$DEST_DIR"
rm -f "$ZIP_FILE"

echo "Listo. Archivos en: $DEST_DIR"
