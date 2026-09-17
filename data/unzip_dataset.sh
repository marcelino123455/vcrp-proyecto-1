#!/usr/bin/env bash
set -euo pipefail

# Descomprime los .tgz descargados de oxbuildings y separa las imagenes
# en dos carpetas: oxford/ y paris/. Al final cuenta cuantas imagenes
# quedaron en cada una.

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$BASE_DIR/oxbuildings"
OXFORD_DIR="$BASE_DIR/oxford"
PARIS_DIR="$BASE_DIR/paris"

mkdir -p "$OXFORD_DIR" "$PARIS_DIR"

echo "Extrayendo Oxford..."
tar -xzf "$SRC_DIR/oxbuild_images.tgz" -C "$OXFORD_DIR"

echo "Extrayendo Paris (parte 1)..."
tar -xzf "$SRC_DIR/paris_1.tgz" -C "$PARIS_DIR"

echo "Extrayendo Paris (parte 2)..."
tar -xzf "$SRC_DIR/paris_2.tgz" -C "$PARIS_DIR"

# Los tgz de Paris traen las imagenes anidadas en paris/<monumento>/*.jpg.
# Como el nombre de archivo ya identifica el monumento (paris_defense_*,
# paris_eiffel_*, etc.), las aplanamos directo en $PARIS_DIR.
echo "Aplanando estructura de Paris..."
find "$PARIS_DIR" -mindepth 2 -type f -iname "*.jpg" -exec mv -t "$PARIS_DIR" {} +
find "$PARIS_DIR" -mindepth 1 -depth -type d -empty -delete

OXFORD_COUNT=$(find "$OXFORD_DIR" -type f -iname "*.jpg" | wc -l)
PARIS_COUNT=$(find "$PARIS_DIR" -type f -iname "*.jpg" | wc -l)

echo ""
echo "Oxford: $OXFORD_COUNT imagenes en $OXFORD_DIR"
echo "Paris:  $PARIS_COUNT imagenes en $PARIS_DIR"
