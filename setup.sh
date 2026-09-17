#!/usr/bin/env bash
set -euo pipefail

# Crea (si no existe) un ambiente conda e instala requirements.txt.
# Uso: ./setup.sh   (ejecutar desde el nodo de acceso de Khipu)

ENV_NAME="vcrp-proyecto-1"
PYTHON_VERSION="3.11"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== Cargando modulo Miniconda ==="
set +u
module load miniconda/3.0
eval "$(conda shell.bash hook)"
set -u

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "=== Ambiente '$ENV_NAME' ya existe, se reutiliza ==="
else
    echo "=== Creando ambiente conda '$ENV_NAME' (python $PYTHON_VERSION) ==="
    conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

set +u
conda activate "$ENV_NAME"
set -u
echo "=== Python activo: $(which python) ==="

echo "=== Instalando dependencias de requirements.txt ==="
pip install -r "$REPO_ROOT/requirements.txt"

echo ""
echo "Listo. Para usar el ambiente en otra sesion o en un .slurm:"
echo "  module load miniconda/3.0"
echo "  eval \"\$(conda shell.bash hook)\""
echo "  conda activate $ENV_NAME"
