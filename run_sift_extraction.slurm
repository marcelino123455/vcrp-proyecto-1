#!/bin/bash

#SBATCH --job-name=sift_extraction
#SBATCH --account=<TU_ACCOUNT>          # ajustar
#SBATCH --partition=<TU_PARTITION_CPU>  # ajustar: NO usar la particion gpu, SIFT (cv2) corre solo en CPU
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_usuario>@utec.edu.pe

set -uo pipefail

# NOTE: igual que en el ejemplo base, no usar $BASH_SOURCE (SLURM copia este
# script a un spool dir y lo corre desde ahi). Se asume que corres
# `sbatch run_sift_extraction.slurm` desde la raiz de proyecto-1/, asi que
# el working directory ya es el correcto.
REPO_ROOT="$(pwd)"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_MODULE="python/3.11"  # ajustar al modulo real disponible en tu cluster (ver `module avail python`)

echo "=== Job $SLURM_JOB_ID on $(hostname), repo: $REPO_ROOT ==="

# Lmod puede referenciar variables no definidas y romper con `set -u`.
set +u
module purge
module load "$PYTHON_MODULE" 2>/dev/null || echo "Aviso: no se pudo cargar modulo $PYTHON_MODULE, se usa python del sistema"
set -u

if [[ ! -d "$VENV_DIR" ]]; then
  echo "=== Creando venv en $VENV_DIR ==="
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "=== Python activo: $(which python) ==="

pip install --quiet -r "$REPO_ROOT/requirements.txt"

echo "=== Corriendo extraccion SIFT completa (oxford + paris) ==="
srun python 1_feature_extraction_sift.py --workers "$SLURM_CPUS_PER_TASK"
STATUS=$?
echo "Extraccion completa exit code: $STATUS"

echo "=== Done. Features en $REPO_ROOT/data/features/sift/ ==="
exit "$STATUS"
