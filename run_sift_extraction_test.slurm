#!/bin/bash
# Prueba rapida antes de mandar la corrida completa (run_sift_extraction.slurm).
# Procesa solo --limit 20 imagenes por dataset para validar que el entorno,
# dependencias y el script corren bien en la HPC. Revisa el .out/.err y el
# contenido de data/features/sift/ antes de someter el job completo.

#SBATCH --job-name=sift_extraction_test
#SBATCH --account=<TU_ACCOUNT>          # ajustar
#SBATCH --partition=<TU_PARTITION_CPU>  # ajustar: NO usar la particion gpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

REPO_ROOT="$(pwd)"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_MODULE="python/3.11"  # ajustar al modulo real disponible en tu cluster

echo "=== Job $SLURM_JOB_ID on $(hostname), repo: $REPO_ROOT ==="

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

echo "=== Prueba rapida: --limit 20 por dataset ==="
srun python 1_feature_extraction_sift.py --workers "$SLURM_CPUS_PER_TASK" --limit 20
STATUS=$?
echo "Prueba exit code: $STATUS"

echo "=== Si todo salio bien (exit code 0), borra data/features/sift/ y somete run_sift_extraction.slurm ==="
exit "$STATUS"
