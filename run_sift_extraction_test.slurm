#!/bin/bash
# Prueba rapida antes de mandar la corrida completa (run_sift_extraction.slurm).
# Procesa solo --limit 20 imagenes por dataset para validar que el entorno,
# dependencias y el script corren bien en la HPC. Revisa el .out/.err y el
# contenido de data/features/sift/ antes de someter el job completo.

#SBATCH --job-name=sift_extraction_test
#SBATCH --account=tesis          # ajustar
#SBATCH --partition=standard  # pruebas rapidas en CPU, max 30 min (Khipu)
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

REPO_ROOT="$(pwd)"
ENV_NAME="vcrp-proyecto-1"

echo "=== Job $SLURM_JOB_ID on $(hostname), repo: $REPO_ROOT ==="

# Lmod y conda pueden referenciar variables no definidas y romper con `set -u`.
set +u
module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

echo "=== Python activo: $(which python) ==="

echo "=== Prueba rapida: --limit 20 por dataset ==="
srun python 1_feature_extraction_sift.py --workers "$SLURM_CPUS_PER_TASK" --limit 20
STATUS=$?
echo "Prueba exit code: $STATUS"

echo "=== Si todo salio bien (exit code 0), borra data/features/sift/ y somete run_sift_extraction.slurm ==="
exit "$STATUS"
