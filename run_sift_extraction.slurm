#!/bin/bash

#SBATCH --job-name=sift_extraction
#SBATCH --account=tesis
#SBATCH --partition=standard
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe

set -uo pipefail

# NOTE: igual que en el ejemplo base, no usar $BASH_SOURCE (SLURM copia este
# script a un spool dir y lo corre desde ahi). Se asume que corres
# `sbatch run_sift_extraction.slurm` desde la raiz de proyecto-1/, asi que
# el working directory ya es el correcto.
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

echo "=== Corriendo extraccion SIFT completa (oxford + paris) ==="
srun python 1_feature_extraction_sift.py --workers "$SLURM_CPUS_PER_TASK"
STATUS=$?
echo "Extraccion completa exit code: $STATUS"

echo "=== Done. Features en $REPO_ROOT/data/features/sift/ ==="
exit "$STATUS"
