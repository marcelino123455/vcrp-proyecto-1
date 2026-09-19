#!/bin/bash
# Evaluacion de una representacion sobre Oxford5k / Paris6k.
#
# Uso:
#   sbatch run_evaluation.slurm <ruta_embeddings.npz> [nombre_experimento]
#
# Ejemplo:
#   sbatch run_evaluation.slurm data/features/vocabulary/image_histograms.npz bovw_k500
#
# Igual que los otros .slurm del repo, asume que se somete desde la raiz del
# proyecto (SLURM copia el script a un spool dir, asi que $BASH_SOURCE no sirve).

#SBATCH --job-name=retrieval_eval
#SBATCH --account=tesis
#SBATCH --partition=standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

REPO_ROOT="$(pwd)"
ENV_NAME="vcrp-proyecto-1"

EMBEDDINGS="${1:-}"
RUN_NAME="${2:-}"

if [[ -z "$EMBEDDINGS" ]]; then
    echo "Uso: sbatch run_evaluation.slurm <ruta_embeddings.npz> [nombre_experimento]" >&2
    exit 2
fi

if [[ ! -f "$EMBEDDINGS" ]]; then
    echo "No existe el archivo de embeddings: $EMBEDDINGS" >&2
    exit 2
fi

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="$(basename "${EMBEDDINGS%.*}")"
fi

echo "=== Job $SLURM_JOB_ID on $(hostname), repo: $REPO_ROOT ==="
echo "=== Embeddings: $EMBEDDINGS  |  run: $RUN_NAME ==="

set +u
module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

echo "=== Python activo: $(which python) ==="

# Limita el multihilo de BLAS a los cores reservados: por defecto numpy abre
# tantos hilos como cores tenga el NODO, no el job, y en un cluster compartido
# eso degrada a todo el mundo (incluido este job).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

if [[ ! -d "$REPO_ROOT/data/groundtruth/oxford" ]]; then
    echo "=== Ground truth ausente, descargando ==="
    ./data/download_groundtruth.sh
fi

echo "=== Sanity check del ground truth ==="
srun python evaluation.py --check
if [[ $? -ne 0 ]]; then
    echo "El ground truth no paso la validacion. Se aborta." >&2
    exit 1
fi

echo "=== Tests del harness ==="
srun python test_evaluation.py
if [[ $? -ne 0 ]]; then
    echo "Los tests del harness fallaron. Se aborta (no reportes un mAP roto)." >&2
    exit 1
fi

echo "=== Barrido de normalizaciones x metricas ==="
srun python 4_evaluate_retrieval.py \
    --embeddings "$EMBEDDINGS" \
    --name "$RUN_NAME" \
    --normalize none l1 l2 power hellinger \
    --metric cosine l2 chi2 hellinger intersection \
    --database same \
    --save-rankings \
    --quiet
STATUS=$?

echo "=== Mejor configuracion + analisis cualitativo ==="
srun python 4_evaluate_retrieval.py \
    --embeddings "$EMBEDDINGS" \
    --name "${RUN_NAME}_best" \
    --normalize l2 \
    --metric cosine \
    --database same \
    --qualitative 6 \
    --qualitative-topk 5

echo "Exit code del barrido: $STATUS"
echo "=== Done. Resultados en $REPO_ROOT/results/ ==="
exit "$STATUS"
