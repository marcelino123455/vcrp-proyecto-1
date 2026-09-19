#!/bin/bash
# Construye el vocabulario visual (BoVW) y evalua el resultado.
#
# Uso:
#   sbatch run_vocabulary.slurm [VOCAB_SIZE] [SAMPLE_SIZE]
#
# Ejemplos:
#   sbatch run_vocabulary.slurm                 # K=1000, muestra de 1M (primera corrida)
#   sbatch run_vocabulary.slurm 20000 4000000   # vocabulario grande
#
# Requisitos previos (desde el NODO DE ACCESO, que si tiene internet):
#   ./setup.sh
#   ./data/download_sift_features.sh
#   ./data/download_groundtruth.sh

#SBATCH --job-name=bovw_vocab
#SBATCH --account=tesis
#SBATCH --partition=standard
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alvaro.garcia.h@utec.edu.pe

set -uo pipefail

REPO_ROOT="$(pwd)"
ENV_NAME="vcrp-proyecto-1"

VOCAB_SIZE="${1:-1000}"
SAMPLE_SIZE="${2:-1000000}"
RUN_NAME="bovw_k${VOCAB_SIZE}"

echo "=== Job $SLURM_JOB_ID on $(hostname), repo: $REPO_ROOT ==="
echo "=== K=$VOCAB_SIZE  sample=$SAMPLE_SIZE  run=$RUN_NAME ==="

set +u
module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

echo "=== Python activo: $(which python) ==="

# numpy/sklearn abren por defecto tantos hilos como cores tenga el NODO, no el
# job. En un cluster compartido eso perjudica a todos, incluido este job.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

H5="data/features/sift/sift_features.h5"
if [[ ! -f "$H5" ]]; then
    echo "No existe $H5." >&2
    echo "Bajalo desde el nodo de acceso con ./data/download_sift_features.sh" >&2
    exit 1
fi

if [[ ! -d "data/groundtruth/oxford" ]]; then
    echo "Falta el ground truth; intentando con el mirror incluido en el repo."
    bash data/download_groundtruth.sh || true
fi

echo ""
echo "=== 1/3 Vocabulario visual ==="
srun python 2_visual_vocabulary.py \
    --vocab-size "$VOCAB_SIZE" \
    --sample-size "$SAMPLE_SIZE" \
    --rootsift \
    --tfidf \
    --normalize l1 \
    --suffix "k${VOCAB_SIZE}" \
    --overwrite
STATUS=$?

if [[ "$STATUS" -ne 0 ]]; then
    echo "Fallo la construccion del vocabulario (exit $STATUS)." >&2
    exit "$STATUS"
fi

EMB="data/features/vocabulary/image_histograms_k${VOCAB_SIZE}.npz"

echo ""
echo "=== 2/3 Barrido de normalizacion x metrica ==="
srun python 4_evaluate_retrieval.py \
    --embeddings "$EMB" \
    --name "$RUN_NAME" \
    --normalize none l1 l2 power hellinger \
    --metric cosine l2 chi2 hellinger intersection \
    --datasets oxford \
    --save-rankings \
    --quiet

echo ""
echo "=== 3/3 Analisis cualitativo de la mejor configuracion ==="
srun python 4_evaluate_retrieval.py \
    --embeddings "$EMB" \
    --name "${RUN_NAME}_best" \
    --normalize l2 \
    --metric cosine \
    --datasets oxford \
    --qualitative 6

echo ""
echo "=== Done ==="
echo "  histogramas : $EMB"
echo "  resultados  : $REPO_ROOT/results/${RUN_NAME}_comparison.csv"
exit 0
