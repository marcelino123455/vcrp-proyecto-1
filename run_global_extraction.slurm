#!/bin/bash
# Extraccion de descriptores globales + evaluacion de cada uno.
#
# Uso:
#   sbatch run_global_extraction.slurm [IMAGE_SIZE] [PYRAMID]
#
# Ejemplos:
#   sbatch run_global_extraction.slurm            # 128 px, piramide 2 niveles
#   sbatch run_global_extraction.slurm 160 3      # mas resolucion y 4x4
#
# Requisitos previos (desde el nodo de acceso):
#   ./setup.sh
#   ./data/download_dataset.sh && ./data/unzip_dataset.sh   # las IMAGENES
#   ./data/download_groundtruth.sh
#
# Ojo: esto necesita las imagenes .jpg, no el sift_features.h5.

#SBATCH --job-name=global_features
#SBATCH --account=tesis
#SBATCH --partition=standard
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alvaro.garcia.h@utec.edu.pe

set -uo pipefail

REPO_ROOT="$(pwd)"
ENV_NAME="vcrp-proyecto-1"

IMAGE_SIZE="${1:-128}"
PYRAMID="${2:-2}"
SUFFIX=""

echo "=== Job $SLURM_JOB_ID on $(hostname), repo: $REPO_ROOT ==="
echo "=== image_size=$IMAGE_SIZE  pyramid=$PYRAMID ==="

set +u
module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

echo "=== Python activo: $(which python) ==="

# El paralelismo lo da el Pool de procesos; que ademas cada worker abra hilos
# de BLAS solo genera contencion.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ ! -d "data/oxford" ]]; then
    echo "No existe data/oxford. Los descriptores globales necesitan las IMAGENES." >&2
    echo "Bajalas desde el nodo de acceso:" >&2
    echo "  ./data/download_dataset.sh && ./data/unzip_dataset.sh" >&2
    exit 1
fi

if [[ ! -d "data/groundtruth/oxford" ]]; then
    bash data/download_groundtruth.sh || true
fi

echo ""
echo "=== 1/3 Extraccion de la familia completa ==="
srun python 3_feature_extraction_global.py \
    --workers "$SLURM_CPUS_PER_TASK" \
    --image-size "$IMAGE_SIZE" \
    --pyramid "$PYRAMID" \
    --overwrite
STATUS=$?

if [[ "$STATUS" -ne 0 ]]; then
    echo "Fallo la extraccion (exit $STATUS)." >&2
    exit "$STATUS"
fi

echo ""
echo "=== 2/3 Evaluacion de cada descriptor por separado ==="
for DESC in hog color lbp gist; do
    echo ""
    echo "--- $DESC ---"
    srun python 4_evaluate_retrieval.py \
        --embeddings "data/features/global/${DESC}.npz" \
        --name "$DESC" \
        --normalize l2 \
        --metric cosine chi2 hellinger l2 \
        --datasets oxford \
        --quiet
done

echo ""
echo "=== 3/3 Concatenacion: barrido de PCA y whitening ==="
for PCA_DIM in 0 128 256 512; do
    for POWER in 0 0.25 0.5; do
        # Sin PCA el whitening no aplica: se corre una sola vez.
        if [[ "$PCA_DIM" -eq 0 && "$POWER" != "0" ]]; then
            continue
        fi

        NAME="combined_pca${PCA_DIM}_w${POWER}"
        echo ""
        echo "--- $NAME ---"

        srun python 3b_combine_global.py \
            --descriptors hog color lbp gist \
            --pca "$PCA_DIM" \
            --whiten-power "$POWER" \
            --out "data/features/global/${NAME}.npz"

        srun python 4_evaluate_retrieval.py \
            --embeddings "data/features/global/${NAME}.npz" \
            --name "$NAME" \
            --normalize l2 \
            --metric cosine \
            --datasets oxford \
            --quiet
    done
done

echo ""
echo "=== Analisis cualitativo del mejor descriptor individual ==="
srun python 4_evaluate_retrieval.py \
    --embeddings data/features/global/color.npz \
    --name color_qualitative \
    --normalize l2 --metric cosine --datasets oxford \
    --qualitative 6

echo ""
echo "=== Done. Tablas en $REPO_ROOT/results/ ==="
grep -h "" results/*_comparison.csv 2>/dev/null | head -40
exit 0
