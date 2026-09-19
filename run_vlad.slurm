#!/bin/bash
# VLAD: barrido de K, ablacion de normalizaciones, y query expansion.
#
# Uso:
#   sbatch run_vlad.slurm
#
# Requiere data/features/sift/sift_features.h5 (NO las imagenes):
#   ./data/download_sift_features.sh   # desde el nodo de acceso

#SBATCH --job-name=vlad
#SBATCH --account=tesis
#SBATCH --partition=standard
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alvaro.garcia.h@utec.edu.pe

set -uo pipefail

REPO_ROOT="$(pwd)"
ENV_NAME="vcrp-proyecto-1"

echo "=== Job $SLURM_JOB_ID on $(hostname), repo: $REPO_ROOT ==="

set +u
module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

H5="data/features/sift/sift_features.h5"
if [[ ! -f "$H5" ]]; then
    echo "No existe $H5. Bajalo con ./data/download_sift_features.sh" >&2
    exit 1
fi

if [[ ! -d "data/groundtruth/oxford" ]]; then
    bash data/download_groundtruth.sh || true
fi

evaluar() {
    local emb="$1"
    local name="$2"

    # Sin QE
    srun python 4_evaluate_retrieval.py --embeddings "$emb" --name "$name" \
        --normalize none --metric cosine --datasets oxford --quiet

    # Con query expansion, barriendo cuantos vecinos se promedian
    for N in 5 10 20 50; do
        srun python 4_evaluate_retrieval.py --embeddings "$emb" --name "${name}_qe${N}" \
            --normalize none --metric cosine --datasets oxford \
            --query-expansion "$N" --qe-alpha 3 --quiet
    done

    # Average QE clasica (alpha = 0), para la comparacion del reporte
    srun python 4_evaluate_retrieval.py --embeddings "$emb" --name "${name}_aqe10" \
        --normalize none --metric cosine --datasets oxford \
        --query-expansion 10 --qe-alpha 0 --quiet
}

echo ""
echo "=== 1/3 Barrido de K ==="
for K in 32 64 128 256; do
    echo ""
    echo "--- VLAD K=$K ---"
    srun python 3c_vlad.py --vocab-size "$K" --sample-size 500000 --overwrite
    evaluar "data/features/vlad/vlad_k${K}.npz" "vlad_k${K}"
done

echo ""
echo "=== 2/3 Ablacion de normalizaciones (K=64) ==="
for OPT in "--no-intra-norm|nointra" "--intra-norm|intra" "--intra-norm --mass-norm|intra_mass" "--no-intra-norm --power|power" "--no-rootsift --intra-norm|norootsift"; do
    FLAGS="${OPT%%|*}"
    TAG="${OPT##*|}"
    echo ""
    echo "--- VLAD K=64 $FLAGS ---"
    srun python 3c_vlad.py --vocab-size 64 --sample-size 500000 \
        --suffix "abl_${TAG}" $FLAGS --overwrite
    srun python 4_evaluate_retrieval.py \
        --embeddings "data/features/vlad/vlad_abl_${TAG}.npz" \
        --name "vlad_abl_${TAG}" --normalize none --metric cosine \
        --datasets oxford --quiet
done

echo ""
echo "=== 3/3 VLAD comprimido con PCA (K=128) ==="
for DIM in 128 256 512; do
    echo ""
    echo "--- VLAD K=128 -> PCA $DIM ---"
    srun python 3c_vlad.py --vocab-size 128 --sample-size 500000 \
        --pca "$DIM" --whiten-power 0.25 --suffix "k128_pca${DIM}" --overwrite
    srun python 4_evaluate_retrieval.py \
        --embeddings "data/features/vlad/vlad_k128_pca${DIM}.npz" \
        --name "vlad_k128_pca${DIM}" --normalize none --metric cosine \
        --datasets oxford --quiet
done

echo ""
echo "=== Analisis cualitativo del mejor VLAD ==="
srun python 4_evaluate_retrieval.py \
    --embeddings data/features/vlad/vlad_k128.npz \
    --name vlad_best --normalize none --metric cosine --datasets oxford \
    --query-expansion 10 --qe-alpha 3 --qualitative 6

echo ""
echo "=== Done. Tablas en $REPO_ROOT/results/ ==="
exit 0
