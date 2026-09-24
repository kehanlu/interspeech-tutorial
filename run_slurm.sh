#!/bin/bash
# Single node, 8 GPUs. Submit with:
#   sbatch run_slurm.sh                                   # configs/selfgen.yaml
#   sbatch run_slurm.sh configs/other.yaml                # any other config
#   sbatch --partition=dev --time=0:40:00 run_slurm.sh configs/asr_gender.yaml --max-steps 400   # timing run
#
# Account and partition are not set here -- export SBATCH_ACCOUNT / SBATCH_PARTITION
# before submitting, or add the matching #SBATCH lines below. (sbatch reads those
# environment variables, but only if no #SBATCH directive overrides them.)
#SBATCH --job-name=desta-librispeech
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --output=exp/%x-%j.out

# Optional container: set SIF to a singularity image and the job runs inside it,
# otherwise train.py runs in whatever environment the job starts in. BIND is the
# path bound into the container -- add your audio_root if it lives outside $PWD.
SIF=${SIF:-}
if [ -n "$SIF" ]; then
    module load singularity 2>/dev/null
    command -v singularity >/dev/null || { echo "singularity not found"; exit 1; }
    RUN=(singularity exec --nv -B "${BIND:-$PWD}" "$SIF" python)
else
    RUN=(python)
fi

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
# wandb reads its API key from ~/.netrc (run `wandb login` once); never put it in this file.

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

CONFIG=${1:-configs/selfgen.yaml}
shift || true                      # any further args go straight to train.py

# Everything else comes from the yaml; only the per-job values are set here.
srun "${RUN[@]}" train.py \
    --config    "$CONFIG" \
    --exp-dir   "exp/$(basename "$CONFIG" .yaml)/$SLURM_JOB_ID" \
    --devices   "$SLURM_NTASKS_PER_NODE" \
    --num-nodes "$SLURM_JOB_NUM_NODES" \
    --wandb-run-name "$(basename "$CONFIG" .yaml)-$SLURM_JOB_ID" \
    "$@"
