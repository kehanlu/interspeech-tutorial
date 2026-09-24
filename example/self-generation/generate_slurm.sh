#!/bin/bash
# Self-generation on LibriSpeech train with vLLM, one GPU. Run from the repo root:
#
#   sbatch example/self-generation/generate_slurm.sh --audio-root "$AUDIO_ROOT"
#   sbatch example/self-generation/generate_slurm.sh --audio-root "$AUDIO_ROOT" \
#       --limit 50 --output data/Librispeech_self-generation/debug.jsonl
#
# Extra args go to generate_vllm.py. Prompt search on the same setup:
#   sbatch --time=0:40:00 --export=ALL,SCRIPT=prompt_search.py example/self-generation/generate_slurm.sh \
#       --audio-root "$AUDIO_ROOT" --n 500
# Account and partition come from SBATCH_ACCOUNT / SBATCH_PARTITION; export them
# before submitting, or add the matching #SBATCH lines here.
#SBATCH --job-name=desta-selfgen
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --output=exp/%x-%j.out

# Optional container holding vLLM; without SIF the script runs in the current env.
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

"${RUN[@]}" example/self-generation/${SCRIPT:-generate_vllm.py} "$@"
