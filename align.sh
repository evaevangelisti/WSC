#!/bin/bash

#SBATCH --job-name=wikt      # Job name

#SBATCH --output=/leonardo/home/userexternal/acecchet/vllm_inference/logs/%x/%x_%j.out       # Name of stdout output file
#SBATCH --error=/leonardo/home/userexternal/acecchet/vllm_inference/logs/%x/%x_%j.err       # Name of stderr error file

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=FAIR_NLP
##SBATCH --qos boost_qos_dbg


set -euo pipefail

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=WARNING

export HF_HUB_OFFLINE=1

project="/leonardo/home/userexternal/acecchet/vllm_inference/WSC"
cache_directory="$project/cache"

# Encoding harmony pre-scaricato da login node: i nodi di calcolo non hanno rete.
export TIKTOKEN_RS_CACHE_DIR="$cache_directory/harmony"
export TIKTOKEN_CACHE_DIR="$cache_directory/harmony"

model="/leonardo_scratch/fast/IscrB_SemTrain/cecchetto/huggingface/gpt-oss-120b"

input="data/senses.jsonl"
output="aligned_senses.jsonl"

prompts="$project/src/wsc/constants/prompts.toml"


python="/leonardo/home/userexternal/acecchet/vllm_inference/WSC/.venv/bin/python"

mkdir -p "$(dirname "$output")"

test -f "$input"
test -f "$cache_directory/wordnet/2025/synsets.jsonl"

test -f "$model/config.json"

"$python" -c 'from wsc.cli import app; app()' align \
  --task translations \
  --task wordnet \
  --model "$model" \
  --gloss-mode last \
  --prompts "$prompts" \
  --temperature 0.0 \
  --maximum-tokens 4096 \
  --batch-size 50000 \
  --reasoning-parser openai_gptoss \
  --reasoning-effort low \
  --engine-option enable_prefix_caching=true \
  --engine-option use_tqdm_on_load=false \
  --engine-option max_model_len=131072 \
  --engine-option max_num_seqs=128 \
  --engine-option gpu_memory_utilization=0.85 \
  --engine-option tensor_parallel_size=4 \
  --wordnet-edition 2025 \
  --cache-dir "$cache_directory" \
  "$input" \
  "$output"
