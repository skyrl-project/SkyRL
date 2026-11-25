#!/usr/bin/env bash
# Tmax through skycap: Tmax's own harness (Vanillux2) drives Daytona sandboxes, trains on Tmax's 13,565
# terminal tasks and evaluates on TB-Lite, as Tmax does (Ivison et al., arXiv 2606.23321). Megatron only.
#
#   SANDBOX_OWNER=<you> bash examples/train_integrations/harbor_skycap/tmax/run_tmax.sh [extra overrides...]
#
# Defaults: Qwen3.5-2B, LoRA r32, 2 GPUs colocated, 8 tasks x 8 samples per step, 16 steps, 64k context,
# DPPO + prompt_mean, TB-Lite x 1 before the first step and after the last. Prepares the tasks once
# (prepare_tmax.py) under DATA_ROOT.
#
#   MODEL                      the policy (default Qwen/Qwen3.5-2B); skycap needs a renderer for its exact name
#   LORA_RANK, LORA_ALPHA, LR  LoRA rank and alpha (default 32, 32) and learning rate (default 1e-4)
#   NUM_PROMPTS, GROUP_SIZE    tasks per step and samples per task (default 8 x 8)
#   STEPS                      training steps (default 16); tasks are drawn from all of Tmax, shuffled with SEED
#   SEED                       the trainer's seed, which also fixes the task order (default 42)
#   MAX_MODEL_LEN              context length, prompt and completions (default 65536)
#   TURN_TOKENS, AGENT_STEPS   Vanillux2's max tokens per reply and model calls per trial (default 16384, 64)
#   AGENT_TIMEOUT_SEC          wall time per trial (default: the task's own, 600 s for Tmax)
#   TOP_K                      sampling top-k (default 64; score centering needs it > 1). Temperature and
#                              top-p are 1.0. skycap imposes these on every call, eval included
#   FP8=0|1                    FP8 rollouts: each weight sync quantises the merged weights in 128x128 blocks
#                              (fp8_weight_sync_mode=blockwise) and vLLM samples with FP8 GEMMs (default 0)
#   CENTER=0|1                 sample-support replay + score centering (default 0)
#   ZERO_VARIANCE_FILTER       mask groups whose samples all scored the same (default true, as Tmax does). A masked
#                              row also drops out of the policy metrics, so set false to measure the trainer-sampler
#                              gap on a model that solves almost nothing
#   EVAL_TRIALS                TB-Lite trials per task (default 1; Tmax uses 3); 0 turns eval off
#   EVAL_INTERVAL              steps between TB-Lite evals (default STEPS: before the first step and after the last)
#   EVAL_DATA                  the eval tasks' directory (default DATA_ROOT/tblite), e.g. a few of them for a smoke run
#   CKPT_INTERVAL, MAX_CKPTS   steps between checkpoints and how many to keep (default 8, 2); -1 saves none
#   RUN_NAME                   the run's name and directory (default tmax-<random>); the same name again resumes it
#   OUTPUT_ROOT                where run directories go (default /mnt/cluster_storage/tmax/runs when it exists,
#                              else /tmp/tmax/runs)
#   NUM_GPUS, NUM_ENGINES, TP_SIZE   GPUs, vLLM engines and their tensor parallelism (default 2, 2, 1)
#   MEGATRON_TP, MEGATRON_PP   Megatron parallelism (default 1, 1)
#   GPU_MEMORY_UTILIZATION     vLLM's share of each GPU next to the trainer (default 0.6)
#   MAX_CONCURRENCY            trials (so sandboxes) in flight, eval included (default 64). The Daytona org is
#                              shared: keep it well under what others leave free
#   SANDBOX_OWNER              required: the `owner` label on every sandbox; `run` is RUN_NAME
#   SANDBOX_IDLE_MINUTES       Daytona stops a sandbox idle this long and then deletes it (default 30)
#   AUTO_SNAPSHOT=0|1          1: Harbor builds one Daytona snapshot per task environment and starts all of that
#                              task's sandboxes from it, so the task image is pulled from Docker Hub once, not
#                              once per sandbox; snapshots are reused across runs. Needs a Daytona key allowed to
#                              create snapshots (else every trial fails "Failed to create snapshot: Access
#                              denied"). 0 (default): every sandbox builds its task's Dockerfile
#   CUDA_HOME                  the CUDA toolkit TileLang compiles Qwen3.5's GatedDeltaNet kernels with (default
#                              /usr/local/cuda): the pip wheels' nvcc and headers don't match
#   ENGINE_INIT_KWARGS         extra vLLM engine kwargs as a JSON object
#   DATA_ROOT                  where prepare_tmax.py writes tmax/ and tblite/ (default ~/data/tmax)
#
# Synchronous training only: the generator's cache salt never reaches Vanillux2 (it takes no llm_kwargs), which
# is a no-op here because a synchronous weight sync resets vLLM's prefix cache, but not under fully-async training.
#
# Needs: the GPUs, `DAYTONA_API_KEY=...` in DAYTONA_KEY_FILE, optionally WANDB_API_KEY (or WANDB_KEY_FILE), and
# git (uv installs Tmax's harness from GitHub).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"
HERE=examples/train_integrations/harbor_skycap/tmax

#-----------------------
# Knobs
#-----------------------
: "${SANDBOX_OWNER:?set SANDBOX_OWNER, the owner label every sandbox of this run carries}"
DAYTONA_KEY_FILE="${DAYTONA_KEY_FILE:-$HOME/.config/skyrl/daytona.env}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/default/wandb_key.key}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-1.0e-4}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
GROUP_SIZE="${GROUP_SIZE:-8}"
STEPS="${STEPS:-16}"
SEED="${SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
TURN_TOKENS="${TURN_TOKENS:-16384}"
AGENT_STEPS="${AGENT_STEPS:-64}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-null}"
TOP_K="${TOP_K:-64}"
FP8="${FP8:-0}"
CENTER="${CENTER:-0}"
ZERO_VARIANCE_FILTER="${ZERO_VARIANCE_FILTER:-true}"
EVAL_TRIALS="${EVAL_TRIALS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-$STEPS}"
CKPT_INTERVAL="${CKPT_INTERVAL:-8}"
MAX_CKPTS="${MAX_CKPTS:-2}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_ENGINES="${NUM_ENGINES:-2}"
TP_SIZE="${TP_SIZE:-1}"
MEGATRON_TP="${MEGATRON_TP:-1}"
MEGATRON_PP="${MEGATRON_PP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
SANDBOX_IDLE_MINUTES="${SANDBOX_IDLE_MINUTES:-30}"
AUTO_SNAPSHOT="${AUTO_SNAPSHOT:-0}"
ENGINE_INIT_KWARGS="${ENGINE_INIT_KWARGS:-}"
SKYCAP_SERVERS="${SKYCAP_SERVERS:-1}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/tmax}"
EVAL_DATA="${EVAL_DATA:-$DATA_ROOT/tblite}"
# Tmax's harness, at the commit this recipe was validated with. Harbor is pinned to SkyRL's lock in the same
# overlay: tmax asks for an unpinned `harbor`, which would otherwise shadow SkyRL's. LiteLLM is newer than SkyRL's
# lock (1.102.1), whose hosted_vllm provider drops `reasoning_content` from every assistant turn it sends back: a
# thinking model's history would then differ from what it sampled at every turn, and each turn would fork skycap's
# graph into a row of its own. 1.110.0 keeps it. No spaces in these
# requirements: Ray rebuilds each worker's `uv run` command from this one by splitting on spaces, so
# "tmax @ git+..." would reach the workers as `--with tmax` followed by a command named `@`.
TMAX_REF="${TMAX_REF:-2c7f55e0e879cb16a21fe8cd6edbd08a976e43d3}"
TMAX_WITH=(
  --with "git+https://github.com/hamishivi/tmax@$TMAX_REF"
  --with "harbor[daytona,modal]==0.14.0"
  --with "litellm==1.110.0"
)

if [[ -d /mnt/cluster_storage ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cluster_storage/tmax/runs}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/tmax/runs}"
fi
RUN_NAME="${RUN_NAME:-tmax-$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

#-----------------------
# Credentials
#-----------------------
[[ -f "$DAYTONA_KEY_FILE" ]] || { echo "no Daytona credentials at $DAYTONA_KEY_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$DAYTONA_KEY_FILE"
set +a
: "${DAYTONA_API_KEY:?$DAYTONA_KEY_FILE must set DAYTONA_API_KEY}"
if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_KEY_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$WANDB_KEY_FILE"
  set +a
fi
LOGGER="${LOGGER:-$([[ -n "${WANDB_API_KEY:-}" ]] && echo wandb || echo console)}"
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY
[[ "$LOGGER" != wandb ]] || export WANDB_API_KEY

#-----------------------
# Data
#-----------------------
uv run --isolated --no-project --with huggingface_hub --with pyyaml "$HERE/prepare_tmax.py" --output-dir "$DATA_ROOT"

#-----------------------
# Switches
#-----------------------
SWITCH_ARGS=()
case "$FP8" in
  0) ;;
  # FP8 rollouts need full-weight syncs: an adapter-only sync (merge_lora=false) would leave vLLM's FP8 base
  # weights at the dummy values it starts with.
  1) SWITCH_ARGS+=(
       generator.inference_engine.fp8_weight_sync_mode=blockwise
       trainer.policy.megatron_config.lora_config.merge_lora=true
     ) ;;
  *) echo "FP8 must be 0 or 1, got $FP8" >&2; exit 1 ;;
esac
case "$CENTER" in
  0) SWITCH_ARGS+=(trainer.algorithm.enable_score_centering=false) ;;
  1) SWITCH_ARGS+=(
       generator.inference_engine.enable_return_sample_support_set=true
       trainer.algorithm.enable_sample_support_replay=true
       trainer.algorithm.enable_score_centering=true
     ) ;;
  *) echo "CENTER must be 0 or 1, got $CENTER" >&2; exit 1 ;;
esac
if [[ "$EVAL_TRIALS" == 0 ]]; then
  EVAL_ARGS=(trainer.eval_interval=-1 trainer.eval_before_train=false)
else
  EVAL_ARGS=(
    data.val_data="['$EVAL_DATA']"
    trainer.eval_interval="$EVAL_INTERVAL"
    trainer.eval_before_train=true
    generator.eval_n_samples_per_prompt="$EVAL_TRIALS"
  )
fi
if [[ "$AUTO_SNAPSHOT" == 1 ]]; then
  SANDBOX_ARGS=(harbor_trial_config.environment.kwargs.auto_snapshot=true)
else
  SANDBOX_ARGS=()
fi
if [[ -n "$ENGINE_INIT_KWARGS" ]]; then
  while IFS= read -r arg; do SWITCH_ARGS+=("$arg"); done < <(uv run --isolated --no-project python -c '
import json, sys
for key, value in json.loads(sys.argv[1]).items():
    print(f"generator.inference_engine.engine_init_kwargs.{key}={value if isinstance(value, str) else json.dumps(value)}")
' "$ENGINE_INIT_KWARGS")
fi

# TileLang JITs Qwen3.5's GatedDeltaNet kernels with nvcc from CUDA_HOME, which otherwise defaults to the pip wheels,
# whose nvcc and runtime versions differ ("CUDA compiler and CUDA toolkit headers are incompatible"). SkyRL forwards
# CUDA_HOME to the trainer and engine actors.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "no nvcc at $CUDA_HOME/bin/nvcc: set CUDA_HOME to a CUDA toolkit" >&2; exit 1; }

# A fresh local Ray cluster. Workspaces (Anyscale) export Ray event settings a local cluster can't serve:
# it segfaults in the event aggregator about a second after ray.init.
export RAY_ADDRESS=local
export RAY_enable_ray_event=0
export RAY_enable_core_worker_ray_event_to_aggregator=0
echo "==> $RUN_NAME in $RUN_DIR: $MODEL, $NUM_PROMPTS tasks x $GROUP_SIZE samples, $STEPS steps, FP8=$FP8 CENTER=$CENTER"
echo "==> sandboxes labelled owner=$SANDBOX_OWNER run=$RUN_NAME, at most $MAX_CONCURRENCY at once, deleted after ${SANDBOX_IDLE_MINUTES}m idle"

uv run --isolated --extra megatron --extra harbor --extra skycap "${TMAX_WITH[@]}" \
  -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap \
  data.train_data="['$DATA_ROOT/tmax']" \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  trainer.project_name=skycap-tmax \
  trainer.run_name="$RUN_NAME" \
  trainer.logger="$LOGGER" \
  trainer.seed="$SEED" \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS" \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=latest \
  harbor_trial_config.trials_dir="$RUN_DIR/trials" \
  harbor_trial_config.agent.name=null \
  harbor_trial_config.agent.import_path=Vanillux2Agent.agent:Vanillux2Agent \
  harbor_trial_config.agent.override_timeout_sec="$AGENT_TIMEOUT_SEC" \
  harbor_trial_config.agent.kwargs.max_steps="$AGENT_STEPS" \
  harbor_trial_config.agent.kwargs.max_tokens="$TURN_TOKENS" \
  harbor_trial_config.environment.override_cpus=null \
  harbor_trial_config.environment.override_memory_mb=null \
  harbor_trial_config.environment.override_storage_mb=null \
  harbor_trial_config.environment.kwargs.labels.owner="$SANDBOX_OWNER" \
  harbor_trial_config.environment.kwargs.labels.project=skyrl-tmax \
  harbor_trial_config.environment.kwargs.labels.run="$RUN_NAME" \
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins="$SANDBOX_IDLE_MINUTES" \
  harbor_trial_config.environment.kwargs.auto_delete_interval_mins=0 \
  skycap.record_dir="$RUN_DIR/skycap" \
  skycap.num_servers="$SKYCAP_SERVERS" \
  skycap.use_raw_content=false \
  skycap.train_paths=all \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=false \
  generator.apply_overlong_filtering=false \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.max_concurrency="$MAX_CONCURRENCY" \
  generator.n_samples_per_prompt="$GROUP_SIZE" \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k="$TOP_K" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  trainer.epochs=1 \
  trainer.max_training_steps="$STEPS" \
  trainer.train_batch_size="$NUM_PROMPTS" \
  trainer.policy_mini_batch_size="$NUM_PROMPTS" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=dppo \
  trainer.algorithm.dppo.delta_low=0.15 \
  trainer.algorithm.dppo.delta_high=0.15 \
  trainer.algorithm.loss_reduction=prompt_mean \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.zero_variance_filter="$ZERO_VARIANCE_FILTER" \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.policy.model.lora.rank="$LORA_RANK" \
  trainer.policy.model.lora.alpha="$LORA_ALPHA" \
  trainer.policy.optimizer_config.lr="$LR" \
  trainer.strategy=megatron \
  trainer.policy.megatron_config.tensor_model_parallel_size="$MEGATRON_TP" \
  trainer.policy.megatron_config.pipeline_model_parallel_size="$MEGATRON_PP" \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.fused_lm_head_logprob=true \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  "${EVAL_ARGS[@]}" \
  "${SWITCH_ARGS[@]}" \
  "${SANDBOX_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$RUN_DIR/run.log"

echo
echo "==> done: $RUN_NAME"
echo "    skycap records: $RUN_DIR/skycap"
echo "    Harbor trials:  $RUN_DIR/trials"
echo "    leftover sandboxes: labelled owner=$SANDBOX_OWNER run=$RUN_NAME, deleted after ${SANDBOX_IDLE_MINUTES}m idle"
