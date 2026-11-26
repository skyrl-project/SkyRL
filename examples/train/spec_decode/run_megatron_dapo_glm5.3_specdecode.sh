set -x

# Disaggregated DAPO training for GLM-5.3 (753B glm_moe_dsa MoE, full fine-tuning) on DAPO data with Megatron,
# with native Multi-Token Prediction (MTP) speculative decoding for faster rollout.
# Runs on 8x8 B200 (multi-node: 6 trainer + 2 inference nodes) on one Ray cluster; see README.md for the setup.
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode.sh                    # MTP
# MTP_ENABLED=false bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode.sh  # baseline

MODEL_NAME="${MODEL_NAME:-zai-org/GLM-5.3-BF16}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
# Trainer: 6 nodes (sizing under "megatron config" below).
NUM_NODES=6
NUM_GPUS_PER_NODE=8
# Inference: 2 nodes, one vLLM engine per node, each 8 data-parallel attention ranks sharing 8-way expert
# parallelism (DP8 x EP8, 1 GPU per rank). One node holds a whole engine only because the routed experts are
# NVFP4: ~725B expert params at ~4.5 bits are ~51 GB/GPU under EP8, and with the bf16 attention / dense weights
# that every DP rank keeps a GPU loads 86-87 GiB. That leaves over 1M tokens of NVFP4 KV per rank at vLLM's
# default gpu_memory_utilization, against the 16 x 10K tokens a rank holds per step. bf16 experts alone would be
# ~181 GB/GPU and FP8 ~91 GB/GPU.
# Two engines = 16 DP ranks = 16 sequences per rank at the start of each rollout (256 per step), where MTP
# helps most; more engines would lower that further but cost nodes.
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1
INFERENCE_ENGINE_DATA_PARALLEL_SIZE=8
INFERENCE_ENGINE_EXPERT_PARALLEL_SIZE=8
LOGGER="${LOGGER:-wandb}"  # change to "console" to print to stdout (or set LOGGER env var)

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
# use token mean loss reduction
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl in main_dapo.py
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# other DAPO parameters
USE_KL_LOSS=false
TEMPERATURE=1.0
# GLM-5.3's RL sampler: top-k 128 / top-p 0.99 trim the low-probability tail.
TOP_K=128
TOP_P=0.99
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))
# vLLM max_model_len / trainer max_seq_len: 2K prompt + 8K response.
MAX_SEQ_LEN=$((1024 * 10))

# repro run parameters
TRAIN_BATCH_SIZE=16
MINI_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=16
EVAL_N_SAMPLES_PER_PROMPT=16
ENFORCE_EAGER=false # FULL_DECODE_ONLY CUDA graphs (capture sizes in the MTP block)
LR=1e-6

# megatron config -- GLM-5.3 is a 753B MoE (~725B in 256 routed experts per layer, ~28B dense).
# bf16 weights + fp32 grads are ~4.5 TB, so the trainer needs 48 GPUs: TP1 / CP16 / PP3 / EP16 puts ~15B expert
# and ~9B dense params on each GPU (~105 GB/GPU with grads, before activations). With 4 trainer nodes (EP8 / PP4)
# model init runs out of memory. CP16 shards the DSA indexer, which TP would replicate; EP16 spans 2 nodes. PP3 at
# 26/28/24 layers: every stage starts on a layer that computes its own DSA top-k (index_topk_freq 4).
MEGATRON_TP=1
MEGATRON_PP=3
MEGATRON_CP=16
MEGATRON_EP=16
MEGATRON_ETP=1
HYBRID_LAYER_PATTERN="D-D-D-$(printf 'DE%.0s' {1..23})|$(printf 'DE%.0s' {1..28})|$(printf 'DE%.0s' {1..24})"
# Packs ~3 sequences per microbatch. Under full recompute, Megatron-Bridge's GLM-5 stack replays a whole pipeline
# stage at once in the backward, so the activations of all of a stage's layers are live together: with fp32 grads,
# 65536 runs out of memory in the first step, and 32768 peaks at ~147 GB/GPU.
MAX_TOKENS_PER_MICROBATCH=32768

# optimizer state on node-local disk: fp32 master params and bf16 Adam moments (8 B/param, ~1.1 TB on the busiest
# trainer node), streamed through the GPU one chunk at a time in the optimizer step. Megatron's CPU-offloaded Adam
# would keep fp32 master, fp32 moments and fp32 grads in host RAM (16 B/param, ~2.2 TB on the busiest node), more
# than the hosts have. Point OPTIMIZER_STATE_DIR at fast node-local NVMe on every trainer node.
OPTIMIZER_STATE_DIR="${OPTIMIZER_STATE_DIR:-/mnt/local_storage/skyrl_optimizer_state}"

# TIS parameters
TIS_IMP_RATIO_CAP=2.0
TIS_TYPE=token


# Multi-Token Prediction (MTP) speculative decoding.
# GLM-5.3 ships 1 native MTP layer (`num_nextn_predict_layers: 1`, layer 78), but Megatron-Bridge's GLM5 bridge
# builds none unless mtp_num_layers=1 is set (the baseline ignores it: trainer.mtp.enabled=false builds no head).
# NUM_SPECULATIVE_TOKENS is the vLLM *draft depth* only -- values > 1 reuse the single layer autoregressively at
# draft time. Here k=4, with probabilistic drafting: each draft token is sampled from the head's distribution
# instead of its argmax. Verification is still vLLM's exact rejection sampler, so this is lossless; it raises
# acceptance at temperature 1. vLLM re-quantizes the drafter's experts to NVFP4 like the target's.
MTP_ENABLED="${MTP_ENABLED:-true}"
MTP_NUM_SPECULATIVE_TOKENS=4
MTP_DRAFT_SAMPLE_METHOD=probabilistic
# hard_label (CE on the generated tokens) is the draft loss that runs under CP, PP and the fused LM head.
# NOTE: the draft loss trains ONLY the MTP-head params -- trunk, re-embedding and output weight are detached.
MTP_LOSS_WEIGHT=0.1
if [ "$MTP_ENABLED" = "true" ]; then
  SPECULATIVE_CONFIG='{"method": "mtp", "num_speculative_tokens": '"$MTP_NUM_SPECULATIVE_TOKENS"', "draft_sample_method": "'"$MTP_DRAFT_SAMPLE_METHOD"'"}'
  # MTP verifies k+1 tokens per sequence: capture decode graphs up to max_num_seqs * (k+1) tokens.
  CUDAGRAPH_CAPTURE_SIZES="[1, 2, 4, 8, 16, 32, 64, 128, 160, 320, 640]"
else
  SPECULATIVE_CONFIG=null
  CUDAGRAPH_CAPTURE_SIZES="[1, 2, 4, 8, 16, 32, 64, 128]"
fi


# GLM-5.3 flags
# Trainer: DSA sparse attention on cuDNN/FlashMLA with the TileLang indexer top-k; no indexer loss, so megatron-core
# freezes the indexer.
export SKYRL_DSA_INDEXER_BACKEND=tilelang
# Inference: NVFP4 routed experts, re-quantized online from the bf16 weights on every sync (W4A4 with per-token
# activation scales); other layers stay bf16. NVFP4 KV cache. NVLink one-sided expert all-to-all (each engine's EP
# group is one node). sparse_mla_force_mqa keeps prefill on sparse MLA's MQA path: the default MHA prefill path
# cannot read an NVFP4 KV cache. The engines boot from the real checkpoint, so the MTP drafter has its weights before
# the first sync; with enable_ep_weight_filter each rank reads only its own experts.
ENGINE_INIT_KWARGS='{
  "max_model_len": '"$MAX_SEQ_LEN"',
  "quantization": "nvfp4_per_token", "kv_cache_dtype": "nvfp4_ds_mla",
  "all2all_backend": "flashinfer_nvlink_one_sided", "enable_ep_weight_filter": true,
  "attention_config": {"sparse_mla_force_mqa": true},
  "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": '"$CUDAGRAPH_CAPTURE_SIZES"'}}'
# No DSA indexer loss (the bridge's 0.001 would run the indexer KL loss every step).
TRANSFORMER_CONFIG_KWARGS='{
  "dsa_kernel_backend": "cudnn", "dsa_indexer_loss_coeff": 0.0,
  "hybrid_layer_pattern": "'"$HYBRID_LAYER_PATTERN"'"}'
DISTRIBUTED_EXECUTOR_BACKEND="ray"
# The router sends each trajectory to the least-loaded engine DP rank (16 per rank). SkyRL's default hashes the session
# id, which spreads them unevenly; rollouts were ~3% slower with it.
ROUTER_POLICY=sticky_least_loaded
# Settings SkyRL does not forward to the Ray workers (your NCCL / EFA settings, WANDB_API_KEY, HF_TOKEN) go in
# EXTRA_ENV_FILE=/path/to/.env. The script copies it to an env file next to itself and passes that with
# `uv run --env-file`, which Ray's uv hook replays on every worker, so keep the checkout at the same path on every node.
WORKER_ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.glm5.3_specdecode.env"
: > "$WORKER_ENV_FILE"
if [ -n "${EXTRA_ENV_FILE:-}" ]; then cat "$EXTRA_ENV_FILE" >> "$WORKER_ENV_FILE"; fi

uv run --isolated --extra megatron --env-file "$WORKER_ENV_FILE" -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=true \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_k=$TOP_K \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=false \
  trainer.strategy=megatron \
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.data_parallel_size=$INFERENCE_ENGINE_DATA_PARALLEL_SIZE \
  generator.inference_engine.expert_parallel_size=$INFERENCE_ENGINE_EXPERT_PARALLEL_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.transformer_config_kwargs="$TRANSFORMER_CONFIG_KWARGS" \
  trainer.policy.megatron_config.mtp_num_layers=1 \
  trainer.policy.megatron_config.optimizer_offload_to_disk.dir="$OPTIMIZER_STATE_DIR" \
  trainer.policy.megatron_config.optimizer_offload_to_disk.moment_storage_dtype=bf16 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.fused_lm_head_logprob=true \
  trainer.fused_lm_head_logprob_backend=triton \
  trainer.remove_microbatch_padding=true \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.epochs=10 \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=5 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=sharded_rdt \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.max_num_seqs=128 \
  generator.inference_engine.router_init_kwargs.policy=$ROUTER_POLICY \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.speculative_config="$SPECULATIVE_CONFIG" \
  generator.batched=false \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  trainer.mtp.enabled=$MTP_ENABLED \
  trainer.mtp.num_speculative_tokens=$MTP_NUM_SPECULATIVE_TOKENS \
  trainer.mtp.loss_weight=$MTP_LOSS_WEIGHT \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm5p3_dapo" \
  trainer.run_name="sd_dapo_glm5p3_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_mtp${MTP_ENABLED}" \
  trainer.export_path="$HOME/exports/sd_dapo_glm5p3_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_mtp${MTP_ENABLED}" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="$HOME/ckpts/sd_dapo_glm5p3_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_mtp${MTP_ENABLED}" \
  "$@"
