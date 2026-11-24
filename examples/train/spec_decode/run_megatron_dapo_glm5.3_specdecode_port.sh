set -x

# DAPO for GLM-5.3 (753B glm_moe_dsa MoE, full fine-tuning) with the model's own MTP layer drafting for
# vLLM. 8 nodes x 8xB200, not colocated: 6 trainer nodes + 2 vLLM engine nodes.
#
# This is a port of the showcase recipe of https://github.com/skyrl-project/SkyRL/pull/2808 to the code
# that is in this branch (main + the MTP stack, PRs 2758-2763). That recipe needs pieces that are not
# upstream (cpu_adamw_bf16, nvfp4_online_quant, the drafter on the NVFP4 wires, tiled sharded_rdt pulls).
# This script keeps the settings that make the speedup visible and replaces the rest:
#
#   kept from PR 2808                          | replaced, and with what
#   -------------------------------------------+---------------------------------------------------
#   6 trainer + 2 engine nodes, not colocated  | NVFP4 DP8 x EP8 engines -> online-FP8 TP8 engines
#   16 prompts x 16 samples per step           | cpu_adamw_bf16         -> optimizer_offload_to_disk
#   k=4 probabilistic drafts, hard-label loss  | TP1/CP16/PP3/EP16 DeepEP -> TP8/CP1/PP3/EP16 alltoall
#   capture sizes up to max_num_seqs * (k+1)   | NVFP4 KV cache         -> bf16 KV cache
#   sharded_rdt weight sync, top-k 128 / top-p 0.99
#
# STATUS: NOT RUN. The config passes validate_cfg; no GPU run exists yet. Known risks are in the PR text.
#
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode_port.sh                    # MTP
#   MTP_ENABLED=false bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode_port.sh  # baseline
#
# The two arms differ only in MTP_ENABLED and in the CUDA-graph capture list.

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/models/glm5p3-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER="${LOGGER:-wandb}"

# ---- placement: 6 trainer nodes + 2 engine nodes ----
NUM_TRAINER_NODES="${NUM_TRAINER_NODES:-6}"
NUM_GPUS_PER_NODE=8
# One online-FP8 TP8 engine per engine node: ~89 GiB/GPU of FP8 weights, the layout the 8-node recipe
# (../glm5_3/run_dapo_glm5p3_fullft_fp8_rollout_8node.sh) runs. PR 2808 uses DP8 x EP8 with NVFP4 experts
# (16 sequences per rank); with FP8 experts that layout leaves almost no room for the KV cache.
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-2}"
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE="${INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE:-8}"
INFERENCE_ENGINE_DATA_PARALLEL_SIZE="${INFERENCE_ENGINE_DATA_PARALLEL_SIZE:-1}"
INFERENCE_ENGINE_EXPERT_PARALLEL_SIZE="${INFERENCE_ENGINE_EXPERT_PARALLEL_SIZE:-1}"
# Engine-only nodes: no trainer shares the GPU, so 0.8 (the colocated runs needed 0.7).
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.8}"
# sharded_rdt is the backend PR 2762 adds the drafter sync to. nccl is the fallback.
WEIGHT_SYNC_BACKEND="${WEIGHT_SYNC_BACKEND:-sharded_rdt}"
# sharded_rdt needs vLLM's Ray executor (the engine refuses mp at init); nccl runs on mp.
if [ "$WEIGHT_SYNC_BACKEND" = "sharded_rdt" ]; then
  INFERENCE_EXECUTOR_BACKEND="${INFERENCE_EXECUTOR_BACKEND:-ray}"
else
  INFERENCE_EXECUTOR_BACKEND="${INFERENCE_EXECUTOR_BACKEND:-mp}"
fi

# ---- batch: 16 prompts x 16 samples = 256 sequences per step, 128 per engine ----
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
EVAL_N_SAMPLES_PER_PROMPT=16
INFERENCE_ENGINE_MAX_NUM_SEQS=128
# Megatron's scheduler asserts unless max_training_steps > num_warmup_steps (0 here). 6 steps: step 1 is
# warm-up, report the mean of steps 2-5 as PR 2808 does.
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-6}"
EVAL_INTERVAL="${EVAL_INTERVAL:--1}"

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
INFERENCE_ENGINE_MAX_MODEL_LEN=10752         # prompt + response + chat-template headroom
MAX_TOKENS_PER_MICROBATCH=8192               # soft cap: a longer sequence gets a microbatch of its own
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# ---- DAPO ----
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
USE_KL_LOSS=false
TEMPERATURE=1.0
# GLM-5.3's RL sampler in PR 2808: top-k 128 / top-p 0.99 trim the low-probability tail.
TOP_K=128
TOP_P=0.99
EVAL_TOP_P=0.7
LR=1e-6

# ---- trainer: TP8 / PP3 / EP16 on 48 GPUs (DP 2) ----
# 753B params: ~725B in 256 routed experts per layer, ~28B dense. PP3 puts a third of the layers on each
# 16-GPU stage; EP16 puts 16 experts of each layer on a GPU: ~15B expert params, ~30 GiB bf16 weights plus
# ~60 GiB fp32 main grads. The 8-node recipe's TP8 / EP64 does not fit 48 GPUs (48 is not a multiple of 64),
# and PR 2808's TP1 / CP16 / EP16 needs the DeepEP flex dispatcher, which this tree could not run
# (../glm5_3/throughput/README.md, "DeepEP / NCCL EP").
MEGATRON_TP="${MEGATRON_TP:-8}"
MEGATRON_PP="${MEGATRON_PP:-3}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-16}"
MEGATRON_ETP=1
# The GLM-5 bridge needs one "|"-separated segment per pipeline stage, each starting on a layer that
# computes its own DSA top-k. 26 / 28 / 24 layers is the split PR 2808 uses: 3 dense layers ("D-"), then
# attention + MoE layers ("DE").
HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN:-D-D-D-$(printf 'DE%.0s' {1..23})|$(printf 'DE%.0s' {1..28})|$(printf 'DE%.0s' {1..24})}"
# [moe] is what ran on B200 with the MTP stack; [core_attn,moe] is the 8-node recipe's default.
RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-[moe]}"
# Sparse attention on cuDNN/FlashMLA, the indexer top-k on TileLang (patch_dsa_hybrid_indexer).
DSA_KERNEL_BACKEND=cudnn
export SKYRL_DSA_INDEXER_BACKEND=tilelang

# ---- optimizer state on node-local disk ----
# fp32 main params + Adam moments are 12 B/param, ~9 TB: more than the GPUs or the hosts of 6 nodes hold.
# The disk optimizer streams the state through the GPU one chunk at a time. It requires
# grad_reduce_in_fp32=true, optimizer_cpu_offload=false and use_precision_aware_optimizer=false.
OPTIMIZER_OFFLOAD_DIR="${OPTIMIZER_OFFLOAD_DIR:-/mnt/local_storage/skyrl_optimizer_offload}"
OPTIMIZER_MOMENT_DTYPE="${OPTIMIZER_MOMENT_DTYPE:-fp32}"

# ---- MTP speculative decoding ----
# GLM-5.3 ships 1 native MTP layer. Megatron-Bridge's GLM5 bridge builds none unless mtp_num_layers=1 is
# set. NUM_SPECULATIVE_TOKENS is the vLLM draft depth: values > 1 reuse the single layer autoregressively.
# probabilistic: each draft token is sampled from the head's distribution instead of its argmax. vLLM's
# rejection sampler still verifies every token, so the output distribution does not change.
MTP_ENABLED="${MTP_ENABLED:-true}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-4}"
MTP_DRAFT_SAMPLE_METHOD="${MTP_DRAFT_SAMPLE_METHOD:-probabilistic}"
MTP_LOSS_WEIGHT=0.1
if [ "$MTP_ENABLED" = "true" ]; then
  MTP_NUM_LAYERS=1
  SPECULATIVE_CONFIG='{"method": "mtp", "num_speculative_tokens": '"$MTP_NUM_SPECULATIVE_TOKENS"', "draft_sample_method": "'"$MTP_DRAFT_SAMPLE_METHOD"'"}'
  # MTP verifies k+1 tokens per sequence: capture decode graphs up to max_num_seqs * (k+1) tokens. With the
  # 8-node recipe's cap of 256, k=4 ran outside the graphs above 51 sequences and was slower than no MTP.
  CUDAGRAPH_CAPTURE_SIZES="[1, 2, 4, 8, 16, 32, 64, 128, 160, 320, 640]"
else
  MTP_NUM_LAYERS=0
  SPECULATIVE_CONFIG=null
  CUDAGRAPH_CAPTURE_SIZES="[1, 2, 4, 8, 16, 32, 64, 128]"
fi

# fp8_per_block needs SkyRL's per_block_cast_to_fp8 patch (installed in every vLLM worker). The trainer
# syncs bf16 weights and vLLM re-quantizes each layer as it arrives. load_format=dummy: SkyRL syncs the
# trainer's weights, and the drafter's, before the first rollout.
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8_per_block}"
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "quantization": "'"$VLLM_QUANTIZATION"'", "load_format": "dummy", "kv_cache_dtype": "bfloat16", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": '"$CUDAGRAPH_CAPTURE_SIZES"', "pass_config": {"fuse_allreduce_rms": false}}}'
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=$INFERENCE_ENGINE_MAX_NUM_SEQS
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# vLLM servers above Ray's default worker-port range (10002-19999).
export SKYRL_VLLM_START_PORT="${SKYRL_VLLM_START_PORT:-20000}"
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1

RUN_NAME="${RUN_NAME:-sd_dapo_glm5p3_port_tp${MEGATRON_TP}_pp${MEGATRON_PP}_ep${MEGATRON_EP}_${WEIGHT_SYNC_BACKEND}_mtp${MTP_ENABLED}}"

${UV_RUN:-uv run --isolated} --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.placement.ref_num_nodes=0 \
  trainer.placement.ref_num_gpus_per_node=0 \
  trainer.placement.critic_num_nodes=0 \
  trainer.placement.policy_num_nodes=$NUM_TRAINER_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.mtp_num_layers=$MTP_NUM_LAYERS \
  trainer.policy.megatron_config.moe_grouped_gemm=true \
  trainer.policy.megatron_config.moe_token_dispatcher_type="alltoall" \
  trainer.policy.megatron_config.moe_router_score_function="sigmoid" \
  trainer.policy.megatron_config.moe_router_load_balancing_type="none" \
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true \
  trainer.policy.megatron_config.transformer_config_kwargs.hybrid_layer_pattern="'$HYBRID_LAYER_PATTERN'" \
  trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel=true \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=$RECOMPUTE_MODULES \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=1 \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_indexer_loss_coeff=0.0 \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_kernel_backend=$DSA_KERNEL_BACKEND \
  trainer.policy.megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.megatron_config.optimizer_offload_to_disk.dir="$OPTIMIZER_OFFLOAD_DIR" \
  trainer.policy.megatron_config.optimizer_offload_to_disk.moment_storage_dtype=$OPTIMIZER_MOMENT_DTYPE \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.remove_microbatch_padding=true \
  trainer.use_expandable_segments=true \
  trainer.fused_lm_head_logprob=true \
  trainer.logprobs_chunk_size=1024 \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=null \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=$WEIGHT_SYNC_BACKEND \
  generator.inference_engine.distributed_executor_backend="$INFERENCE_EXECUTOR_BACKEND" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.data_parallel_size=$INFERENCE_ENGINE_DATA_PARALLEL_SIZE \
  generator.inference_engine.expert_parallel_size=$INFERENCE_ENGINE_EXPERT_PARALLEL_SIZE \
  generator.inference_engine.max_num_seqs=$INFERENCE_ENGINE_MAX_NUM_SEQS \
  generator.inference_engine.gpu_memory_utilization=$INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.router_init_kwargs="$ROUTER_INIT_KWARGS" \
  generator.inference_engine.speculative_config="$SPECULATIVE_CONFIG" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_k=$TOP_K \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  environment.env_class=aime \
  trainer.mtp.enabled=$MTP_ENABLED \
  trainer.mtp.num_speculative_tokens=$MTP_NUM_SPECULATIVE_TOKENS \
  trainer.mtp.loss_weight=$MTP_LOSS_WEIGHT \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm5p3_dapo" \
  trainer.run_name="$RUN_NAME" \
  "$@"
