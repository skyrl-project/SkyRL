"""GPU E2E tests for Megatron LoRA Hugging Face PEFT adapter export.

Verifies:
1. Dense model (`Qwen/Qwen3-0.6B`): training step with Megatron LoRA, exporting
   with `lora_export_format="adapter"` and `lora_export_format="both"`, checking
   directory layouts and file sizes, loading the exported adapter with
   `peft.PeftModel.from_pretrained` and comparing logits against the merged
   export, and comparing greedy vLLM `--enable-lora` generations against the
   merged export.
2. MoE shared-outer LoRA (`eatang/qwen3-moe-tiny-random` with
   `experts_shared_outer_loras=True`): exporting with `expand_shared_outer=True`
   (loadable by `PeftModel.from_pretrained` and matching merged logits) and
   `allow_serving_layout=True`.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_lora_hf_export.py
"""

import gc
import json
import os
import tempfile

import pytest
import ray
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_worker import (
    get_test_training_batch,
)
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

pytestmark = pytest.mark.megatron

DENSE_MODEL = "Qwen/Qwen3-0.6B"
MOE_TINY_MODEL = "eatang/qwen3-moe-tiny-random"
NUM_GPUS = 1


def _make_dense_lora_config(lora_export_format: str) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = DENSE_MODEL
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=16)
    cfg.trainer.policy.optimizer_config.lr = 1e-3
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.logger = "console"
    cfg.trainer.policy.megatron_config.hf_export_config.lora_export_format = lora_export_format
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.generator.n_samples_per_prompt = 1
    validate_cfg(cfg)
    return cfg


def _make_moe_shared_outer_config(
    *,
    lora_export_format: str,
    expand_shared_outer: bool = False,
    allow_serving_layout: bool = False,
) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MOE_TINY_MODEL
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8)
    cfg.trainer.policy.optimizer_config.lr = 1e-3
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.logger = "console"
    cfg.trainer.policy.megatron_config.lora_config.experts_shared_outer_loras = True
    cfg.trainer.policy.megatron_config.hf_export_config.lora_export_format = lora_export_format
    cfg.trainer.policy.megatron_config.hf_export_config.expand_shared_outer = expand_shared_outer
    cfg.trainer.policy.megatron_config.hf_export_config.allow_serving_layout = allow_serving_layout
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.generator.n_samples_per_prompt = 1
    validate_cfg(cfg)
    return cfg


def _make_moe_training_batch(batch_size: int = 4) -> TrainingInputBatch:
    tokenizer = AutoTokenizer.from_pretrained(MOE_TINY_MODEL, trust_remote_code=True)
    sentences = [
        "<|im_start|>system\nYou are a helpful assistant.",
        "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n",
    ] * (batch_size // 2)
    sequences = [tokenizer.encode(s) for s in sentences]
    num_actions = 8
    max_seq_len = max(len(s) for s in sequences) + 4
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attention_masks = []
    for i, seq in enumerate(sequences):
        pad = max_seq_len - len(seq)
        attention_masks.append([1] * len(seq) + [0] * pad)
        sequences[i] = seq + [pad_id] * pad

    sequences = torch.tensor(sequences)
    attention_masks = torch.tensor(attention_masks)
    loss_masks = attention_masks[:, -num_actions:].float()
    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_masks,
            "action_log_probs": torch.full((batch_size, num_actions), 0.1),
            "base_action_log_probs": torch.full((batch_size, num_actions), 0.2),
            "rollout_logprobs": torch.full((batch_size, num_actions), 0.11),
            "values": torch.full((batch_size, num_actions), 0.1),
            "returns": torch.full((batch_size, num_actions), 0.1),
            "advantages": torch.full((batch_size, num_actions), 0.5),
            "loss_mask": loss_masks,
            "response_mask": loss_masks,
        }
    )
    data.metadata = {"response_length": num_actions, "global_step": 0}
    return data


def _dir_safetensors_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(path, name)) for name in os.listdir(path) if name.endswith(".safetensors"))


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_megatron_lora_hf_export_adapter_and_both_match_merged(ray_init_fixture):
    """Export trained Megatron LoRA policy in `both` and `adapter` formats and verify PEFT + vLLM parity."""
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, trust_remote_code=True)
    with tempfile.TemporaryDirectory(prefix="megatron_lora_hf_export_") as tmp_dir:
        both_dir = os.path.join(tmp_dir, "both")
        adapter_only_dir = os.path.join(tmp_dir, "adapter_only")

        cfg = _make_dense_lora_config("both")
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=NUM_GPUS,
            cfg=cfg,
        )

        # Train one step so adapter weights are non-zero.
        batch = get_test_training_batch(4)
        batch.metadata["global_step"] = 0
        ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        # Export `both` (merged in both_dir, adapter in both_dir/lora_adapter).
        ray.get(
            actor_group.async_run_ray_method("pass_through", "save_hf_model", export_dir=both_dir, tokenizer=tokenizer)
        )

        # Release Ray workers before loading HF / vLLM models on the same GPU.
        ray.shutdown()
        _clear_cuda()

        # Verify `both` layout: merged model at root, PEFT adapter at root/lora_adapter.
        both_adapter_dir = os.path.join(both_dir, "lora_adapter")
        assert os.path.exists(os.path.join(both_dir, "config.json"))
        assert os.path.exists(os.path.join(both_dir, "tokenizer.json"))
        assert _dir_safetensors_bytes(both_dir) > 0
        assert os.path.exists(os.path.join(both_adapter_dir, "adapter_config.json"))
        assert os.path.exists(os.path.join(both_adapter_dir, "adapter_model.safetensors"))
        assert os.path.exists(os.path.join(both_adapter_dir, "tokenizer.json"))
        assert not os.path.exists(os.path.join(both_adapter_dir, "config.json"))

        merged_bytes = _dir_safetensors_bytes(both_dir)
        adapter_bytes = _dir_safetensors_bytes(both_adapter_dir)
        assert (
            adapter_bytes < merged_bytes * 0.1
        ), f"Expected adapter ({adapter_bytes} B) to be much smaller than merged model ({merged_bytes} B)"

        with open(os.path.join(both_adapter_dir, "adapter_config.json")) as f:
            adapter_cfg = json.load(f)
        assert adapter_cfg["base_model_name_or_path"] == DENSE_MODEL
        assert adapter_cfg["r"] == 8
        assert adapter_cfg["lora_alpha"] == 16

        # Compare HF logits: base vs merged vs base + PeftModel(adapter).
        prompts = [
            "The capital of France is",
            "2 + 2 equals",
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        base_model = AutoModelForCausalLM.from_pretrained(DENSE_MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
        with torch.no_grad():
            base_logits = base_model(**inputs).logits.float().cpu()

        peft_model = PeftModel.from_pretrained(base_model, both_adapter_dir).eval()
        with torch.no_grad():
            adapter_logits = peft_model(**inputs).logits.float().cpu()
        merged_from_peft = peft_model.merge_and_unload().eval()
        with torch.no_grad():
            peft_merged_logits = merged_from_peft(**inputs).logits.float().cpu()
        del merged_from_peft, peft_model, base_model
        _clear_cuda()

        merged_model = AutoModelForCausalLM.from_pretrained(both_dir, torch_dtype=torch.bfloat16).to("cuda").eval()
        with torch.no_grad():
            merged_logits = merged_model(**inputs).logits.float().cpu()
        del merged_model
        _clear_cuda()

        base_vs_merged_max = (merged_logits - base_logits).abs().max().item()
        adapter_vs_merged_mean = (adapter_logits - merged_logits).abs().mean().item()
        peft_merged_vs_merged_max = (peft_merged_logits - merged_logits).abs().max().item()
        print(
            f"HF logits: base_vs_merged_max={base_vs_merged_max:.4f}, "
            f"adapter_vs_merged_mean={adapter_vs_merged_mean:.6f}, "
            f"peft_merge_and_unload_vs_merged_max={peft_merged_vs_merged_max:.6f}, "
            f"merged_bytes={merged_bytes}, adapter_bytes={adapter_bytes}"
        )
        assert base_vs_merged_max > 1e-2, f"Training step did not move weights noticeably: {base_vs_merged_max}"
        assert (
            adapter_vs_merged_mean < 5e-2
        ), f"Unmerged PEFT adapter mean logit diff too large: {adapter_vs_merged_mean}"
        assert (
            peft_merged_vs_merged_max < 1e-2
        ), f"PEFT merge_and_unload logits diverged from merged export: {peft_merged_vs_merged_max}"

        # Compare vLLM prompt logprobs & greedy next tokens: merged export vs base + exported PEFT adapter.
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        def _extract_prompt_logprobs(outputs):
            values = []
            for out in outputs:
                for token_id, entry in zip(out.prompt_token_ids[1:], out.prompt_logprobs[1:]):
                    values.append(entry[token_id].logprob)
            return torch.tensor(values, dtype=torch.float32)

        sampling = SamplingParams(temperature=0.0, max_tokens=8, prompt_logprobs=1)

        llm_merged = LLM(model=both_dir, dtype="bfloat16", gpu_memory_utilization=0.5, enforce_eager=True)
        merged_res = llm_merged.generate(prompts, sampling)
        merged_lps = _extract_prompt_logprobs(merged_res)
        merged_texts = [out.outputs[0].text for out in merged_res]
        del llm_merged
        _clear_cuda()

        llm_adapter = LLM(
            model=DENSE_MODEL,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=16,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )
        base_res = llm_adapter.generate(prompts, sampling)
        base_lps = _extract_prompt_logprobs(base_res)
        adapter_res = llm_adapter.generate(
            prompts,
            sampling,
            lora_request=LoRARequest("exported", 1, both_adapter_dir),
        )
        adapter_lps = _extract_prompt_logprobs(adapter_res)
        adapter_texts = [out.outputs[0].text for out in adapter_res]
        del llm_adapter
        _clear_cuda()

        vllm_base_vs_merged_max = (merged_lps - base_lps).abs().max().item()
        vllm_adapter_vs_merged_mean = (adapter_lps - merged_lps).abs().mean().item()
        vllm_adapter_vs_merged_max = (adapter_lps - merged_lps).abs().max().item()
        print(
            f"vLLM prompt logprobs: base_vs_merged_max={vllm_base_vs_merged_max:.4f}, "
            f"adapter_vs_merged_mean={vllm_adapter_vs_merged_mean:.6f}, "
            f"adapter_vs_merged_max={vllm_adapter_vs_merged_max:.6f}"
        )
        print(f"vLLM merged outputs:  {merged_texts!r}")
        print(f"vLLM adapter outputs: {adapter_texts!r}")
        assert vllm_base_vs_merged_max > 1e-2, f"vLLM base vs merged diff too small: {vllm_base_vs_merged_max}"
        assert (
            vllm_adapter_vs_merged_mean < 5e-2
        ), f"vLLM adapter vs merged prompt logprobs diverged: {vllm_adapter_vs_merged_mean}"
        assert [out.outputs[0].token_ids[0] for out in adapter_res] == [
            out.outputs[0].token_ids[0] for out in merged_res
        ]

        # Also test `adapter`-only export after a training step writes directly to export_dir.
        from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests

        ray_init_for_tests()
        cfg_adapter = _make_dense_lora_config("adapter")
        actor_group_2 = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=NUM_GPUS,
            cfg=cfg_adapter,
        )
        ray.get(actor_group_2.async_run_ray_method("mesh", "forward_backward", data=batch))
        ray.get(actor_group_2.async_run_ray_method("pass_through", "optim_step"))
        ray.get(
            actor_group_2.async_run_ray_method(
                "pass_through", "save_hf_model", export_dir=adapter_only_dir, tokenizer=tokenizer
            )
        )
        ray.shutdown()
        _clear_cuda()

        assert os.path.exists(os.path.join(adapter_only_dir, "adapter_config.json"))
        assert os.path.exists(os.path.join(adapter_only_dir, "adapter_model.safetensors"))
        assert os.path.exists(os.path.join(adapter_only_dir, "tokenizer.json"))
        assert not os.path.exists(os.path.join(adapter_only_dir, "config.json"))
        assert not os.path.exists(os.path.join(adapter_only_dir, "model.safetensors"))

        base_model_2 = AutoModelForCausalLM.from_pretrained(DENSE_MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
        peft_model_2 = PeftModel.from_pretrained(base_model_2, adapter_only_dir).eval()
        with torch.no_grad():
            adapter_only_logits = peft_model_2(**inputs).logits.float().cpu()
        del peft_model_2, base_model_2
        _clear_cuda()
        assert (adapter_only_logits - base_logits).abs().max().item() > 1e-2


@pytest.mark.parametrize(
    "expand_shared_outer, allow_serving_layout",
    [
        pytest.param(True, False, id="expand_shared_outer"),
        pytest.param(False, True, id="allow_serving_layout"),
    ],
)
def test_megatron_moe_shared_outer_lora_hf_adapter_export(ray_init_fixture, expand_shared_outer, allow_serving_layout):
    """Verify shared-outer MoE LoRA adapter export with `expand_shared_outer` and `allow_serving_layout`."""
    from safetensors.torch import load_file

    tokenizer = AutoTokenizer.from_pretrained(MOE_TINY_MODEL, trust_remote_code=True)
    with tempfile.TemporaryDirectory(prefix="megatron_moe_lora_hf_export_") as tmp_dir:
        export_dir = os.path.join(tmp_dir, "policy")
        cfg = _make_moe_shared_outer_config(
            lora_export_format="both" if expand_shared_outer else "adapter",
            expand_shared_outer=expand_shared_outer,
            allow_serving_layout=allow_serving_layout,
        )
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=NUM_GPUS,
            cfg=cfg,
        )

        batch = _make_moe_training_batch(4)
        ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        ray.get(
            actor_group.async_run_ray_method(
                "pass_through", "save_hf_model", export_dir=export_dir, tokenizer=tokenizer
            )
        )
        ray.shutdown()
        _clear_cuda()

        adapter_dir = os.path.join(export_dir, "lora_adapter") if expand_shared_outer else export_dir
        assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
        weights_file = os.path.join(adapter_dir, "adapter_model.safetensors")
        assert os.path.exists(weights_file)

        state = load_file(weights_file)
        assert state, "Expected non-empty adapter state dict"
        assert any(
            t.abs().max().item() > 0 for k, t in state.items() if "lora_B" in k
        ), "Expected non-zero trained lora_B weights in exported MoE adapter"

        if expand_shared_outer:
            inputs = tokenizer(["What is 2 + 2?"], return_tensors="pt").to("cuda")
            base_model = (
                AutoModelForCausalLM.from_pretrained(MOE_TINY_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
                .to("cuda")
                .eval()
            )
            with torch.no_grad():
                base_logits = base_model(**inputs).logits.float().cpu()
            peft_model = PeftModel.from_pretrained(base_model, adapter_dir).eval()
            with torch.no_grad():
                adapter_logits = peft_model(**inputs).logits.float().cpu()
            merged_from_peft = peft_model.merge_and_unload().eval()
            with torch.no_grad():
                peft_merged_logits = merged_from_peft(**inputs).logits.float().cpu()
            del merged_from_peft, peft_model, base_model
            _clear_cuda()

            merged_model = (
                AutoModelForCausalLM.from_pretrained(export_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
                .to("cuda")
                .eval()
            )
            with torch.no_grad():
                merged_logits = merged_model(**inputs).logits.float().cpu()
            del merged_model
            _clear_cuda()

            base_vs_merged_max = (merged_logits - base_logits).abs().max().item()
            adapter_vs_merged_mean = (adapter_logits - merged_logits).abs().mean().item()
            peft_merged_vs_merged_max = (peft_merged_logits - merged_logits).abs().max().item()
            print(
                f"MoE shared-outer expand_shared_outer: base_vs_merged_max={base_vs_merged_max:.4f}, "
                f"adapter_vs_merged_mean={adapter_vs_merged_mean:.6f}, "
                f"peft_merge_and_unload_vs_merged_max={peft_merged_vs_merged_max:.6f}"
            )
            assert base_vs_merged_max > 1e-2, f"MoE training step did not move weights: {base_vs_merged_max}"
            assert adapter_vs_merged_mean < 5e-2
            assert peft_merged_vs_merged_max < 1e-2
        else:
            # Compact serving layout keeps the shared outer factor as a 3D [1, ...] tensor.
            shared_tensors = [k for k, t in state.items() if t.ndim == 3 and t.shape[0] == 1]
            assert (
                shared_tensors
            ), f"Expected shared [1, ...] 3D tensor(s) in serving layout, got {list(state.keys())[:10]}"
