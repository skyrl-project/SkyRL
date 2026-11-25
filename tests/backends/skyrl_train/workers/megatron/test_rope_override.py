"""A RoPE override on the Megatron backend reaches the provider Megatron-Bridge builds, or fails loudly.

Each case builds a real provider the way ``MegatronWorker.init_configs`` does: ``AutoBridge.from_hf_pretrained``
on a checkpoint directory (here a tiny config saved to ``tmp_path``, no weights), ``apply_rope_override``,
``to_megatron_provider``, then ``check_rope_override``.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/workers/megatron/test_rope_override.py
"""

import pytest

pytest.importorskip("megatron.bridge", reason="requires the megatron extra")

pytestmark = pytest.mark.megatron

from loguru import logger  # noqa: E402
from megatron.bridge import AutoBridge  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    DeepseekV3Config,
    Exaone4Config,
    Glm4MoeConfig,
    GptOssConfig,
    LlamaConfig,
    Qwen2Config,
)

from skyrl.backends.skyrl_train.workers.megatron.model_bridges import (  # noqa: E402
    apply_rope_override,
    check_rope_override,
)

# Values differ from every provider default (GPT and MLA), so each comparison can fail.
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024, "rope_theta": 1_000_000.0}
YARN_ALL_KEYS = {**YARN, "beta_fast": 16.0, "beta_slow": 2.0, "mscale": 2.0, "mscale_all_dim": 1.0}
GPT_OSS_YARN = {
    "rope_type": "yarn",
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "truncate": False,
    "original_max_position_embeddings": 4096,
    "rope_theta": 300_000.0,
}
LINEAR = {"rope_type": "linear", "factor": 2.0, "rope_theta": 1_000_000.0}
# Megatron Core applies llama3 scaling with fixed low/high frequency factors and original length; only the factor
# is free (check_rope_override rejects anything else).
LLAMA3 = {
    "rope_type": "llama3",
    "factor": 16.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_theta": 500_000.0,
}
_TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, max_position_embeddings=4096)


def _qwen2():
    return Qwen2Config(num_attention_heads=4, num_key_value_heads=2, architectures=["Qwen2ForCausalLM"], **_TINY)


def _llama():
    return LlamaConfig(num_attention_heads=4, num_key_value_heads=2, architectures=["LlamaForCausalLM"], **_TINY)


def _deepseek_v3():
    # MLA attention without query compression (q_lora_rank=None, like Moonlight) and a minimal MoE.
    return DeepseekV3Config(
        num_attention_heads=4, num_key_value_heads=4, q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=8,
        qk_nope_head_dim=8, v_head_dim=8, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        moe_intermediate_size=32, first_k_dense_replace=1, n_group=1, topk_group=1,
        architectures=["DeepseekV3ForCausalLM"], **_TINY,
    )  # fmt: skip


def _gpt_oss():
    return GptOssConfig(
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_local_experts=4, num_experts_per_tok=2,
        architectures=["GptOssForCausalLM"], **_TINY,
    )  # fmt: skip


def _glm4_moe():
    return Glm4MoeConfig(
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=1, moe_intermediate_size=32, first_k_dense_replace=1, n_group=1, topk_group=1,
        architectures=["Glm4MoeForCausalLM"], **_TINY,
    )  # fmt: skip


def _exaone4():
    return Exaone4Config(
        num_attention_heads=4, num_key_value_heads=2, sliding_window=None, layer_types=["full_attention"] * 2,
        architectures=["Exaone4ForCausalLM"], **_TINY,
    )  # fmt: skip


def _bridge(config, tmp_path):
    # The call init_configs makes; weights load lazily and to_megatron_provider(load_weights=False) never asks.
    config.save_pretrained(tmp_path)
    return AutoBridge.from_hf_pretrained(tmp_path)


@pytest.mark.parametrize(
    "make_config, override, match",
    [
        pytest.param(_qwen2, {**YARN_ALL_KEYS, "truncate": False}, None, id="qwen2-yarn-all-keys"),
        pytest.param(
            _qwen2,
            {**YARN, "mscale": 2.0, "mscale_all_dim": 0.0},
            "with either at 0, Megatron Core and vLLM scale the attention differently",
            id="qwen2-yarn-mscale-zero",
        ),
        pytest.param(_deepseek_v3, YARN_ALL_KEYS, None, id="deepseek-v3-mla-yarn-all-keys"),
        pytest.param(
            _deepseek_v3,
            {**YARN, "partial_rotary_factor": 0.5},
            "on a multi-latent-attention model",
            id="deepseek-v3-mla-partial-rotary",
        ),
        pytest.param(_gpt_oss, GPT_OSS_YARN, None, id="gpt-oss-yarn"),
        pytest.param(
            _gpt_oss,
            {**GPT_OSS_YARN, "beta_fast": 16.0},
            "GPTOSSBridge does not translate it",
            id="gpt-oss-overwritten",
        ),
        pytest.param(_glm4_moe, YARN, "rotates 50% of each attention head", id="glm4-moe-yarn-partial-rotary"),
        pytest.param(
            _glm4_moe,
            {"rope_type": "default", "rope_theta": 500_000.0, "partial_rotary_factor": 0.5},
            None,
            id="glm4-moe-theta-restating-partial-rotary",
        ),
        pytest.param(
            _glm4_moe,
            {"rope_type": "default", "rope_theta": 500_000.0, "partial_rotary_factor": 1.0},
            "GLM45Bridge does not translate it",
            id="glm4-moe-partial-rotary-changed",
        ),
        pytest.param(_llama, LINEAR, None, id="llama-linear"),
        pytest.param(_llama, LLAMA3, None, id="llama-llama3"),
        pytest.param(
            _exaone4,
            {**LLAMA3, "low_freq_factor": 2.0},
            "but Megatron Core applies llama3 with",
            id="exaone4-llama3-unfixed",
        ),
        pytest.param(
            _llama,
            {"rope_type": "llama3", "factor": 16.0, "rope_theta": 500_000.0},
            "needs 'low_freq_factor', 'high_freq_factor', 'original_max_position_embeddings'",
            id="llama3-no-fixed-keys",
        ),
        pytest.param(_qwen2, LINEAR, "Qwen2Bridge does not translate it", id="qwen2-linear-dropped"),
        pytest.param(_qwen2, {"rope_type": "dynamic", "factor": 2.0}, "'dynamic' is not supported", id="unknown-type"),
        pytest.param(_qwen2, {"rope_type": "yarn", "rope_theta": 1_000_000.0}, "needs a 'factor'", id="no-factor"),
        pytest.param(
            _qwen2,
            {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096},
            "needs 'rope_theta'",
            id="no-theta",
        ),
        pytest.param(
            _qwen2, {**YARN, "attention_factor": 1.0}, "does not translate for rope_type='yarn'", id="untranslated-key"
        ),
        pytest.param(
            _qwen2,
            {"rope_type": "yarn", "factor": 4.0, "rope_theta": 1_000_000.0},
            "needs 'original_max_position_embeddings'",
            id="yarn-no-original-length",
        ),
    ],
)
def test_rope_override_reaches_the_provider_or_raises(tmp_path, make_config, override, match):
    bridge = _bridge(make_config(), tmp_path)
    apply_rope_override(bridge.hf_pretrained.config, override)

    def build_and_check():
        provider = bridge.to_megatron_provider(load_weights=False)
        check_rope_override(provider, override, type(bridge._model_bridge).__name__)

    if match is None:
        build_and_check()
    else:
        with pytest.raises(ValueError, match=match):
            build_and_check()


@pytest.mark.parametrize(
    "make_config, override, warns",
    [pytest.param(_qwen2, YARN, False, id="verified"), pytest.param(_llama, LINEAR, True, id="unverified")],
)
def test_rope_override_without_a_parity_result_warns(tmp_path, make_config, override, warns):
    bridge = _bridge(make_config(), tmp_path)
    apply_rope_override(bridge.hf_pretrained.config, override)
    provider = bridge.to_megatron_provider(load_weights=False)
    messages = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        check_rope_override(provider, override, type(bridge._model_bridge).__name__)
    finally:
        logger.remove(sink)
    assert any("best-effort" in message for message in messages) == warns


def _bridge_with_top_level_rope_theta(tmp_path):
    bridge = _bridge(_qwen2(), tmp_path)
    # On the loaded config: transformers v5 folds a saved top-level rope_theta into rope_parameters on load.
    bridge.hf_pretrained.config.rope_theta = 10_000.0
    return bridge


def test_theta_written_only_into_rope_parameters_loses_to_a_top_level_rope_theta(tmp_path):
    # A top-level rope_theta attribute (remote-code configs keep one) beats a theta written only into rope_parameters.
    bridge = _bridge_with_top_level_rope_theta(tmp_path)
    bridge.hf_pretrained.config.rope_parameters = dict(YARN)
    with pytest.raises(ValueError, match="Qwen2Bridge does not translate it"):
        check_rope_override(bridge.to_megatron_provider(load_weights=False), YARN, "Qwen2Bridge")

    bridge = _bridge_with_top_level_rope_theta(tmp_path)
    apply_rope_override(bridge.hf_pretrained.config, YARN)
    check_rope_override(bridge.to_megatron_provider(load_weights=False), YARN, "Qwen2Bridge")


def test_exported_config_keeps_the_rope_override(tmp_path):
    # init_configs applies the override to the config save_hf_configs writes with save_pretrained, so a model trained
    # under it is exported with it and reloads with the same positions.
    config = _qwen2()
    apply_rope_override(config, YARN)
    config.save_pretrained(tmp_path)

    reloaded = AutoConfig.from_pretrained(tmp_path)
    assert {key: reloaded.rope_parameters.get(key) for key in YARN} == YARN
