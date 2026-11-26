"""CPU tests verifying FSDP rejects LoRA adapter HF export formats."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.train.config import SkyRLTrainConfig


@pytest.mark.parametrize("fmt", ["adapter", "both"])
def test_fsdp_strategy_save_hf_model_rejects_adapter_export(tmp_path, fmt):
    strategy = FSDPStrategy.__new__(FSDPStrategy)
    with pytest.raises(ValueError, match=r"trainer\.strategy='megatron'"):
        strategy.save_hf_model(
            model=Mock(),
            output_dir=str(tmp_path / "export"),
            tokenizer=None,
            lora_export_format=fmt,
        )


@pytest.mark.parametrize("fmt", ["adapter", "both"])
def test_fsdp_policy_worker_save_hf_model_rejects_adapter_export(tmp_path, fmt):
    cfg = SkyRLTrainConfig().trainer
    cfg.strategy = "fsdp"
    cfg.policy.model.lora.rank = 8
    cfg.policy.megatron_config.hf_export_config.lora_export_format = fmt

    worker = FSDPPolicyWorkerBase.__new__(FSDPPolicyWorkerBase)
    worker.cfg = cfg
    worker.model = Mock()
    worker.strategy = SimpleNamespace(save_hf_model=Mock())

    with pytest.raises(ValueError, match=r"trainer\.strategy='megatron'"):
        worker.save_hf_model(str(tmp_path / "export"), tokenizer=None)
    worker.strategy.save_hf_model.assert_not_called()


def test_fsdp_policy_worker_save_hf_model_allows_merged_default(tmp_path):
    cfg = SkyRLTrainConfig().trainer
    cfg.strategy = "fsdp"
    worker = FSDPPolicyWorkerBase.__new__(FSDPPolicyWorkerBase)
    worker.cfg = cfg
    worker.model = Mock()
    worker.strategy = SimpleNamespace(save_hf_model=Mock())
    tokenizer = Mock()

    worker.save_hf_model(str(tmp_path / "export"), tokenizer=tokenizer)
    worker.strategy.save_hf_model.assert_called_once_with(
        worker.model,
        str(tmp_path / "export"),
        tokenizer=tokenizer,
    )
