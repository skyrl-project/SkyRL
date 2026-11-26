"""CPU tests for DistributedStrategy.config_with_export_dtype.

The HF export writes this config as ``config.json``, so the dtype it records must match the
dtype of the exported weights.

uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/distributed/test_config_with_export_dtype.py
"""

import json

import torch
from transformers import Qwen3Config

from skyrl.backends.skyrl_train.distributed.strategy import DistributedStrategy


def test_none_returns_the_same_config():
    config = Qwen3Config(dtype=torch.float32)
    assert DistributedStrategy.config_with_export_dtype(config, None) is config


def test_copy_records_export_dtype_and_leaves_original(tmp_path):
    config = Qwen3Config(dtype=torch.float32)
    exported = DistributedStrategy.config_with_export_dtype(config, torch.bfloat16)

    assert exported is not config
    assert config.dtype == torch.float32
    exported.save_pretrained(tmp_path)
    with open(tmp_path / "config.json") as f:
        assert json.load(f)["dtype"] == "bfloat16"
