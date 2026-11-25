"""HarborTaskDataset: a directory of task directories, listed in a machine-independent order."""

from pathlib import Path
from unittest import mock

from skyrl.integration.harbor.dataset import HarborTaskDataset


def test_task_directories_are_sorted_whatever_the_filesystem_order(tmp_path: Path):
    for name in ["task_b", "task_c", "task_a"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "instruction.md").write_text("do it")
    (tmp_path / "not_a_task").mkdir()
    listed = [tmp_path / "task_c", tmp_path / "not_a_task", tmp_path / "task_a", tmp_path / "task_b"]

    with mock.patch.object(Path, "iterdir", return_value=iter(listed)):
        dataset = HarborTaskDataset(data_files=[str(tmp_path)])

    assert dataset.task_paths == [tmp_path / "task_a", tmp_path / "task_b", tmp_path / "task_c"]
