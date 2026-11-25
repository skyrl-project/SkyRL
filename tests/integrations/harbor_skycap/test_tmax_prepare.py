"""prepare_tmax.py: only a platform-only compose file is deleted, so those tasks run in Daytona's direct mode."""

import stat
import zipfile
from pathlib import Path

import pytest

from examples.train_integrations.harbor_skycap.tmax import prepare_tmax


def make_task(root: Path, name: str, compose: str = None) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("do it")
    (task / "environment" / "Dockerfile").write_text("FROM --platform=linux/amd64 ubuntu:24.04\n")
    if compose is not None:
        (task / "environment" / "docker-compose.yaml").write_text(compose)
    return task


@pytest.mark.parametrize(
    "compose, dropped",
    [
        ("services:\n  main:\n    platform: linux/amd64\n", True),
        ("services:\n  main: {}\n", True),
        ("services:\n  main:\n    platform: linux/amd64\n    volumes: [./data:/data]\n", False),
        ("services:\n  main: {}\n  db:\n    image: postgres\n", False),
        ("services:\n  main: {}\nnetworks:\n  inner: {}\n", False),
    ],
    ids=["platform-only", "empty-main", "main-with-volume", "second-service", "top-level-network"],
)
def test_only_a_platform_only_compose_file_is_deleted(tmp_path, compose, dropped):
    task = make_task(tmp_path, "task_a", compose)
    assert prepare_tmax.drop_platform_only_compose(task) is dropped
    assert (task / "environment" / "docker-compose.yaml").exists() is not dropped


def test_tasks_are_collected_flat_from_any_depth(tmp_path):
    extracted = tmp_path / "extracted"
    make_task(extracted / "tasks", "task_a", "services:\n  main:\n    platform: linux/amd64\n")
    make_task(extracted / "tasks" / "nested", "task_b", "services:\n  main: {}\n  db: {image: postgres}\n")
    make_task(extracted, "task_c")

    counts = prepare_tmax.collect_tasks(extracted, tmp_path / "out")

    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["task_a", "task_b", "task_c"]
    assert counts == {"tasks": 3, "compose_dropped": 1, "compose_kept": 1}


def test_two_tasks_with_one_name_are_refused(tmp_path):
    make_task(tmp_path / "extracted" / "a", "task_a")
    make_task(tmp_path / "extracted" / "b", "task_a")
    with pytest.raises(ValueError, match="two tasks are named task_a"):
        prepare_tmax.collect_tasks(tmp_path / "extracted", tmp_path / "out")


def test_extraction_keeps_the_verifier_executable(tmp_path):
    archive = tmp_path / "tasks.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("tasks/task_a/tests/test.sh")
        info.external_attr = 0o750 << 16
        bundle.writestr(info, "echo 1 > /logs/verifier/reward.txt\n")

    prepare_tmax.extract_zip(archive, tmp_path / "out")

    mode = (tmp_path / "out" / "tasks" / "task_a" / "tests" / "test.sh").stat().st_mode
    assert mode & stat.S_IXUSR


def test_a_finished_prepare_of_the_same_source_is_not_redone(tmp_path):
    fetches = []

    def fetch(scratch: Path) -> Path:
        fetches.append(scratch)
        make_task(scratch, "task_a", "services:\n  main:\n    platform: linux/amd64\n")
        return scratch

    output = tmp_path / "tmax"
    prepare_tmax.prepare(output, "source@1", fetch)
    prepare_tmax.prepare(output, "source@1", fetch)
    assert len(fetches) == 1
    assert not (output / "task_a" / "environment" / "docker-compose.yaml").exists()

    prepare_tmax.prepare(output, "source@2", fetch)
    assert len(fetches) == 2
    assert (output / prepare_tmax.PREPARED).read_text().strip() == "source@2"
