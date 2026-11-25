"""Tmax's training tasks and TB-Lite's eval tasks as Harbor task directories for Daytona.

Writes two roots under ``--output-dir``:

- ``tmax/``: the 13,565 tasks of Tmax v1.1 (Ivison et al., arXiv 2606.23321), from the Harbor export
  ``TMaxxx/tmax-13k-v1.1-harbor`` at a pinned revision, checked against the export's own sha256.
- ``tblite/``: the 100 tasks of OpenThoughts-TBLite (``openthoughts-tblite@2.0`` in Harbor's registry), which
  Tmax selects checkpoints on.

Every Tmax task ships an ``environment/docker-compose.yaml`` whose only content pins the main service's
platform, which its Dockerfile already pins. Harbor's Daytona environment runs any task with a compose file in
compose mode: a Docker-in-Docker sandbox that builds the image with ``docker compose`` on every trial. Such a
platform-only compose file is deleted, so the task runs in direct mode (Daytona builds the Dockerfile itself,
and ``auto_snapshot`` reuses that build). A compose file that does anything else is kept.

    uv run --isolated --with huggingface_hub --with pyyaml \\
        examples/train_integrations/harbor_skycap/tmax/prepare_tmax.py --output-dir ~/data/tmax

A root that finished preparing records its source in ``.prepared`` and is skipped next time.
"""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict

import yaml

TMAX_REPO = "TMaxxx/tmax-13k-v1.1-harbor"
TMAX_REVISION = "5305c26042bf4c312fe24cb1b6e3191a0057040a"
TBLITE_REPO = "open-thoughts/OpenThoughts-TBLite"
TBLITE_COMMIT = "f075e463472c7790b85793b392dff1fff20cc0e3"
PREPARED = ".prepared"


def is_platform_only(compose: Dict) -> bool:
    """Whether a docker-compose.yaml does nothing but pin the main service's platform."""
    services = (compose or {}).get("services") or {}
    return (
        set(compose or {}) == {"services"} and set(services) == {"main"} and set(services["main"] or {}) <= {"platform"}
    )


def drop_platform_only_compose(task_dir: Path) -> bool:
    """Delete the task's compose file when it only pins the platform; whether it was deleted."""
    compose = task_dir / "environment" / "docker-compose.yaml"
    if not compose.is_file() or not is_platform_only(yaml.safe_load(compose.read_text())):
        return False
    compose.unlink()
    return True


def collect_tasks(extracted: Path, output: Path) -> Dict[str, int]:
    """Move every task directory (one holding a ``task.toml``) under ``extracted`` flat into ``output``."""
    output.mkdir(parents=True)
    counts = {"tasks": 0, "compose_dropped": 0, "compose_kept": 0}
    for toml in sorted(extracted.rglob("task.toml")):
        task_dir = toml.parent
        target = output / task_dir.name
        if target.exists():
            raise ValueError(f"two tasks are named {task_dir.name}")
        shutil.move(str(task_dir), target)
        counts["tasks"] += 1
        if drop_platform_only_compose(target):
            counts["compose_dropped"] += 1
        elif (target / "environment" / "docker-compose.yaml").is_file():
            counts["compose_kept"] += 1
    return counts


def extract_zip(archive: Path, output: Path) -> None:
    """Extract ``archive`` keeping each file's mode (test.sh must stay executable), refusing paths outside it."""
    output = output.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = (output / member.filename).resolve()
            if not path.is_relative_to(output):
                raise ValueError(f"unsafe path in {archive}: {member.filename}")
            bundle.extract(member, output)
            mode = (member.external_attr >> 16) & 0o777
            if mode:
                path.chmod(mode)


def fetch_tmax(scratch: Path) -> Path:
    from huggingface_hub import hf_hub_download

    def download(name: str) -> Path:
        return Path(hf_hub_download(TMAX_REPO, name, repo_type="dataset", revision=TMAX_REVISION))

    archive = download("tasks.zip")
    expected = json.loads(download("dataset_release.json").read_text())["tasks_zip_sha256"]
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"{archive} does not match the export's tasks_zip_sha256")
    extract_zip(archive, scratch)
    return scratch


def fetch_tblite(scratch: Path) -> Path:
    url = f"https://codeload.github.com/{TBLITE_REPO}/tar.gz/{TBLITE_COMMIT}"
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 - GitHub
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as bundle:
        bundle.extractall(scratch, filter="data")
    return scratch


def prepare(output: Path, source: str, fetch: Callable[[Path], Path]) -> None:
    """Fetch ``source`` and write its tasks to ``output``, unless a finished prepare of the same source is there."""
    marker = output / PREPARED
    if marker.is_file() and marker.read_text().strip() == source:
        print(f"==> {output}: already prepared from {source}")
        return
    partial = output.with_name(output.name + ".partial")
    scratch = output.with_name(output.name + ".download")
    for path in (partial, scratch):
        shutil.rmtree(path, ignore_errors=True)
    scratch.mkdir(parents=True)
    counts = collect_tasks(fetch(scratch), partial)
    shutil.rmtree(scratch)
    (partial / PREPARED).write_text(source + "\n")
    shutil.rmtree(output, ignore_errors=True)
    partial.rename(output)
    print(f"==> {output}: {counts['tasks']} tasks from {source}")
    print(
        f"    platform-only compose files deleted: {counts['compose_dropped']}, other compose files kept: {counts['compose_kept']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.expanduser()
    prepare(root / "tmax", f"hf://datasets/{TMAX_REPO}@{TMAX_REVISION}", fetch_tmax)
    prepare(root / "tblite", f"github://{TBLITE_REPO}@{TBLITE_COMMIT}", fetch_tblite)


if __name__ == "__main__":
    main()
