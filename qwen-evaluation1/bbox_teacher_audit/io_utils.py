"""Crash-safe small-file persistence and run provenance helpers."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    atomic_write_text(path, text)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_sample_record(root: Path, stage: str, sample_id: str, record: Mapping[str, object]) -> Path:
    path = root / "records" / stage / f"{sample_id}.json"
    write_json(path, record)
    return path


def load_sample_records(root: Path, stage: str) -> Dict[str, Dict[str, object]]:
    directory = root / "records" / stage
    if not directory.exists():
        return {}
    output = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        output[str(record["sample_id"])] = record
    return output


def command_output(command: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return (completed.stdout + completed.stderr).strip()


def environment_snapshot(repo_root: Path) -> Dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_branch": command_output(["git", "branch", "--show-current"], repo_root),
        "git_commit": command_output(["git", "rev-parse", "HEAD"], repo_root),
        "git_status_short": command_output(["git", "status", "--short"], repo_root),
    }
