"""Atomic checkpoint publication for trainers, evaluators, and spectators.

The model file is written before its manifest.  Readers therefore either see
the previous complete generation or the next complete generation, never a
partially-written ``torch.save`` file.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


SCHEMA = "dcss-checkpoint-v1"
CHANNELS = {"candidate", "champion"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_torch_save(value, path: Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_json_write(value, path: Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def manifest_path(data_root: Path, variant: str, channel: str) -> Path:
    if channel not in CHANNELS:
        raise ValueError(f"unknown checkpoint channel: {channel}")
    return Path(data_root) / f"rl_checkpoint.{variant}.{channel}.json"


def publish_manifest(checkpoint: Path, manifest: Path, *, variant: str,
                     channel: str, update: int = 0, architecture: str = "spatial-v1",
                     action_names=(), metrics=None) -> dict:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = {
        "schema": SCHEMA,
        "channel": channel,
        "variant": variant,
        "architecture": architecture,
        "checkpoint": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "update": int(update),
        "action_names": list(action_names),
        "metrics": dict(metrics or {}),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_write(payload, manifest)
    return payload

def read_manifest(path: Path) -> dict:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported checkpoint manifest: {path}")
    checkpoint = Path(payload["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (path.parent / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload["checkpoint"] = str(checkpoint)
    return payload
