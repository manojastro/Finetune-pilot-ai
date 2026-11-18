"""Shared test setup. Enforces CLAUDE.md rule 2: tests never touch the network."""

import os
import socket

import pytest

# Make Hugging Face libraries fail fast instead of trying to download anything.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to open a network connection."""

    def guard(*args, **kwargs):
        raise RuntimeError("Network access is not allowed in tests (CLAUDE.md rule 2).")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "data" / "sample" / "train.jsonl"
TEST_PATH = REPO_ROOT / "data" / "sample" / "test.jsonl"
CONFIG_PATH = REPO_ROOT / "configs" / "qwen05b_toolcall.yaml"


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """Offline dry-run tokenizer with the real Qwen2.5 chat template (built once per session)."""
    from finetunepilot.dryrun import build_tiny_tokenizer

    return build_tiny_tokenizer([TRAIN_PATH])


@pytest.fixture
def train_config_dict(tmp_path):
    """A minimal valid train config pointing at the sample data."""
    return {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_path": str(TRAIN_PATH),
        "output_dir": str(tmp_path / "out"),
    }
