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
