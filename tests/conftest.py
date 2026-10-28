"""Keep tests away from the real home directory: no automatic skill installation into ~/.claude or ~/.codex, and no
job history written to ~/.kjob (tests that check the history point HISTORY_FILE somewhere else themselves)."""
import os
import tempfile
from pathlib import Path

import pytest

os.environ["KJOB_NO_SKILLS"] = "1"


@pytest.fixture(autouse=True)
def _private_history(monkeypatch):
    from kjob import core
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(core, "HISTORY_FILE", Path(d) / "history.jsonl")
        yield


@pytest.fixture(autouse=True)
def _no_reconnect_pause(monkeypatch):
    # Most mocked streams end without the wrapper's exit line, as a killed container's does.
    from kjob import core
    monkeypatch.setattr(core, "RECONNECT_SECONDS", 0)


@pytest.fixture(autouse=True)
def _no_real_kubernetes_requests(monkeypatch):
    from kubernetes import client

    def reject(*args, **kwargs):
        raise AssertionError("Offline tests must mock every Kubernetes API call")

    monkeypatch.setattr(client.ApiClient, "call_api", reject)
