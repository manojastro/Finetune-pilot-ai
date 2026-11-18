"""API endpoints with TestClient and dry-run sessions (background tasks run inline in tests)."""

import pytest
from fastapi.testclient import TestClient

from finetunepilot import __version__
from finetunepilot.api.app import ApiSettings, create_app
from finetunepilot.runlog import RunLog, write_run
from tests.conftest import REPO_ROOT


@pytest.fixture
def settings(tmp_path):
    return ApiSettings(
        runs_dir=str(tmp_path / "runs"), reports_dir=str(tmp_path / "reports"),
        sessions_dir=str(tmp_path / "sessions"), output_root=str(tmp_path / "out"),
        agent_config=str(REPO_ROOT / "configs" / "agent.yaml"),
    )  # fmt: skip


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


DATASET = str(REPO_ROOT / "data" / "sample")


def test_health(client) -> None:
    assert client.get("/health").json() == {"status": "ok", "version": __version__}


def test_ui_page(client) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "FineTunePilot" in page.text and "<script>" in page.text


def test_dry_run_session_end_to_end(client) -> None:
    created = client.post(
        "/sessions", json={"objective": "tool calling", "max_runs": 2, "dataset_dir": DATASET}
    )
    assert created.status_code == 202
    sid = created.json()["session_id"]

    (brief,) = client.get("/sessions").json()
    assert brief["session_id"] == sid and brief["status"] == "completed" and brief["dry_run"]
    assert brief["runs"] == 2 and brief["best_args_exact_match"] is not None

    detail = client.get(f"/sessions/{sid}").json()
    assert detail["session"]["status"] == "completed"
    assert "DRY RUN" in detail["report_markdown"]

    run_id = detail["session"]["runs"][0]["train_run_id"]
    run = client.get(f"/runs/{run_id}").json()
    assert run["run_id"] == run_id and run["kind"] == "train" and run["dry_run"] is True


def test_real_sessions_are_refused_unless_allowed(client, settings) -> None:
    body = {"objective": "tool calling", "dry_run": False, "dataset_dir": DATASET}
    assert client.post("/sessions", json=body).status_code == 403

    calls = []

    def fake_deps(agent_settings):  # never build a real LLM client in tests
        calls.append(agent_settings)
        raise RuntimeError("no GPU here")

    allowed = TestClient(
        create_app(settings.model_copy(update={"allow_real": True}), deps_factory=fake_deps)
    )
    created = allowed.post("/sessions", json=body)
    assert created.status_code == 202 and calls and calls[0].auto_approve is True
    failed = allowed.get(f"/sessions/{created.json()['session_id']}").json()["session"]
    assert failed["status"] == "failed" and "no GPU here" in failed["error"]


def test_bad_requests(client) -> None:
    assert client.post("/sessions", json={"objective": "x"}).status_code == 422  # too short
    assert (
        client.post(
            "/sessions", json={"objective": "tool calling", "dataset_dir": "/nope"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/sessions", json={"objective": "tool calling", "gpu": "h100", "dataset_dir": DATASET}
        ).status_code
        == 422
    )
    assert (
        client.post("/sessions", json={"objective": "tool calling", "extra": 1}).status_code == 422
    )


def test_unknown_ids_and_path_traversal(client, settings, tmp_path) -> None:
    assert client.get("/sessions/s-missing").status_code == 404
    assert client.get("/runs/missing").status_code == 404
    (tmp_path / "secret.json").write_text("{}")
    assert client.get("/runs/..%2Fsecret").status_code == 404
    assert client.get("/sessions/..%2F..%2Fsecret").status_code == 404


def test_run_endpoint_returns_a_run_log(client, settings) -> None:
    write_run(
        RunLog(run_id="eval-x", kind="eval", name="x", status="success", metrics={"n": 1}),
        settings.runs_dir,
    )
    assert client.get("/runs/eval-x").json()["metrics"] == {"n": 1}


def test_empty_session_list(client) -> None:
    assert client.get("/sessions").json() == []
