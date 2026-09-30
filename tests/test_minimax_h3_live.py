"""Integration tier for the MiniMax H3 Colab skill: a real `colab` round trip on a CPU runtime.

Deselected by default (marker `live`) and skipped unless H3_LIVE_COLAB=1, because it signs in to
Colab and holds a CPU runtime for a minute or two. It proves what the offline fakes can only assume:
the transport works end to end, `--env` reaches a .py file run by `colab exec`, markers stream back
line by line, and what `colab exec` returns when the remote code raises. The real-inference tier (A100,
minutes of GPU) is not a pytest; it is the run.py command in the skill README.

    $env:H3_LIVE_COLAB = "1"; .venv-dev\\Scripts\\python.exe -m pytest -m live tests/test_minimax_h3_live.py
"""

import os
import uuid

import pytest

from minimax_h3_fakes import h3

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("H3_LIVE_COLAB") != "1", reason="set H3_LIVE_COLAB=1 to use a CPU Colab runtime"),
]

SCRIPT = """import json, os
print("H3_STAGE " + json.dumps({"name": "LOADING_MODEL", "env": os.environ.get("H3_JOB_FILE")}), flush=True)
print("H3_OUTPUT " + json.dumps({"path": "/content/h3_live_hello.txt"}), flush=True)
"""

FAILING = """import json
print("H3_ERROR " + json.dumps({"code": "INFERENCE_FAILED", "message": "on purpose"}), flush=True)
raise RuntimeError("on purpose")
"""


@pytest.fixture(scope="module")
def session():
    config = h3.load_config()
    transport = h3.make_transport(config)
    transport.preflight()
    h3.colab_usage(transport)
    name = f"h3-live-{uuid.uuid4().hex[:8]}"
    transport.call(["new", "--session", name], label="colab new (CPU)", timeout=900)
    try:
        yield transport, name
    finally:
        h3.stop_session(transport, name)


def test_upload_exec_with_env_and_download_round_trip(session, tmp_path):
    transport, name = session
    src = tmp_path / "hello.txt"
    src.write_text("hello from the H3 skill\n", encoding="utf-8")
    h3.upload_file(transport, name, src, "/content/h3_live_hello.txt", timeout=300)
    script = tmp_path / "probe.py"
    script.write_text(SCRIPT, encoding="utf-8")
    lines = []
    transport.call(
        ["exec", "--session", name, "--timeout", "300", "--env", "H3_JOB_FILE=/content/h3_live_job.json"]
        + ["--file", transport.cli_path(script, tmp_path)],
        label="colab exec",
        timeout=600,
        mount_dir=tmp_path,
        on_line=lines.append,
    )
    markers = dict(m for m in map(h3.parse_marker, lines) if m)
    assert markers["STAGE"]["env"] == "/content/h3_live_job.json"
    back = tmp_path / "back" / "hello.txt"
    h3.download_file(transport, name, markers["OUTPUT"]["path"], back, timeout=300)
    assert back.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_remote_exception_still_delivers_the_error_marker(session, tmp_path):
    transport, name = session
    script = tmp_path / "failing.py"
    script.write_text(FAILING, encoding="utf-8")
    lines = []
    try:
        transport.call(
            ["exec", "--session", name, "--file", transport.cli_path(script, tmp_path)],
            label="colab exec",
            timeout=600,
            mount_dir=tmp_path,
            on_line=lines.append,
        )
        exit_status = 0
    except h3.ColabCommandError as exc:
        exit_status = exc.returncode
    markers = dict(m for m in map(h3.parse_marker, lines) if m)
    assert markers["ERROR"]["code"] == "INFERENCE_FAILED"
    print(f"colab exec exit status after a remote exception: {exit_status}")
