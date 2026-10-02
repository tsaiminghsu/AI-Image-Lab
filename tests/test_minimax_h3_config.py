"""Config loading for the MiniMax H3 Colab skill: defaults, override order, and that credentials can
never be put in a config file (they live in google-colab-cli's own config volume)."""

import json
from pathlib import Path

import pytest

from minimax_h3_fakes import SKILL_DIR, h3, import_skill, make_config

REPO = Path(__file__).resolve().parents[1]


def test_defaults_come_from_the_example_config(tmp_path):
    config = make_config(tmp_path)
    assert config["colab"]["gpu"] == "A100"
    assert config["colab"]["high_mem"] is True
    assert config["colab"]["timeout_seconds"] == 3600
    assert config["colab"]["transport"] == "auto"
    assert config["colab"]["auth"] == "oauth2"
    assert config["video"] == {"default_duration": 8, "short_side": 768, "max_megapixels": 1.2}
    assert h3.output_dir(config) == tmp_path / "out"


def test_local_file_overrides_defaults_and_env_overrides_the_file(tmp_path):
    local = tmp_path / "config.json"
    local.write_text(json.dumps({"colab": {"gpu": "L4", "timeout_seconds": 1800}}), encoding="utf-8")
    config = h3.load_config(local, env={})
    assert config["colab"]["gpu"] == "L4"
    assert config["colab"]["timeout_seconds"] == 1800
    assert config["colab"]["high_mem"] is True  # untouched keys keep their defaults
    config = h3.load_config(local, env={"H3_GPU": "h100", "H3_HIGH_MEM": "0"})
    assert config["colab"]["gpu"] == "H100"
    assert config["colab"]["high_mem"] is False


def test_command_line_flags_override_config(tmp_path):
    run = import_skill("run")
    args = run.build_parser().parse_args(
        ["--image", "x.png", "--prompt", "p", "--gpu", "L4", "--no-high-mem", "--timeout", "900", "--duration", "5"]
    )
    spec = run.spec_from_args(args)
    assert (spec.gpu, spec.high_mem, spec.timeout, spec.duration) == ("L4", False, 900, 5.0)


@pytest.mark.parametrize("key", ["api_key", "oauth_token", "client_secret", "password", "google_credentials"])
def test_secret_like_keys_are_refused(tmp_path, key):
    local = tmp_path / "config.json"
    local.write_text(json.dumps({"colab": {key: "x"}}), encoding="utf-8")
    with pytest.raises(h3.InputError) as err:
        h3.load_config(local, env={})
    assert err.value.code == "INVALID_CONFIG"
    assert key in str(err.value)


@pytest.mark.parametrize(
    "override",
    [
        {"colab": {"timeout_seconds": 30}},
        {"colab": {"timeout_seconds": "3600"}},
        {"colab": {"gpu": "V100"}},
        {"colab": {"transport": "wsl"}},
        {"colab": {"high_mem": "yes"}},
        {"video": {"default_duration": 20}},
        {"video": {"short_side": 770}},
    ],
)
def test_invalid_values_are_rejected(tmp_path, override):
    local = tmp_path / "config.json"
    local.write_text(json.dumps(override), encoding="utf-8")
    with pytest.raises(h3.InputError) as err:
        h3.load_config(local, env={})
    assert err.value.code == "INVALID_CONFIG"


def test_invalid_env_value_is_rejected(tmp_path):
    with pytest.raises(h3.InputError):
        make_config(tmp_path, H3_HIGH_MEM="maybe")
    with pytest.raises(h3.InputError):
        make_config(tmp_path, H3_TIMEOUT_SECONDS="soon")


def test_example_config_holds_no_secrets_and_validates():
    example = json.loads((SKILL_DIR / "config" / "config.example.json").read_text(encoding="utf-8"))
    assert h3.secret_like_keys(example) == []
    h3.validate_config(example)


def test_local_config_outputs_and_secrets_are_gitignored():
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in ("/output/", "/input/", ".claude/skills/minimax-h3-colab/config/config.json", ".env", "token.json"):
        assert pattern in ignored, pattern


def test_relative_config_paths_resolve_against_the_repo_root():
    assert h3.project_path("./output") == (REPO / "output").resolve()
    assert h3.project_path(REPO / "x") == REPO / "x"


def test_missing_configured_ffprobe_is_reported_as_missing(tmp_path):
    config = make_config(tmp_path, H3_FFPROBE=str(tmp_path / "nope" / "ffprobe.exe"))
    assert h3.find_ffprobe(config) is None


def test_settle_wait_defaults_to_two_minutes_and_is_validated(tmp_path):
    assert h3.load_config(tmp_path / "none.json", env={})["colab"]["cu_settle_seconds"] == 120
    assert make_config(tmp_path, H3_CU_SETTLE_SECONDS="45")["colab"]["cu_settle_seconds"] == 45
    local = tmp_path / "config.json"
    local.write_text(json.dumps({"colab": {"cu_settle_seconds": -1}}), encoding="utf-8")
    with pytest.raises(h3.InputError) as err:
        h3.load_config(local, env={})
    assert err.value.code == "INVALID_CONFIG"


def test_exec_idle_limit_defaults_to_fifteen_minutes_and_is_validated(tmp_path):
    assert h3.load_config(tmp_path / "none.json", env={})["colab"]["exec_idle_timeout_seconds"] == 900
    local = tmp_path / "config.json"
    local.write_text(json.dumps({"colab": {"exec_idle_timeout_seconds": 60}}), encoding="utf-8")
    with pytest.raises(h3.InputError):
        h3.load_config(local, env={})


def test_model_source_defaults_to_drive_and_is_validated(tmp_path):
    assert h3.load_config(tmp_path / "none.json", env={})["drive"]["model_source"] == "drive"
    assert make_config(tmp_path, H3_MODEL_SOURCE="download")["drive"]["model_source"] == "download"
    with pytest.raises(h3.InputError) as err:
        make_config(tmp_path, H3_MODEL_SOURCE="copy")
    assert err.value.code == "INVALID_CONFIG"
