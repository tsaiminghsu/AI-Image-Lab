"""Config loading for the Z-Image Colab skill: defaults, override order, the bounds that keep the cost
dials sane, and that neither credentials nor a private Drive path can end up in a config file."""

import json
from pathlib import Path

import pytest
from zimage_colab_fakes import SKILL_DIR, make_config, z

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((SKILL_DIR / "config" / "config.example.json").read_text(encoding="utf-8"))


def load(tmp_path, overrides=None, **env):
    local = tmp_path / "config.json"
    if overrides is not None:
        local.write_text(json.dumps(overrides), encoding="utf-8")
    return z.load_config(local, env=env)


def test_defaults_come_from_the_example_config(tmp_path):
    config = make_config(tmp_path)
    assert config["worker"] == {
        "idle_timeout_seconds": 600,
        "poll_interval_seconds": 5,
        "job_timeout_seconds": 1800,
        "max_session_seconds": 10800,
        "heartbeat_seconds": 5,
    }
    assert config["colab"]["gpu"] == "L4"
    assert config["colab"]["startup_timeout_seconds"] == 900
    assert (config["comfyui"]["host"], config["comfyui"]["port"]) == ("127.0.0.1", 8188)
    assert z.output_root(config) == tmp_path / "out"


def test_the_idle_timeout_is_a_setting_not_a_constant(tmp_path):
    assert load(tmp_path, {"worker": {"idle_timeout_seconds": 90}})["worker"]["idle_timeout_seconds"] == 90
    config = load(tmp_path, {"worker": {"idle_timeout_seconds": 90}}, ZIMG_IDLE_TIMEOUT_SECONDS="120")
    assert config["worker"]["idle_timeout_seconds"] == 120  # the environment beats the file
    assert config["worker"]["job_timeout_seconds"] == 1800  # untouched keys keep their defaults


def test_the_worker_gets_the_configured_timeouts(tmp_path):
    config = load(tmp_path, {"worker": {"idle_timeout_seconds": 75, "job_timeout_seconds": 300}})
    config["output"]["directory"] = str(tmp_path / "out")
    controller = z.Controller(config, transport=z.Transport(), echo=lambda line: None)
    controller.session = "zimg-x"
    run_cfg = controller.run_config()
    assert run_cfg["worker"]["idle_timeout_seconds"] == 75
    job = z.build_job(config, prompt="a red cube")
    assert z.remote_payload(job, config)["timeout_seconds"] == 300


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("worker", "idle_timeout_seconds", 5),
        ("worker", "idle_timeout_seconds", 10**6),
        ("worker", "idle_timeout_seconds", "600"),
        ("worker", "idle_timeout_seconds", True),
        ("worker", "job_timeout_seconds", 1),
        ("worker", "poll_interval_seconds", 0),
        ("colab", "gpu", "RTX2070"),
        ("colab", "drive", "always"),
        ("colab", "drive_consent_wait_seconds", 300),  # drive.mount gives up after 120 s
        ("comfyui", "host", "0.0.0.0"),  # ComfyUI is never exposed as a web service
        ("comfyui", "commit", "v0.38.0"),
        ("model", "revision", "main"),
        ("controller", "heartbeat_timeout_seconds", 10),
    ],
)
def test_out_of_range_values_are_invalid_config(tmp_path, section, key, value):
    with pytest.raises(z.ZImageError) as err:
        load(tmp_path, {section: {key: value}})
    assert err.value.code == "INVALID_CONFIG"


@pytest.mark.parametrize("path", ["/content/drive/MyDrive/AI", "MyDrive/../secrets", "C:\\Users\\me\\Drive", ""])
def test_the_persistent_path_must_be_relative_to_the_drive_mount(tmp_path, path):
    with pytest.raises(z.ZImageError) as err:
        load(tmp_path, {"colab": {"persistent_path": path}})
    assert err.value.code == "INVALID_CONFIG"


def test_the_persistent_path_can_be_changed(tmp_path):
    config = load(tmp_path, ZIMG_PERSISTENT_PATH="MyDrive/Work/zimg")
    config["output"]["directory"] = str(tmp_path / "out")
    controller = z.Controller(config, transport=z.Transport(), echo=lambda line: None)
    assert controller.persistent_root == "/content/drive/MyDrive/Work/zimg"


@pytest.mark.parametrize("key", ["api_key", "oauth_token", "client_secret", "password", "google_credentials"])
def test_secret_like_keys_are_refused(tmp_path, key):
    with pytest.raises(z.ZImageError) as err:
        load(tmp_path, {"colab": {key: "x"}})
    assert err.value.code == "INVALID_CONFIG"
    assert key in str(err.value)


def test_a_broken_local_config_is_reported_not_ignored(tmp_path):
    local = tmp_path / "config.json"
    local.write_text("{not json", encoding="utf-8")
    with pytest.raises(z.ZImageError) as err:
        z.load_config(local, env={})
    assert err.value.code == "INVALID_CONFIG"


def test_a_bad_environment_value_is_invalid_config(tmp_path):
    with pytest.raises(z.ZImageError) as err:
        load(tmp_path, ZIMG_IDLE_TIMEOUT_SECONDS="ten minutes")
    assert err.value.code == "INVALID_CONFIG"


def test_a_model_file_without_a_checksum_is_refused(tmp_path):
    files = json.loads(json.dumps(EXAMPLE["model"]["files"]))
    files[0]["sha256"] = ""
    with pytest.raises(z.ZImageError):
        load(tmp_path, {"model": {"files": files}})
    with pytest.raises(z.ZImageError):  # one unet, one text encoder, one vae - no more, no fewer
        load(tmp_path, {"model": {"files": EXAMPLE["model"]["files"][:2]}})


def test_the_example_config_pins_versions_and_holds_no_secrets():
    assert z.secret_like_keys(EXAMPLE) == []
    assert len(EXAMPLE["comfyui"]["commit"]) == 40 and len(EXAMPLE["model"]["revision"]) == 40
    assert {f["role"] for f in EXAMPLE["model"]["files"]} == {"unet", "text_encoder", "vae"}
    # A relative path under the mount: nobody's private Drive layout is baked into the repo.
    assert not EXAMPLE["colab"]["persistent_path"].startswith("/")


def test_the_local_config_and_credential_files_are_gitignored():
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in (
        ".claude/skills/z-image-colab/config/config.json",
        "credentials/",
        "secrets/",
        "*.token",
        "*.key",
        ".env",
    ):
        assert pattern in ignored, pattern
