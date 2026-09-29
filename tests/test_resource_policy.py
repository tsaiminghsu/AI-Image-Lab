"""resource_policy: the rules that pick local / wait / defer / cloud for a generation.

Every test feeds decide() a hand-built Snapshot, so nothing here touches nvidia-smi, ComfyUI or
psutil. The order of the rules matters as much as each rule (a job that is both outside its time
window and too hot must be deferred, not made to wait), so several tests stack two conditions.
"""

import datetime
import json
import subprocess
import types

import pytest

import resource_policy as rp

NOON = datetime.datetime(2026, 9, 27, 12, 0)


def settings(**overrides):
    return {**rp.DEFAULT_SETTINGS, **overrides}


def calm(**overrides):
    """A cool, idle machine with plenty of RAM and ComfyUI warm."""
    base = dict(
        gpu_temp_c=50.0,
        gpu_throttling=False,
        vram_used_gb=2.0,
        vram_total_gb=8.0,
        ram_available_gb=20.0,
        comfy_running=True,
        comfy_rss_gb=4.0,
        comfy_queue=0,
        local_busy_seconds=0.0,
    )
    base.update(overrides)
    return rp.Snapshot(**base)


# --- the rules ----------------------------------------------------------------------------------


def test_a_calm_machine_runs_locally():
    d = rp.decide("txt2img", settings(), calm(), NOON)
    assert d.route == rp.LOCAL
    assert d.reasons == []
    assert d.cloud_target is None


def test_wan_is_always_cloud_even_when_forced_local():
    d = rp.decide("wan_i2v", settings(), calm(), NOON, force_local=True)
    assert d.route == rp.CLOUD
    assert d.cloud_target == rp.CLOUD_WAN
    assert "本地跑不了" in d.reasons[0]


def test_hq_full_short_on_ram_suggests_the_fp8_variant_first():
    # headroom = 14 + 4 - 3 = 15 GB: under the full path's 15.8, over fp8's 13.2
    d = rp.decide("txt2img_hq_full", settings(), calm(ram_available_gb=14.0), NOON)
    assert d.route == rp.LOCAL
    assert d.suggest_quant
    assert "量化版" in d.reasons[0]


def test_hq_full_with_too_little_ram_even_for_fp8_suggests_cloud():
    d = rp.decide("txt2img_hq_full", settings(), calm(ram_available_gb=8.0), NOON)
    assert d.route == rp.CLOUD
    assert d.cloud_target == rp.CLOUD_WORKFLOW
    assert not d.suggest_quant


def test_comfyui_rss_counts_as_headroom_because_it_is_already_resident():
    # 11 GB free alone would not fit fp8 HQ (13.2), but ComfyUI already holds 6 of what it needs
    assert (
        rp.decide("txt2img_hq_quant", settings(), calm(ram_available_gb=11.0, comfy_rss_gb=6.0), NOON).route == rp.LOCAL
    )
    assert (
        rp.decide("txt2img_hq_quant", settings(), calm(ram_available_gb=11.0, comfy_rss_gb=0.0), NOON).route == rp.CLOUD
    )


def test_ram_shortage_without_a_cloud_route_warns_but_runs():
    d = rp.decide("svd", settings(), calm(ram_available_gb=4.0, comfy_rss_gb=0.0), NOON)
    assert d.route == rp.LOCAL
    assert "分頁檔" in d.reasons[0]


def test_force_local_ignores_the_ram_advice():
    d = rp.decide("txt2img_hq_full", settings(), calm(ram_available_gb=8.0), NOON, force_local=True)
    assert d.route == rp.LOCAL
    assert d.reasons == []


def test_unknown_ram_is_not_a_reason_to_block():
    d = rp.decide("txt2img_hq_full", settings(), calm(ram_available_gb=None, comfy_rss_gb=None), NOON)
    assert d.route == rp.LOCAL


def test_vram_above_the_card_suggests_cloud_when_the_profile_is_smaller():
    d = rp.decide("animatediff", settings(gpu_vram_gb=6.0), calm(), NOON)
    assert d.route == rp.CLOUD
    assert d.cloud_target == rp.CLOUD_ANIMATEDIFF


def test_another_program_holding_vram_makes_it_wait():
    d = rp.decide("animatediff", settings(), calm(comfy_running=False, comfy_rss_gb=None, vram_used_gb=5.0), NOON)
    assert d.route == rp.WAIT
    assert "其他程式" in d.reasons[-1]


def test_vram_in_use_by_comfyui_itself_is_not_another_program():
    d = rp.decide("animatediff", settings(), calm(comfy_running=True, vram_used_gb=7.5), NOON)
    assert d.route == rp.LOCAL


def test_a_long_local_backlog_suggests_cloud_only_where_a_route_exists():
    busy = calm(local_busy_seconds=4000)
    assert rp.decide("animatediff", settings(), busy, NOON).route == rp.CLOUD
    assert rp.decide("sadtalker", settings(), busy, NOON).route == rp.LOCAL


def test_outside_the_video_window_defers_to_the_next_start():
    d = rp.decide("animatediff", settings(video_windows=["23:00-08:00"]), calm(), NOON)
    assert d.route == rp.DEFER
    assert datetime.datetime.fromtimestamp(d.not_before) == datetime.datetime(2026, 9, 27, 23, 0)


def test_image_jobs_are_not_held_by_the_video_window():
    assert rp.decide("txt2img", settings(video_windows=["23:00-08:00"]), calm(), NOON).route == rp.LOCAL


def test_force_local_ignores_the_window():
    assert (
        rp.decide("animatediff", settings(video_windows=["23:00-08:00"]), calm(), NOON, force_local=True).route
        == rp.LOCAL
    )


def test_hot_gpu_waits():
    d = rp.decide("txt2img", settings(), calm(gpu_temp_c=80.0), NOON)
    assert d.route == rp.WAIT
    assert "80°C" in d.reasons[0]


def test_throttle_flag_waits_even_below_the_threshold():
    assert rp.decide("txt2img", settings(), calm(gpu_temp_c=50.0, gpu_throttling=True), NOON).route == rp.WAIT


def test_default_start_threshold_is_55c():
    """Measured on this card: an image started at 68C still throttles on the way; one started at
    55C does not. The rule is temp >= threshold -> wait, so 60C and 55C wait and 54C starts."""
    assert rp.DEFAULT_SETTINGS["max_start_temp_c"] == 55
    assert rp.decide("txt2img", settings(), calm(gpu_temp_c=60.0), NOON).route == rp.WAIT
    assert rp.decide("txt2img", settings(), calm(gpu_temp_c=55.0), NOON).route == rp.WAIT
    assert rp.decide("txt2img", settings(), calm(gpu_temp_c=54.0), NOON).route == rp.LOCAL


def test_force_local_does_not_skip_the_cool_down_but_skip_wait_does():
    hot = calm(gpu_temp_c=82.0)
    assert rp.decide("txt2img", settings(), hot, NOON, force_local=True).route == rp.WAIT
    assert rp.decide("txt2img", settings(), hot, NOON, skip_wait=True).route == rp.LOCAL


def test_defer_wins_over_wait():
    """Outside the window AND hot: waiting for the card to cool would start the job at the wrong
    time of day, so the window decides."""
    d = rp.decide("animatediff", settings(video_windows=["23:00-08:00"]), calm(gpu_temp_c=83.0), NOON)
    assert d.route == rp.DEFER


def test_cloud_wins_over_defer():
    d = rp.decide("wan_i2v", settings(video_windows=["23:00-08:00"]), calm(), NOON)
    assert d.route == rp.CLOUD


def test_est_seconds_override_is_carried():
    assert rp.decide("gif", settings(), calm(), NOON, est_seconds=400).est_seconds == 400


def test_every_decision_has_a_summary_in_chinese():
    for kind in rp.JOB_COSTS:
        text = rp.decide(kind, settings(), calm(), NOON).summary()
        assert rp.JOB_COSTS[kind].label in text


def test_unmeasured_costs_say_so():
    assert "未實測" in rp.decide("svd", settings(), calm(), NOON).summary()
    assert "未實測" not in rp.decide("animatediff", settings(), calm(), NOON).summary()


# --- windows ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hhmm, inside",
    [
        ((22, 59), False),
        ((23, 0), True),
        ((2, 0), True),
        ((7, 59), True),
        ((8, 0), False),
        ((12, 0), False),
    ],
)
def test_a_window_wraps_past_midnight(hhmm, inside):
    now = datetime.datetime(2026, 9, 27, *hhmm)
    assert rp.in_windows(["23:00-08:00"], now) is inside


def test_a_same_day_window():
    assert rp.in_windows(["09:30-18:00"], datetime.datetime(2026, 9, 27, 9, 30))
    assert not rp.in_windows(["09:30-18:00"], datetime.datetime(2026, 9, 27, 18, 0))


def test_next_start_rolls_to_tomorrow_once_today_has_passed():
    assert rp.next_window_start(["06:00-07:00"], NOON) == datetime.datetime(2026, 9, 28, 6, 0)
    assert rp.next_window_start(["06:00-07:00", "13:00-14:00"], NOON) == datetime.datetime(2026, 9, 27, 13, 0)
    assert rp.next_window_start([], NOON) is None


@pytest.mark.parametrize("bad", ["23-08", "25:00-08:00", "10:00-10:00", "", "abc"])
def test_bad_windows_are_refused(bad):
    with pytest.raises(ValueError):
        rp.parse_window(bad)


def test_windows_accept_a_comma_separated_string():
    assert rp.parse_windows("23:00-08:00， 9:05-10:00") == ["23:00-08:00", "09:05-10:00"]


# --- settings -----------------------------------------------------------------------------------


def test_missing_settings_file_gives_the_defaults(tmp_path):
    assert rp.load_settings(str(tmp_path / "nope.json")) == rp.DEFAULT_SETTINGS


def test_broken_settings_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "hardware.json"
    path.write_text("{not json", encoding="utf-8")
    assert rp.load_settings(str(path)) == rp.DEFAULT_SETTINGS


def test_save_round_trips_and_validates(tmp_path):
    path = str(tmp_path / "s" / "hardware.json")
    saved = rp.save_settings({"max_start_temp_c": 75, "video_windows": "23:00-08:00"}, path)
    assert saved["video_windows"] == ["23:00-08:00"]
    assert rp.load_settings(path)["max_start_temp_c"] == 75
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["max_start_temp_c"] == 75
    with pytest.raises(ValueError):
        rp.save_settings({"max_start_temp_c": 90}, path)  # above the throttle point
    with pytest.raises(ValueError):
        rp.save_settings({"ram_margin_gb": "lots"}, path)


# --- probing ------------------------------------------------------------------------------------


def test_parse_nvidia_smi_with_throttle_flag():
    assert rp.parse_nvidia_smi("67, 6670, 8192, Not Active") == {
        "gpu_temp_c": 67.0,
        "vram_used_gb": 6.51,
        "vram_total_gb": 8.0,
        "gpu_throttling": False,
    }
    assert rp.parse_nvidia_smi("85, 1, 8192, Active")["gpu_throttling"] is True


def test_parse_nvidia_smi_tolerates_not_available_fields():
    parsed = rp.parse_nvidia_smi("[N/A], 512, 8192")
    assert parsed["gpu_temp_c"] is None
    assert parsed["gpu_throttling"] is None


def _completed(stdout, rc=0):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


class _Http:
    def __init__(self, up):
        self.up = up

    def get(self, url, timeout=None):
        if not self.up:
            raise rp.requests.exceptions.ConnectionError("refused")
        payload = (
            {"system": {"ram_free": 16 * 1024**3}}
            if url.endswith("/system_stats")
            else {"queue_running": [1], "queue_pending": [2, 3]}
        )
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


def test_probe_falls_back_to_the_three_basic_nvidia_smi_fields():
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd[1])
        return _completed("", rc=2) if "throttle" in cmd[1] else _completed("55, 1000, 8192\n")

    snap = rp.probe(run=run, http=_Http(up=False), psutil_module=None)
    assert len(calls) == 2
    assert snap.gpu_temp_c == 55.0
    assert snap.gpu_throttling is None
    assert snap.comfy_running is False


def test_probe_never_raises_when_nothing_answers():
    def run(cmd, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    snap = rp.probe(run=run, http=_Http(up=False), psutil_module=None)
    assert snap.gpu_temp_c is None
    assert any("GPU" in e for e in snap.errors)


def test_probe_reads_ram_and_queue_from_comfyui_without_psutil(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)  # force the ImportError path
    snap = rp.probe(run=lambda cmd, **k: _completed("50, 1000, 8192, Not Active"), http=_Http(up=True))
    assert snap.comfy_running is True
    assert snap.ram_available_gb == 16.0
    assert snap.comfy_queue == 3
    assert snap.local_busy_seconds == 2 * rp.COMFY_PENDING_PROMPT_SECONDS


# --- in-flight tracking and waiting -------------------------------------------------------------


def test_track_reports_remaining_seconds_and_clears():
    clock = [1000.0]
    with rp.track("animatediff", clock=lambda: clock[0]):
        assert rp.local_busy_seconds(clock=lambda: clock[0]) == 526
        clock[0] += 600  # overran its estimate: still counted, as 30 s
        assert rp.local_busy_seconds(clock=lambda: clock[0]) == 30
    assert rp.local_busy_seconds() == 0


def test_wait_until_ready_proceeds_when_the_card_cools():
    temps = iter([70.0, 52.0])
    slept = []
    first = rp.decide("txt2img", settings(), calm(gpu_temp_c=82.0), NOON)
    final = rp.wait_until_ready(
        "txt2img",
        first,
        settings=settings(),
        probe_fn=lambda: calm(gpu_temp_c=next(temps)),
        sleep=slept.append,
        clock=lambda: 0.0,
    )
    assert final.route == rp.LOCAL
    assert len(slept) == 2


def test_wait_until_ready_gives_up_waiting_after_the_timeout_and_runs():
    t = [0.0]

    def sleep(seconds):
        t[0] += seconds

    first = rp.decide("txt2img", settings(cooldown_timeout_s=30), calm(gpu_temp_c=83.0), NOON)
    final = rp.wait_until_ready(
        "txt2img",
        first,
        settings=settings(cooldown_timeout_s=30),
        probe_fn=lambda: calm(gpu_temp_c=83.0),
        sleep=sleep,
        clock=lambda: t[0],
    )
    assert final.route == rp.LOCAL
    assert any("照樣開始" in n for n in final.notes)


# --- kinds --------------------------------------------------------------------------------------


def test_image_kind_follows_the_variant_setting(isolated_variants, monkeypatch):
    assert rp.image_kind("pony", hq=False) == "txt2img"
    assert rp.image_kind("z_image_turbo", hq=True) == "zimage"
    assert rp.image_kind("pony", hq=True) == "txt2img_hq_full"
    monkeypatch.setenv("MODEL_VARIANT", "quant")
    assert rp.image_kind("pony", hq=True) == "txt2img_hq_quant"
    # a pose ControlNet (control-lora) forces the full file, so the cost is the full one
    assert rp.image_kind("pony", hq=True, uses_pose=True) == "txt2img_hq_full"


def test_animatediff_kind():
    assert rp.animatediff_kind(hires=False, use_facedetailer=False, upscale_to=0) == "animatediff_fast"
    assert rp.animatediff_kind() == "animatediff"
    assert rp.animatediff_kind(interp=4) == "animatediff_rife"


# --- finding ComfyUI's RSS ------------------------------------------------------------------------


class _FakePsutil:
    """The shape of this machine: a uv launcher shim naming ComfyUI (4 MB) and the real server,
    its child, whose command line does not (7.2 GB)."""

    CONN_LISTEN = "LISTEN"
    Error = OSError

    def __init__(self, sockets=True):
        self.sockets = sockets
        gb = 1024**3
        self.procs = {
            1: (
                ["D:/AI-Image-Lab/ComfyUI/.venv/Scripts/python.exe", "main.py", "--port", "8188"],
                "D:/AI-Image-Lab/ComfyUI",
                4 * 1024**2,
            ),
            2: (
                ["C:/uv/python/cpython-3.11/python.exe", "main.py", "--port", "8188"],
                "D:/AI-Image-Lab/ComfyUI",
                int(7.2 * gb),
            ),
            3: (["python.exe", "main.py"], "D:/other-project", 9 * gb),
        }

    def net_connections(self, kind):
        if not self.sockets:
            raise OSError("access denied")
        addr = types.SimpleNamespace(port=8188)
        return [types.SimpleNamespace(status="LISTEN", laddr=addr, pid=2)]

    def Process(self, pid):
        return types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=self.procs[pid][2]))

    def process_iter(self, attrs):
        for cmdline, cwd, rss in self.procs.values():
            info = {"cmdline": cmdline, "cwd": cwd, "memory_info": types.SimpleNamespace(rss=rss)}
            yield types.SimpleNamespace(info=info)


def test_comfy_rss_is_the_process_listening_on_the_port_not_the_launcher_shim():
    assert rp._comfy_process_rss_gb(_FakePsutil()) == 7.2


def test_comfy_rss_fallback_takes_the_largest_comfyui_main_py():
    # the other project's 9 GB main.py is not ComfyUI; the shim is, but is not the server
    assert rp._comfy_process_rss_gb(_FakePsutil(sockets=False)) == 7.2
