"""Decides, per generation, whether this machine should run it now, wait, run it later, or hand it
to the cloud - from what the job costs and what the hardware is doing at the moment it is asked.

Why this exists: the RTX 2070 here has 8 GB of VRAM, sits on a PCIe gen3 x1 link, throttles at
84C, and shares 32 GB of RAM with a ComfyUI that peaks near 16 GB on the full-precision HQ path.
Until now every one of those limits was something the operator had to remember. The rules below
write them down once, so the GUI and image_api can say "this will page to disk - use the fp8
variant" or "this clip needs more VRAM than the card has - RunPod?" before a generation is spent.

Three inputs, as asked for:
  * static cost per job kind (JOB_COSTS - the measured numbers from README/CHANGELOG),
  * live state (probe(): GPU temperature and throttle flag, VRAM, available RAM, ComfyUI's own
    RSS and queue, this process's in-flight jobs),
  * time windows (settings: e.g. video only 23:00-08:00).

The cloud is only ever *suggested*. decide() never sends anything anywhere; a caller shows the
reasons and the operator confirms (GUI checkbox, image_api's force_local). Cloud GPUs cost money
and this repo's cloud path has never run against a real account.

Deliberately requests + stdlib only, like comfyui_client: psutil is imported lazily (it is in
ComfyUI\\.venv, where the GUI and image_api run, but not in .venv-dev or CI), and nvidia-smi is a
subprocess, so nothing here needs torch or a GPU to import or to test.
"""

import contextlib
import datetime
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field

import requests

import comfyui_client as client

SETTINGS_FILE = os.environ.get(
    "HARDWARE_SETTINGS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings", "hardware.json"),
)

# Defaults describe the machine this repo was built on (CLAUDE.md "硬體與環境"). Anything here can
# be overridden in settings/hardware.json, which is also what the GUI's 資源排程 tab writes.
DEFAULT_SETTINGS = {
    "version": 1,
    "gpu_vram_gb": 8.0,
    "system_ram_gb": 32.0,
    "throttle_temp_c": 84,       # the card's own hardware slowdown point
    "max_start_temp_c": 78,      # don't START a job above this; wait for it to cool first
    "cooldown_timeout_s": 600,   # ...but never wait longer than this - then run anyway
    "ram_margin_gb": 3.0,        # left for Windows + the browser + the GUI itself
    "max_local_wait_s": 1800,    # local backlog above this -> suggest the cloud (when a route exists)
    "video_windows": [],         # e.g. ["23:00-08:00"]; empty = any time
    "image_windows": [],
}
_NUMERIC_BOUNDS = {
    "gpu_vram_gb": (1.0, 128.0),
    "system_ram_gb": (4.0, 1024.0),
    "throttle_temp_c": (50, 110),
    "max_start_temp_c": (40, 105),
    "cooldown_timeout_s": (0, 7200),
    "ram_margin_gb": (0.0, 64.0),
    "max_local_wait_s": (0, 86400),
}

LOCAL = "local"
WAIT = "wait"
DEFER = "defer"
CLOUD = "cloud_suggested"
ROUTES = (LOCAL, WAIT, DEFER, CLOUD)

CLOUD_WORKFLOW = "workflow"                              # comfyui_client.backend_scope("runpod")
CLOUD_ANIMATEDIFF = "cloud_video:video_animatediff"      # cloud_video.run_cloud_video
CLOUD_WAN = "cloud_video:video_wan_i2v"

COMFY_PENDING_PROMPT_SECONDS = 60   # rough cost of one prompt waiting in ComfyUI's own queue
WAIT_POLL_SECONDS = 15
NVIDIA_SMI_TIMEOUT_SECONDS = 5
COMFY_PROBE_TIMEOUT_SECONDS = 2


@dataclass(frozen=True)
class JobCost:
    label: str               # user-facing name
    media: str               # "image" | "video" - selects which time window applies
    seconds: float           # typical wall time on this machine, warm
    vram_gb: float           # peak VRAM
    ram_gb: float            # peak ComfyUI RSS (what competes with the rest of the system)
    local_ok: bool = True
    cloud_target: str = None
    uses_comfyui: bool = True
    measured: bool = True    # False -> the numbers are an estimate and the UI says so


# Sources: README "3c" table (HQ full vs fp8), README AnimateDiff timing table, CHANGELOG
# 2026-09 A/B (txt2img 29-40 s, VRAM ~4.5 GB, RSS ~8.4 GB), comfyui_client's Z-Image note
# (~75 s/image warm, 11 GB of weights that don't fit together), CHANGELOG on Wan (18 GB weights,
# no local entry). SVD and SadTalker were never timed with these instruments: measured=False.
JOB_COSTS = {
    "txt2img": JobCost("文字生圖", "image", 40, 4.5, 8.4, cloud_target=CLOUD_WORKFLOW),
    "txt2img_hq_full": JobCost("高清生圖（完整版權重）", "image", 272, 6.5, 15.8, cloud_target=CLOUD_WORKFLOW),
    "txt2img_hq_quant": JobCost("高清生圖（fp8 量化版）", "image", 130, 5.5, 13.2, cloud_target=CLOUD_WORKFLOW),
    "zimage": JobCost("Z-Image 生圖", "image", 120, 7.0, 13.0, cloud_target=CLOUD_WORKFLOW, measured=False),
    "gif": JobCost("批次 GIF", "image", 40, 4.5, 8.4, cloud_target=CLOUD_WORKFLOW),  # per frame; caller scales
    "animatediff_fast": JobCost("AnimateDiff 快速（不高清）", "video", 69, 5.0, 8.0, cloud_target=CLOUD_ANIMATEDIFF),
    "animatediff": JobCost("AnimateDiff 預設高清", "video", 526, 6.5, 9.0, cloud_target=CLOUD_ANIMATEDIFF),
    "animatediff_rife": JobCost("AnimateDiff + RIFE 補幀", "video", 683, 6.8, 9.5, cloud_target=CLOUD_ANIMATEDIFF),
    "svd": JobCost("SVD 圖生影片", "video", 180, 7.5, 9.0, measured=False),
    "sadtalker": JobCost("SadTalker 對嘴影片", "video", 180, 4.0, 6.0, uses_comfyui=False, measured=False),
    "wan_i2v": JobCost("Wan 2.2 圖生影片", "video", 0, 18.0, 24.0, local_ok=False, cloud_target=CLOUD_WAN,
                       measured=False),
}


# --- settings -----------------------------------------------------------------------------------

_WINDOW_RE = re.compile(r"^\s*([01]?\d|2[0-3]):([0-5]\d)\s*-\s*([01]?\d|2[0-3]):([0-5]\d)\s*$")


def parse_window(text):
    """"23:00-08:00" -> ((23, 0), (8, 0)). Raises ValueError with a zh-TW message."""
    m = _WINDOW_RE.match(text or "")
    if not m:
        raise ValueError(f"時段格式要像 23:00-08:00，收到「{text}」")
    start, end = (int(m[1]), int(m[2])), (int(m[3]), int(m[4]))
    if start == end:
        raise ValueError(f"時段「{text}」開始跟結束一樣；不想限制時段就留空")
    return start, end


def parse_windows(value):
    """A list, or one comma/newline-separated string, of windows -> normalised "HH:MM-HH:MM" list."""
    if isinstance(value, str):
        value = [part for part in re.split(r"[,\n，]", value) if part.strip()]
    out = []
    for item in value or []:
        (sh, sm), (eh, em) = parse_window(item)
        out.append(f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}")
    return out


def _clean_settings(data):
    settings = dict(DEFAULT_SETTINGS)
    for name, (lo, hi) in _NUMERIC_BOUNDS.items():
        value = data.get(name, settings[name])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} 要是數字，收到 {value!r}")
        if not lo <= value <= hi:
            raise ValueError(f"{name} 要在 {lo} 到 {hi} 之間，收到 {value}")
        settings[name] = value
    if settings["max_start_temp_c"] > settings["throttle_temp_c"]:
        raise ValueError("開始門檻溫度不能高於降頻溫度")
    for name in ("video_windows", "image_windows"):
        settings[name] = parse_windows(data.get(name, settings[name]))
    return settings


def load_settings(path=None):
    """Settings file merged over DEFAULT_SETTINGS. A missing file is the normal case; an unreadable
    one falls back to the defaults with a notice rather than blocking every generation."""
    path = path or SETTINGS_FILE
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("最外層要是 JSON 物件")
        return _clean_settings(data)
    except FileNotFoundError:
        return dict(DEFAULT_SETTINGS)
    except (OSError, ValueError) as exc:
        print(f"[resource_policy] {path} 讀不懂（{exc}），改用預設值", flush=True)
        return dict(DEFAULT_SETTINGS)


def save_settings(updates, path=None):
    """Validate the merged result, then write atomically. Raises ValueError (zh-TW) on bad input."""
    path = path or SETTINGS_FILE
    merged = _clean_settings({**load_settings(path), **updates})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return merged


# --- time windows -------------------------------------------------------------------------------

def _minutes(hm):
    return hm[0] * 60 + hm[1]


def in_windows(windows, now):
    """True when `now` (naive local datetime) is inside any window. Empty list = no restriction.
    A window whose end is before its start wraps past midnight."""
    if not windows:
        return True
    t = now.hour * 60 + now.minute
    for text in windows:
        start, end = (_minutes(hm) for hm in parse_window(text))
        if (start <= t < end) if start < end else (t >= start or t < end):
            return True
    return False


def next_window_start(windows, now):
    """The earliest window start strictly after `now` (naive local datetime), or None."""
    if not windows:
        return None
    base = now.replace(second=0, microsecond=0)
    candidates = []
    for text in windows:
        (sh, sm), _ = parse_window(text)
        start = base.replace(hour=sh, minute=sm)
        if start <= now:
            start += datetime.timedelta(days=1)
        candidates.append(start)
    return min(candidates)


# --- live state ---------------------------------------------------------------------------------

@dataclass
class Snapshot:
    """What the machine looked like at one instant. None always means "could not read", never 0."""
    taken_at: float = field(default_factory=time.time)
    gpu_temp_c: float = None
    gpu_throttling: bool = None
    vram_used_gb: float = None
    vram_total_gb: float = None
    ram_available_gb: float = None
    comfy_running: bool = None
    comfy_rss_gb: float = None
    comfy_queue: int = None
    local_busy_seconds: float = 0.0
    errors: list = field(default_factory=list)


def parse_nvidia_smi(line):
    """One CSV line of temperature.gpu, memory.used, memory.total[, hw_thermal_slowdown] (MiB)."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 3:
        raise ValueError(f"nvidia-smi 輸出看不懂：{line!r}")

    def num(text):
        try:
            return float(text)
        except ValueError:
            return None   # "[N/A]", "[Not Supported]"

    temp, used, total = num(parts[0]), num(parts[1]), num(parts[2])
    throttling = None
    if len(parts) >= 4 and parts[3] in ("Active", "Not Active"):
        throttling = parts[3] == "Active"
    return {
        "gpu_temp_c": temp,
        "vram_used_gb": None if used is None else round(used / 1024, 2),
        "vram_total_gb": None if total is None else round(total / 1024, 2),
        "gpu_throttling": throttling,
    }


def _query_nvidia_smi(run=subprocess.run):
    base = "temperature.gpu,memory.used,memory.total"
    # Newer drivers renamed clocks_throttle_reasons -> clocks_event_reasons; ask with the flag
    # first and fall back to the three fields every driver has.
    for fields in (base + ",clocks_throttle_reasons.hw_thermal_slowdown", base):
        result = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_SECONDS)
        if result.returncode == 0 and result.stdout.strip():
            return parse_nvidia_smi(result.stdout.strip().splitlines()[0])
    raise RuntimeError(f"nvidia-smi 回傳 {result.returncode}")


def _comfy_process_rss_gb(psutil):
    """RSS of the local ComfyUI (a python running ComfyUI's main.py), or None if not found."""
    for proc in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            if "main.py" in cmd and "comfyui" in cmd and proc.info.get("memory_info"):
                return round(proc.info["memory_info"].rss / 1024 ** 3, 2)
        except (psutil.Error, OSError):
            continue
    return None


def probe(*, run=subprocess.run, http=None, psutil_module=None):
    """Read everything decide() looks at. Never raises: a failed reading is None plus an entry in
    `errors`, and decide() treats unknown as "no evidence against running", not as a block."""
    http = http or requests
    snap = Snapshot(local_busy_seconds=local_busy_seconds())
    try:
        for name, value in _query_nvidia_smi(run).items():
            setattr(snap, name, value)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        snap.errors.append(f"讀不到 GPU 狀態（{exc}）")

    stats = None
    try:
        r = http.get(f"{client.COMFYUI_URL}/system_stats", timeout=COMFY_PROBE_TIMEOUT_SECONDS)
        r.raise_for_status()
        stats = r.json()
        snap.comfy_running = True
    except (requests.exceptions.RequestException, ValueError):
        snap.comfy_running = False
    if snap.comfy_running:
        try:
            q = http.get(f"{client.COMFYUI_URL}/queue", timeout=COMFY_PROBE_TIMEOUT_SECONDS).json()
            snap.comfy_queue = len(q.get("queue_running") or []) + len(q.get("queue_pending") or [])
            pending = len(q.get("queue_pending") or [])
            snap.local_busy_seconds += pending * COMFY_PENDING_PROMPT_SECONDS
        except (requests.exceptions.RequestException, ValueError, AttributeError):
            snap.errors.append("讀不到 ComfyUI 佇列")

    psutil = psutil_module
    if psutil is None:
        try:
            import psutil  # noqa: PLC0415 - optional: present in ComfyUI\.venv, absent in .venv-dev
        except ImportError:
            psutil = None
    if psutil is not None:
        try:
            snap.ram_available_gb = round(psutil.virtual_memory().available / 1024 ** 3, 2)
            if snap.comfy_running:
                snap.comfy_rss_gb = _comfy_process_rss_gb(psutil)
        except (OSError, AttributeError) as exc:
            snap.errors.append(f"讀不到系統記憶體（{exc}）")
    if snap.ram_available_gb is None and stats:
        try:
            snap.ram_available_gb = round(stats["system"]["ram_free"] / 1024 ** 3, 2)
        except (KeyError, TypeError):
            pass
    if snap.ram_available_gb is None:
        snap.errors.append("讀不到可用 RAM")
    return snap


# --- in-flight jobs in this process -------------------------------------------------------------

_INFLIGHT = {}
_INFLIGHT_LOCK = threading.Lock()


@contextlib.contextmanager
def track(kind, est_seconds=None, *, clock=time.time):
    """Mark a local job as running in this process so probe() can report the backlog."""
    token = object()
    seconds = est_seconds if est_seconds is not None else JOB_COSTS[kind].seconds
    with _INFLIGHT_LOCK:
        _INFLIGHT[token] = (kind, clock(), seconds)
    try:
        yield
    finally:
        with _INFLIGHT_LOCK:
            _INFLIGHT.pop(token, None)


def local_busy_seconds(clock=time.time):
    """Estimated seconds until this process's local jobs finish. An overrunning job still counts
    30 s: it is evidently still busy, we just no longer know for how long."""
    now = clock()
    with _INFLIGHT_LOCK:
        entries = list(_INFLIGHT.values())
    return float(sum(max(30.0, started + seconds - now) for _, started, seconds in entries))


def local_jobs_running():
    with _INFLIGHT_LOCK:
        return len(_INFLIGHT)


# --- job kinds ----------------------------------------------------------------------------------

def image_kind(checkpoint=None, hq=True, *, uses_pose=False, default_checkpoint="cyberrealistic_pony"):
    """Kind for a still-image request. HQ picks the fp8 row when that model is set to quant - via
    comfyui_client.chosen_variant, the same resolution apply_model_variants uses - unless a pose
    ControlNet (control-lora) forces the full file."""
    if checkpoint in client.ZIMAGE_MODELS:
        return "zimage"
    if not hq:
        return "txt2img"
    key = checkpoint or default_checkpoint
    quant = client.chosen_variant(key) == "quant" and not (uses_pose and client._control_lora_forces_full())
    return "txt2img_hq_quant" if quant else "txt2img_hq_full"


def animatediff_kind(*, hires=True, use_facedetailer=True, upscale_to=None, interp=1, lcm=False):
    if interp and int(interp) > 1:
        return "animatediff_rife"
    if not hires and not use_facedetailer and not upscale_to:
        return "animatediff_fast"
    return "animatediff"


# --- the decision -------------------------------------------------------------------------------

@dataclass
class Decision:
    route: str
    kind: str
    reasons: list
    est_seconds: float = None
    not_before: float = None       # epoch seconds, for DEFER
    cloud_target: str = None
    suggest_quant: bool = False
    notes: list = field(default_factory=list)   # informational, never the cause of the route

    def to_dict(self):
        return asdict(self)

    @property
    def runs_now(self):
        return self.route == LOCAL

    def summary(self):
        cost = JOB_COSTS[self.kind]
        head = {
            LOCAL: "✅ 本地執行",
            WAIT: "⏳ 先等一下再本地執行",
            DEFER: "🕒 排到下個時段",
            CLOUD: "☁️ 建議改送雲端",
        }[self.route]
        eta = ""
        if self.est_seconds:
            eta = f"，預估約 {int(self.est_seconds)} 秒" + ("" if cost.measured else "（未實測的估計）")
        lines = [f"**{head}**｜{cost.label}{eta}"]
        if self.route == DEFER and self.not_before:
            lines.append(f"- 會在 {datetime.datetime.fromtimestamp(self.not_before):%m/%d %H:%M} 開始")
        lines += [f"- {r}" for r in self.reasons]
        lines += [f"- ℹ️ {n}" for n in self.notes]
        return "\n".join(lines)


def _state_notes(snap):
    parts = []
    if snap.gpu_temp_c is not None:
        parts.append(f"GPU {snap.gpu_temp_c:.0f}°C" + ("（降頻中）" if snap.gpu_throttling else ""))
    if snap.vram_used_gb is not None and snap.vram_total_gb:
        parts.append(f"VRAM {snap.vram_used_gb:.1f}/{snap.vram_total_gb:.1f} GB")
    if snap.ram_available_gb is not None:
        parts.append(f"RAM 可用 {snap.ram_available_gb:.1f} GB")
    if snap.comfy_rss_gb is not None:
        parts.append(f"ComfyUI 佔 {snap.comfy_rss_gb:.1f} GB")
    if snap.local_busy_seconds:
        parts.append(f"本地還要約 {int(snap.local_busy_seconds)} 秒")
    notes = ["目前：" + "、".join(parts)] if parts else []
    return notes + list(snap.errors)


def decide(kind, settings, snap, now=None, *, est_seconds=None, force_local=False, skip_wait=False):
    """Pure: the same inputs always give the same Decision. `now` is a naive local datetime.

    force_local is the operator overriding the *advice* (cloud suggestion, time window, RAM/backlog
    warnings). It does not skip the cool-down wait, which is protection with a timeout, not advice;
    skip_wait does that (the scheduler sets it once a job has waited cooldown_timeout_s already).

    Rule order is the one in the plan and the tests pin it:
      1. the card cannot run it at all          -> cloud
      2. not enough RAM (HQ: fp8 first)          -> cloud, or local with a warning if no cloud route
      3. more VRAM than the card has             -> cloud
      4. another program is holding the VRAM     -> wait
      5. local backlog longer than allowed       -> cloud (if a route exists)
      6. outside the time window                 -> defer
      7. too hot / throttling                    -> wait
      8. otherwise                               -> local
    """
    cost = JOB_COSTS[kind]
    now = now or datetime.datetime.now()
    seconds = est_seconds if est_seconds is not None else cost.seconds
    notes = _state_notes(snap)

    def make(route, *reasons, not_before=None, suggest_quant=False):
        return Decision(route, kind, list(reasons), est_seconds=seconds or None, not_before=not_before,
                        cloud_target=cost.cloud_target if route == CLOUD else None,
                        suggest_quant=suggest_quant, notes=notes)

    if not cost.local_ok:
        return make(CLOUD, f"{cost.label} 需要約 {cost.vram_gb:.0f} GB VRAM，"
                           f"這張 {settings['gpu_vram_gb']:.0f} GB 的卡本地跑不了")

    warnings, suggest_quant = [], False
    if not force_local:
        if snap.ram_available_gb is not None:
            headroom = snap.ram_available_gb + (snap.comfy_rss_gb or 0.0) - settings["ram_margin_gb"]
            if cost.ram_gb > headroom:
                shortage = (f"可用 RAM 約 {max(headroom, 0):.1f} GB（已扣保留 {settings['ram_margin_gb']:.0f} GB），"
                            f"{cost.label} 峰值約 {cost.ram_gb:.1f} GB，會用到分頁檔、明顯變慢")
                quant = JOB_COSTS["txt2img_hq_quant"]
                if kind == "txt2img_hq_full" and quant.ram_gb <= headroom:
                    suggest_quant = True
                    warnings.append(shortage + "；到「模型版本」把這個 checkpoint 切成量化版就夠"
                                    f"（實測 {quant.seconds:.0f} 秒 vs 完整版分頁時 530 秒）")
                elif cost.cloud_target:
                    return make(CLOUD, shortage)
                else:
                    warnings.append(shortage)

        if cost.vram_gb > settings["gpu_vram_gb"]:
            if cost.cloud_target:
                return make(CLOUD, f"{cost.label} 需要約 {cost.vram_gb:.1f} GB VRAM，超過這張卡的 "
                                   f"{settings['gpu_vram_gb']:.0f} GB")
            warnings.append(f"{cost.label} 需要約 {cost.vram_gb:.1f} GB VRAM，可能會 OOM")

    # ComfyUI not running but VRAM in use = some other program (a kohya run, a local LLM) has the
    # card. With ComfyUI up the reading includes ComfyUI's own model cache, so it means nothing.
    if (not skip_wait and snap.comfy_running is False and snap.vram_used_gb is not None
            and snap.vram_total_gb and cost.vram_gb > snap.vram_total_gb - snap.vram_used_gb):
        return make(WAIT, *warnings, f"其他程式正占用 {snap.vram_used_gb:.1f} GB VRAM（ComfyUI 沒在跑，"
                                     "可能是訓練或其他 AI 程式），等它釋放再開始", suggest_quant=suggest_quant)

    if not force_local and cost.cloud_target and snap.local_busy_seconds > settings["max_local_wait_s"]:
        return make(CLOUD, *warnings, f"本地前面還有約 {int(snap.local_busy_seconds)} 秒的工作，"
                                      f"超過設定的 {int(settings['max_local_wait_s'])} 秒")

    windows = settings["video_windows"] if cost.media == "video" else settings["image_windows"]
    if not force_local and not in_windows(windows, now):
        start = next_window_start(windows, now)
        what = "影片" if cost.media == "video" else "圖片"
        return make(DEFER, *warnings, f"現在不在{what}時段（{'、'.join(windows)}）",
                    not_before=start.timestamp() if start else None, suggest_quant=suggest_quant)

    if not skip_wait:
        hot = snap.gpu_temp_c is not None and snap.gpu_temp_c >= settings["max_start_temp_c"]
        if hot or snap.gpu_throttling:
            why = (f"GPU {snap.gpu_temp_c:.0f}°C" if snap.gpu_temp_c is not None else "GPU")
            why += "，正在降頻" if snap.gpu_throttling else f"，高於開始門檻 {settings['max_start_temp_c']}°C"
            return make(WAIT, *warnings, why + f"；降溫後自動開始（最多等 {int(settings['cooldown_timeout_s'])} 秒）",
                        suggest_quant=suggest_quant)

    return make(LOCAL, *warnings, suggest_quant=suggest_quant)


def wait_until_ready(kind, decision, *, settings=None, probe_fn=probe, sleep=time.sleep, clock=time.monotonic,
                     cancel_event=None, on_status=None, est_seconds=None, force_local=False):
    """Block while `decision` is WAIT, re-probing every WAIT_POLL_SECONDS. Returns the decision it
    finally proceeds on: LOCAL once the card is ready, or the last WAIT with skip_wait applied once
    cooldown_timeout_s runs out (a hot card is slower, not broken - never block forever). Any other
    route coming back (say the time window closed meanwhile) is returned as-is for the caller."""
    settings = settings or load_settings()
    deadline = clock() + settings["cooldown_timeout_s"]
    while decision.route == WAIT:
        remaining = deadline - clock()
        if remaining <= 0:
            final = decide(kind, settings, probe_fn(), est_seconds=est_seconds, force_local=force_local,
                           skip_wait=True)
            final.notes.append(f"已等 {int(settings['cooldown_timeout_s'])} 秒仍未就緒，照樣開始")
            return final
        if on_status:
            on_status(decision, int(remaining))
        if cancel_event is not None and cancel_event.is_set():
            return decision
        sleep(min(WAIT_POLL_SECONDS, max(remaining, 0.1)))
        decision = decide(kind, settings, probe_fn(), est_seconds=est_seconds, force_local=force_local)
    return decision


def status_markdown(settings=None, snap=None):
    """The 資源排程 tab's live panel."""
    settings = settings or load_settings()
    snap = snap or probe()
    rows = [
        ("GPU 溫度", "—" if snap.gpu_temp_c is None else
         f"{snap.gpu_temp_c:.0f}°C（開始門檻 {settings['max_start_temp_c']}°C、降頻 {settings['throttle_temp_c']}°C）"
         + ("⚠️ 降頻中" if snap.gpu_throttling else "")),
        ("VRAM", "—" if snap.vram_used_gb is None else f"{snap.vram_used_gb:.1f} / {snap.vram_total_gb:.1f} GB"),
        ("可用 RAM", "—" if snap.ram_available_gb is None else f"{snap.ram_available_gb:.1f} GB"
         f"（保留 {settings['ram_margin_gb']:.0f} GB 給系統）"),
        ("ComfyUI", "沒在跑" if snap.comfy_running is False else
         "執行中" + (f"，佔 {snap.comfy_rss_gb:.1f} GB RAM" if snap.comfy_rss_gb is not None else "")
         + (f"，佇列 {snap.comfy_queue}" if snap.comfy_queue else "")),
        ("本地待完成", f"約 {int(snap.local_busy_seconds)} 秒" if snap.local_busy_seconds else "無"),
        ("影片時段", "、".join(settings["video_windows"]) or "不限"),
        ("圖片時段", "、".join(settings["image_windows"]) or "不限"),
    ]
    lines = ["| 項目 | 狀態 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    lines += [f"\n⚠️ {e}" for e in snap.errors]
    return "\n".join(lines)
