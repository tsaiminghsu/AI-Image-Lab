"""Runs ON the Colab VM, sent by the local runner with `colab exec -f`. Keep this file ASCII-only.

It installs ComfyUI, downloads MiniMax H3 (first-frame / I2VA mode), renders one clip and copies the
MP4 to the path the local runner downloads. Parameters come from the JSON file named by H3_JOB_FILE
(uploaded next to the first frame), so no value ever has to survive Windows -> Docker -> CLI quoting.

Progress and results are single-line markers, `H3_<KIND> {json}`, parsed by the runner while it streams
the output. `colab exec` exits 0 even when this code raises, so every failure path prints `H3_ERROR`
first, and a run that never prints `H3_OUTPUT` counts as failed.

The graph follows the official ComfyUI MiniMax H3 nodes (comfy_extras/nodes_minimax_h3.py) and the
model pairing of the killkli/minimax-h3-colab-skill notebook's first-frame mode: pruned FL2VA base
(fp8_scaled, or int8_convrot on CUDA 13) + the pruned turbo v4 LoRA at 8 steps. A pruned LoRA needs
the pruned base. No `from __future__` import here: `colab exec --env` prepends code to this file.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

JOB_FILE_ENV = "H3_JOB_FILE"
COMFY = Path("/content/ComfyUI")
VHS = COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"
SETUP_MARKER = Path("/content/.h3_comfyui_setup_v1")
COMFY_LOG = Path("/content/comfyui_h3.log")
COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
VHS_REPO = "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
PORT = 8188
BASE_URL = "http://127.0.0.1:%d" % PORT

FPS = 24
BASE_REPO = "Comfy-Org/MiniMax-H3"
LORA_REPO = "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI"
DIFFUSION_FP8 = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
# Official guidance: int8 ConvRot only with a CUDA 13.x PyTorch build.
DIFFUSION_INT8 = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
MIN_FREE_DISK_GIB = 60
# ComfyUI --lowvram below this: the 33B base does not fit an A100-40GB without offloading.
LOWVRAM_BELOW_GIB = 48
OUTPUT_NODE = "15"
REQUIRED_NODES = (
    "UNETLoader",
    "LoraLoaderModelOnly",
    "CLIPLoader",
    "VAELoader",
    "LoadImage",
    "ImageScaleToTotalPixels",
    "MiniMaxH3ImageToVideo",
    "BasicGuider",
    "RandomNoise",
    "KSamplerSelect",
    "BasicScheduler",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEDecodeAudio",
    "VHS_VideoCombine",
)


class StageError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def marker(kind, **fields):
    print("H3_%s %s" % (kind, json.dumps(fields, ensure_ascii=True, sort_keys=True)), flush=True)


def frames_for(duration):
    """Requested seconds -> frame count on H3's 17k+5 grid at 24 fps (same rule as ComfyUI's node).
    Must match h3_colab.frames_for; tests/test_minimax_h3_remote.py asserts it."""
    requested = max(5, round(duration * FPS))
    return requested + (5 - requested % 17) % 17


def choose_diffusion(cuda_version):
    return DIFFUSION_INT8 if (cuda_version or "").startswith("13.") else DIFFUSION_FP8


def model_files(diffusion):
    """(repo, file in repo, ComfyUI models/ subfolder)."""
    return [
        (BASE_REPO, "diffusion_models/" + diffusion, "diffusion_models"),
        (BASE_REPO, "text_encoders/" + TEXT_ENCODER, "text_encoders"),
        (BASE_REPO, "vae/" + VIDEO_VAE, "vae"),
        (BASE_REPO, "vae/" + AUDIO_VAE, "vae"),
        (LORA_REPO, LORA, "loras"),
    ]


def build_graph(job, diffusion):
    """ComfyUI API graph for one first-frame (I2VA) clip with audio. Pure; tested offline."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": diffusion, "weight_dtype": "default"}},
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": job["lora_strength"]},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "6": {"class_type": "LoadImage", "inputs": {"image": job["comfy_image_name"]}},
        "16": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {"image": ["6", 0], "upscale_method": "nearest-exact", "megapixels": 1.0, "resolution_steps": 32},
        },
        "7": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "first_frame": ["16", 0],
                "prompt": job["prompt"],
                "width": job["width"],
                "height": job["height"],
                "length": job["frames"],
            },
        },
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["7", 0]}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": job["seed"]}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": job["sampler"]}},
        "11": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["2", 0], "scheduler": job["scheduler"], "steps": job["steps"], "denoise": 1.0},
        },
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["8", 0],
                "sampler": ["10", 0],
                "sigmas": ["11", 0],
                "latent_image": ["7", 1],
            },
        },
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        OUTPUT_NODE: {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["13", 0],
                "audio": ["14", 0],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": "h3_" + job["job_id"],
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": job["crf"],
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def unexpected_inputs(graph, object_info):
    """Graph inputs ComfyUI no longer declares -> {node_id: [names]}. Catches a renamed node input
    after a ComfyUI update before the job is queued, instead of a confusing validation error."""
    problems = {}
    for node_id, node in graph.items():
        spec = object_info[node["class_type"]]["input"]
        known = set(spec.get("required", {})) | set(spec.get("optional", {}))
        if node["class_type"] == "VHS_VideoCombine":
            formats = spec["required"]["format"][1].get("formats", {})
            known.update(widget[0] for widget in formats.get(node["inputs"]["format"], []))
        extra = sorted(set(node["inputs"]) - known)
        if extra:
            problems[node_id] = extra
    return problems


def run(cmd, **kwargs):
    subprocess.run(cmd, check=True, **kwargs)


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise StageError("GPU_UNAVAILABLE", "No CUDA GPU in this Colab runtime.")
    props = torch.cuda.get_device_properties(0)
    info = {
        "name": torch.cuda.get_device_name(0),
        "vram_gib": round(props.total_memory / 2**30, 1),
        "capability": "%d.%d" % torch.cuda.get_device_capability(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "free_disk_gib": round(shutil.disk_usage("/content").free / 2**30, 1),
    }
    marker("GPU", **info)
    if info["free_disk_gib"] < MIN_FREE_DISK_GIB:
        raise StageError(
            "INSUFFICIENT_DISK",
            "%.1f GiB free, need %d GiB for the models" % (info["free_disk_gib"], MIN_FREE_DISK_GIB),
        )
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise StageError("COMFYUI_SETUP_FAILED", "ffmpeg/ffprobe missing on the runtime")
    return info


def setup_comfyui():
    if SETUP_MARKER.exists() and COMFY.is_dir() and VHS.is_dir():
        print("reusing the ComfyUI install from this session", flush=True)
    else:
        if not COMFY.exists():
            run(["git", "clone", "--depth", "1", COMFY_REPO, str(COMFY)])
        run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(COMFY / "requirements.txt")]
            + ["huggingface_hub", "hf_xet"]
        )
        if not VHS.exists():
            run(["git", "clone", "--depth", "1", VHS_REPO, str(VHS)])
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(VHS / "requirements.txt")])
        SETUP_MARKER.touch()
    commit = subprocess.run(
        ["git", "-C", str(COMFY), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    marker("COMFYUI", commit=commit)


def download_models(diffusion):
    from huggingface_hub import hf_hub_download

    sizes = {}
    for repo, remote_name, folder in model_files(diffusion):
        target = COMFY / "models" / folder / Path(remote_name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.stat().st_size == 0:
            local_dir = COMFY / "models" if repo == BASE_REPO else COMFY / "models" / "loras"
            # token=False: the repos are public, and without it huggingface_hub asks the Colab secret
            # store for HF_TOKEN, which only answers inside the Colab UI (a 10 s timeout, measured).
            got = Path(hf_hub_download(repo_id=repo, filename=remote_name, local_dir=str(local_dir), token=False))
            if got.resolve() != target.resolve():
                raise StageError("MODEL_DOWNLOAD_FAILED", "downloaded to %s, expected %s" % (got, target))
        sizes[target.name] = round(target.stat().st_size / 2**30, 2)
        print("model %s %.2f GiB" % (target.name, sizes[target.name]), flush=True)
    return sizes


def http_json(path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def start_comfyui(vram_gib):
    try:
        http_json("/system_stats", timeout=3)
        print("using the ComfyUI already running in this session", flush=True)
        return None
    except (urllib.error.URLError, OSError, ValueError):
        pass
    launch = [sys.executable, str(COMFY / "main.py"), "--listen", "127.0.0.1", "--port", str(PORT)]
    launch.append("--disable-auto-launch")
    if vram_gib < LOWVRAM_BELOW_GIB:
        launch.append("--lowvram")
    print("starting ComfyUI: %s" % " ".join(launch[2:]), flush=True)
    log = open(COMFY_LOG, "a")
    proc = subprocess.Popen(launch, cwd=str(COMFY), stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        if proc.poll() is not None:
            raise StageError("COMFYUI_SETUP_FAILED", "ComfyUI exited on start:\n" + tail_log())
        try:
            http_json("/system_stats", timeout=3)
            return proc
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(2)
    raise StageError("COMFYUI_SETUP_FAILED", "ComfyUI did not answer within 10 minutes:\n" + tail_log())


def tail_log(chars=4000):
    try:
        return COMFY_LOG.read_text(errors="replace")[-chars:]
    except OSError:
        return ""


class VramSampler(threading.Thread):
    """Peak GPU memory in MiB via nvidia-smi: ComfyUI is a separate process, so torch's own
    max_memory_allocated in this kernel would read 0."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = None
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                used = int(out.split()[0])
                self.peak = used if self.peak is None else max(self.peak, used)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self.stopped.wait(2)


def execution_error(status):
    for kind, data in status.get("messages", []):
        if kind == "execution_error":
            return "%s: %s" % (data.get("exception_type"), data.get("exception_message"))
    return json.dumps(status)[:2000]


def wait_for_video(prompt_id, deadline, proc):
    last_beat = 0.0
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise StageError("INFERENCE_FAILED", "ComfyUI exited during inference:\n" + tail_log())
        history = http_json("/history/" + prompt_id)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise StageError("INFERENCE_FAILED", execution_error(status))
            outputs = entry.get("outputs", {})
            if OUTPUT_NODE in outputs:
                files = outputs[OUTPUT_NODE].get("gifs", [])
                video = next((f for f in files if f.get("filename", "").endswith(".mp4")), None)
                if video is None:
                    raise StageError("OUTPUT_NOT_FOUND", "VideoHelperSuite returned no MP4: %s" % files)
                return COMFY / "output" / video.get("subfolder", "") / video["filename"]
            if status.get("completed") or status.get("status_str") == "success":
                raise StageError("OUTPUT_NOT_FOUND", "the job finished without a video output")
        if time.monotonic() - last_beat >= 30:
            marker("HEARTBEAT", queued=entry is None)
            last_beat = time.monotonic()
        time.sleep(5)
    raise StageError("TIMEOUT", "inference did not finish before the job deadline")


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise StageError("OUTPUT_NOT_FOUND", "ffprobe cannot read %s: %s" % (path, out.stderr[-500:]))
    return sorted({s.get("codec_type") for s in json.loads(out.stdout).get("streams", [])})


def main():
    started = time.monotonic()
    job = json.loads(Path(os.environ[JOB_FILE_ENV]).read_text(encoding="utf-8"))
    deadline = started + float(job["deadline_seconds"])
    stage = "setup"
    codes = {"setup": "COMFYUI_SETUP_FAILED", "download": "MODEL_DOWNLOAD_FAILED", "inference": "INFERENCE_FAILED"}
    sampler = VramSampler()
    try:
        gpu = check_gpu()
        t = time.monotonic()
        setup_comfyui()
        marker("TIMING", stage="setup", seconds=round(time.monotonic() - t, 1))

        stage = "download"
        marker("STAGE", name="LOADING_MODEL")
        diffusion = choose_diffusion(gpu["cuda"])
        t = time.monotonic()
        sizes = download_models(diffusion)
        marker("TIMING", stage="download", seconds=round(time.monotonic() - t, 1))

        stage = "inference"
        t = time.monotonic()
        proc = start_comfyui(gpu["vram_gib"])
        marker("TIMING", stage="comfyui_start", seconds=round(time.monotonic() - t, 1))
        object_info = http_json("/object_info", timeout=60)
        missing = [name for name in REQUIRED_NODES if name not in object_info]
        if missing:
            raise StageError("COMFYUI_SETUP_FAILED", "ComfyUI is missing nodes %s; see %s" % (missing, COMFY_LOG))

        image = Path(job["remote_image"])
        if not image.is_file():
            raise StageError("INFERENCE_FAILED", "uploaded first frame not found: %s" % image)
        (COMFY / "input").mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, COMFY / "input" / job["comfy_image_name"])

        graph = build_graph(job, diffusion)
        changed = unexpected_inputs(graph, object_info)
        if changed:
            raise StageError("COMFYUI_SETUP_FAILED", "node inputs changed upstream: %s" % changed)
        marker(
            "CONFIG",
            diffusion=diffusion,
            lora=LORA,
            text_encoder=TEXT_ENCODER,
            steps=job["steps"],
            sampler=job["sampler"],
            scheduler=job["scheduler"],
            lowvram=gpu["vram_gib"] < LOWVRAM_BELOW_GIB,
            model_gib=sizes,
        )
        try:
            queued = http_json("/prompt", {"prompt": graph, "client_id": "h3-" + job["job_id"]}, timeout=60)
        except urllib.error.HTTPError as exc:
            raise StageError("INFERENCE_FAILED", "ComfyUI rejected the graph: %s" % exc.read().decode()[:2000])
        if queued.get("node_errors"):
            raise StageError("INFERENCE_FAILED", "node errors: %s" % json.dumps(queued["node_errors"])[:2000])
        marker("STAGE", name="INFERENCE")
        sampler.start()
        t = time.monotonic()
        video = wait_for_video(queued["prompt_id"], deadline, proc)
        marker("TIMING", stage="inference", seconds=round(time.monotonic() - t, 1))
        sampler.stopped.set()
        marker("VRAM_PEAK", mib=sampler.peak)

        streams = probe(video)
        out = Path(job["remote_output"])
        shutil.copy2(video, out)
        marker("OUTPUT", path=str(out), bytes=out.stat().st_size, streams=streams)
        marker("TIMING", stage="total", seconds=round(time.monotonic() - started, 1))
    except StageError as exc:
        marker("ERROR", code=exc.code, message=str(exc)[-2000:])
        raise
    except Exception as exc:
        marker("ERROR", code=codes[stage], message=("%s: %s" % (type(exc).__name__, exc))[-2000:])
        raise
    finally:
        sampler.stopped.set()


if __name__ == "__main__":
    main()
