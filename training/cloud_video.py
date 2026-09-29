"""Run video jobs on cloud GPUs - RunPod Serverless or Replicate - from the local GUI and CLI.

Why this exists: the local RTX 2070 has 8 GB, which caps AnimateDiff's hires pass at 768^2 and rules
out current image-to-video models entirely. Two backends, deliberately not equivalent:

- RunPod runs THIS repo's worker (worker/handler.py + worker/jobs.py), so the prompt is composed by
  the same generate_character code as locally: tier + age safety negatives and the cfg floor are
  guaranteed server-side, and both job types work (Wan 2.2 I2V and AnimateDiff with FaceID).
- Replicate runs someone else's hosted model. Nothing about its prompt handling is ours, so before
  any use the model's input schema is checked (see replicate_safety_gate): it must accept a negative
  prompt AND expose a guidance/cfg field that can be set to at least SAFETY_MIN_CFG, otherwise it is
  refused. Many hosted video models fail that check; that is the expected outcome, not a bug.

Secrets are read from the environment only (RUNPOD_API_KEY, REPLICATE_API_TOKEN) and never written
to disk. Non-secret settings (endpoint id, Replicate model, timeouts, field names) live in
training/settings/cloud.json, which is gitignored.

API details used here were read from the providers' docs on 2026-09-17, not recalled:
RunPod  POST /v2/{endpoint}/run {"input", "policy": {"executionTimeout": ms}} (payload <= 10 MB,
        results kept 30 min), GET /status/{id} -> status IN_QUEUE|IN_PROGRESS|COMPLETED|FAILED|
        CANCELLED|TIMED_OUT with delayTime/executionTime in ms, POST /cancel/{id}; Bearer auth.
Replicate GET /v1/models/{owner}/{name} -> latest_version.openapi_schema.components.schemas.Input,
        POST /v1/predictions {"version", "input"}, GET /v1/predictions/{id} -> starting|processing|
        succeeded|failed|canceled, POST /v1/predictions/{id}/cancel; data URLs only for files
        <= 256 KB; outputs deleted after an hour; Bearer auth.

CLI (run with ComfyUI's venv, like the other training scripts):
    python training/cloud_video.py status
    python training/cloud_video.py run --provider runpod --job wan-i2v --image anchor.png --prompt "..."
    python training/cloud_video.py run ... --preview --count 3     # three cheap 480x832 drafts, one batch
    python training/cloud_video.py cancel --provider runpod --job-id <id>
"""

import argparse
import base64
import dataclasses
import io
import json
import os
import sys
import time

import requests

import comfyui_client as client
import generate_character as gc

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.environ.get("CLOUD_SETTINGS_FILE") or os.path.join(TRAINING_DIR, "settings", "cloud.json")
SETTINGS_VERSION = 1

PROVIDERS = ("runpod", "replicate")
JOB_TYPES = ("video_wan_i2v", "video_animatediff")
RUNPOD_API = "https://api.runpod.ai/v2"
REPLICATE_API = "https://api.replicate.com/v1"
RUNPOD_MAX_PAYLOAD_BYTES = 9_500_000       # documented /run limit is 10 MB; keep headroom
REPLICATE_MAX_DATA_URI_BYTES = 256_000     # documented: data URLs only for files <= 256 KB
POLL_INTERVAL_SECONDS = 4
HTTP_TIMEOUT_SECONDS = 30
GET_RETRIES = 4

DEFAULT_REPLICATE_FIELDS = {
    "prompt": "prompt",
    "negative_prompt": "negative_prompt",
    "image": "image",
    "cfg": "guidance_scale",
    "frames": "num_frames",
    "fps": "frames_per_second",
    "seed": "seed",
}

# A cheap draft to judge motion and framing before paying for the 704x1280 render: about 19% of
# the full job's latent tokens and 12 of its 20 steps, so roughly a tenth of the cost (estimated
# from the token count, not measured). Resolution and length differ from the final, so the same
# seed does NOT give a smaller copy of the final video - only the direction of it.
WAN_PREVIEW_PORTRAIT = {"width": 480, "height": 832, "frames": 49, "steps": 12}
WAN_PREVIEW_LANDSCAPE = {"width": 832, "height": 480, "frames": 49, "steps": 12}

DEFAULT_SETTINGS = {
    "version": SETTINGS_VERSION,
    "runpod": {"endpoint_id": "", "execution_timeout_s": 1800},
    "replicate": {"model": "", "fields": dict(DEFAULT_REPLICATE_FIELDS)},
    "max_wait_s": 2400,
}

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


class CloudJobFailed(RuntimeError):
    """The remote job ended without a usable result (failed, timed out, cancelled, bad output)."""

    def __init__(self, message, *, provider=None, job_id=None, detail=None):
        super().__init__(message)
        self.provider = provider
        self.job_id = job_id
        self.detail = detail


class CloudJobCancelled(CloudJobFailed):
    """The job was cancelled from this side (cancel button, Ctrl-C)."""


@dataclasses.dataclass
class CloudResult:
    path: str
    provider: str
    job_id: str
    seed: int
    elapsed_s: float
    queue_s: float = None
    execution_s: float = None
    raw: dict = None


@dataclasses.dataclass
class CloudJobSpec:
    """One job of a run_cloud_batch call. stem names the downloaded file; the extension comes from
    the provider's output."""
    stem: str
    prompt: str
    image_path: str = None
    trigger: str = None
    seed: int = None
    params: dict = None
    extra_negative: str = ""
    tier: str = "safe"


# --- settings ---------------------------------------------------------------------------------


def _merge(base, updates):
    out = dict(base)
    for k, v in updates.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _secret_like_keys(obj, prefix=""):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}{k}"
            if any(word in str(k).lower() for word in ("key", "token", "secret", "password")):
                found.append(name)
            found.extend(_secret_like_keys(v, name + "."))
    return found


def load_cloud_settings(path=None):
    """Settings merged over DEFAULT_SETTINGS. A missing or unreadable file gives the defaults: the
    worst case is 'not configured yet', which config_status reports, never a crash."""
    path = path or SETTINGS_FILE
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return json.loads(json.dumps(DEFAULT_SETTINGS))
    if not isinstance(data, dict):
        return json.loads(json.dumps(DEFAULT_SETTINGS))
    data.pop("version", None)
    merged = _merge(DEFAULT_SETTINGS, data)
    merged["version"] = SETTINGS_VERSION
    return merged


def save_cloud_settings(updates, path=None):
    """Merge `updates` into the stored settings and write atomically. Refuses anything that looks
    like a secret - keys and tokens belong in environment variables, not a file on disk."""
    secrets = _secret_like_keys(updates)
    if secrets:
        raise gc.UsageError(f"API 金鑰／token 不能存進設定檔（{', '.join(secrets)}）——請改用環境變數 "
                            "RUNPOD_API_KEY／REPLICATE_API_TOKEN")
    path = path or SETTINGS_FILE
    merged = _merge(load_cloud_settings(path), updates)
    merged["version"] = SETTINGS_VERSION
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def config_status(provider=None, settings=None, env=None):
    """What's missing before a provider can run, as user-facing Chinese lines (empty = ready)."""
    settings = settings or load_cloud_settings()
    env = os.environ if env is None else env
    problems = []
    for name in ([provider] if provider else PROVIDERS):
        if name == "runpod":
            if not env.get("RUNPOD_API_KEY"):
                problems.append("RunPod：沒有設定環境變數 RUNPOD_API_KEY")
            if not settings["runpod"].get("endpoint_id"):
                problems.append("RunPod：還沒填 Serverless endpoint ID（settings/cloud.json 的 runpod.endpoint_id）")
        elif name == "replicate":
            if not env.get("REPLICATE_API_TOKEN"):
                problems.append("Replicate：沒有設定環境變數 REPLICATE_API_TOKEN")
            if not settings["replicate"].get("model"):
                problems.append("Replicate：還沒選模型（settings/cloud.json 的 replicate.model，格式 owner/name）")
    return problems


# --- HTTP -------------------------------------------------------------------------------------


def _raise_for_http(resp, provider):
    code = resp.status_code
    if code < 400:
        return
    try:
        detail = resp.json()
    except ValueError:
        detail = (resp.text or "")[:300]
    if code in (401, 403):
        raise gc.UsageError(f"{provider}：API 金鑰無效或權限不足（HTTP {code}）")
    if code == 402:
        raise gc.UsageError(f"{provider}：帳戶餘額不足（HTTP 402）")
    if code == 429:
        raise gc.UsageError(f"{provider}：請求太頻繁，請稍後再試（HTTP 429）")
    if code == 404:
        raise gc.UsageError(f"{provider}：找不到資源（HTTP 404）——endpoint ID 或模型名稱可能打錯：{detail}")
    raise CloudJobFailed(f"{provider}：伺服器錯誤 HTTP {code}：{detail}", provider=provider, detail=detail)


def _get_json(session, url, headers, provider, sleep=time.sleep):
    """GETs are idempotent, so connection errors and 5xx are retried with backoff."""
    delay = 2
    for attempt in range(GET_RETRIES + 1):
        try:
            resp = session.get(url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == GET_RETRIES:
                raise CloudJobFailed(f"{provider}：連線失敗（已重試 {GET_RETRIES} 次）", provider=provider)
            sleep(delay)
            delay *= 2
            continue
        if resp.status_code >= 500 and attempt < GET_RETRIES:
            sleep(delay)
            delay *= 2
            continue
        _raise_for_http(resp, provider)
        return resp.json()


def _post_json(session, url, headers, body, provider):
    """POSTs that CREATE a job are never retried: a request that reached the server but whose
    response was lost would otherwise start a second billed job."""
    try:
        resp = session.post(url, headers=headers, json=body, timeout=HTTP_TIMEOUT_SECONDS)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        raise CloudJobFailed(f"{provider}：送出工作時連線失敗（{type(exc).__name__}）——請到 {provider} 後台確認"
                             "是否已經建立了工作，避免重複計費", provider=provider) from exc
    _raise_for_http(resp, provider)
    return resp.json()


def _download(session, url, dest, headers=None):
    part = dest + ".part"
    with session.get(url, headers=headers or {}, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as resp:
        if resp.status_code >= 400:
            raise CloudJobFailed(f"下載結果失敗（HTTP {resp.status_code}）——連結可能已過期")
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    os.replace(part, dest)
    return dest


# --- images -----------------------------------------------------------------------------------


def _read_image(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise gc.UsageError(f"讀不到第一幀圖片：{path}（{exc}）") from exc
    if raw.startswith(_PNG_MAGIC):
        return raw, "image/png"
    if raw.startswith(_JPEG_MAGIC):
        return raw, "image/jpeg"
    raise gc.UsageError("第一幀圖片必須是 PNG 或 JPEG")


def _shrink_to_jpeg(raw, max_bytes, max_side=1280):
    """Re-encode as a JPEG no larger than max_bytes, stepping quality then size down. PIL is imported
    lazily - it's in ComfyUI's venv, where the GUI and CLI run, but not in .venv-dev."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise gc.UsageError(f"圖片太大（{len(raw)} bytes），需要 Pillow 才能自動縮小") from exc
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    side = max_side
    while side >= 256:
        scaled = img.copy()
        scaled.thumbnail((side, side), Image.LANCZOS)
        for quality in (90, 80, 70, 60):
            buf = io.BytesIO()
            scaled.save(buf, format="JPEG", quality=quality)
            if buf.tell() <= max_bytes:
                return buf.getvalue()
        side = int(side * 0.75)
    raise gc.UsageError(f"圖片無法壓縮到 {max_bytes} bytes 以內")


# --- payloads (pure, tested offline) ----------------------------------------------------------


def build_runpod_input(job_type, *, prompt, extra_negative, tier, trigger, image_path, seed, params):
    """The worker's input dict. Sends the RAW prompt and tier: worker/jobs.py composes the negatives
    with the same _build_prompt_and_negative as locally, so nothing here can weaken them."""
    if job_type not in JOB_TYPES:
        raise gc.UsageError(f"unknown job type {job_type!r}; choices: {list(JOB_TYPES)}")
    inp = {"jobType": job_type, "prompt": prompt, "tier": tier, "seed": int(seed)}
    if extra_negative:
        inp["negativePrompt"] = extra_negative
    if trigger:
        inp["characterId"] = trigger
    if image_path:
        raw, _ = _read_image(image_path)
        inp["referenceImageBase64"] = base64.b64encode(raw).decode("ascii")
    elif not trigger:
        raise gc.UsageError("需要第一幀／臉部參考圖，或選一個角色（worker 會用 volume 上的 anchor）")
    inp.update({k: v for k, v in params.items() if v is not None})
    size = len(json.dumps({"input": inp}).encode("utf-8"))
    if size > RUNPOD_MAX_PAYLOAD_BYTES:
        raise gc.UsageError(f"送出的資料太大（{size} bytes，RunPod /run 上限約 10 MB）——請換一張比較小的圖")
    return inp


def replicate_safety_gate(schema_properties, fields):
    """Refuse a hosted model unless its prompt handling can carry this project's safety guarantees.

    Required: a negative-prompt input (AGE_SAFETY_NEGATIVE has to go somewhere), an image input, and a
    guidance/cfg input whose allowed maximum reaches SAFETY_MIN_CFG. The cfg requirement is not
    optional polish: a model with no cfg control is frequently a distilled cfg-1 model, and at cfg 1
    the negative conditioning is skipped - the negative field would exist and do nothing."""
    props = schema_properties or {}
    problems = []
    if fields.get("negative_prompt") not in props:
        problems.append(f"沒有負面詞欄位「{fields.get('negative_prompt')}」")
    if fields.get("image") not in props:
        problems.append(f"沒有圖片輸入欄位「{fields.get('image')}」")
    cfg_spec = props.get(fields.get("cfg"))
    if cfg_spec is None:
        problems.append(f"沒有 guidance／cfg 欄位「{fields.get('cfg')}」")
    elif isinstance(cfg_spec.get("maximum"), (int, float)) and cfg_spec["maximum"] < client.SAFETY_MIN_CFG:
        problems.append(f"cfg 上限只有 {cfg_spec['maximum']}，低於安全下限 {client.SAFETY_MIN_CFG}")
    if problems:
        raise gc.UsageError(
            "這個 Replicate 模型無法套用本專案的安全規則，拒絕使用：" + "；".join(problems)
            + "。（若欄位只是名稱不同，可在 settings/cloud.json 的 replicate.fields 對應）")


def build_replicate_input(schema_properties, fields, *, prompt, extra_negative, tier, trigger, image_path,
                          seed, cfg, frames=None, fps=None, shrink=_shrink_to_jpeg):
    """Input dict for a hosted model that already passed replicate_safety_gate. The prompt pair is
    composed locally with the shared builder, so the safety negatives are present no matter what."""
    replicate_safety_gate(schema_properties, fields)
    full_prompt, negative_prompt = gc.build_wan_prompts(prompt, extra_negative, tier, trigger)
    raw, mime = _read_image(image_path)
    if len(raw) > REPLICATE_MAX_DATA_URI_BYTES:
        raw, mime = shrink(raw, REPLICATE_MAX_DATA_URI_BYTES), "image/jpeg"

    cfg_spec = schema_properties[fields["cfg"]]
    effective_cfg = max(float(cfg), client.SAFETY_MIN_CFG)
    if isinstance(cfg_spec.get("maximum"), (int, float)):
        effective_cfg = min(effective_cfg, float(cfg_spec["maximum"]))
    if isinstance(cfg_spec.get("minimum"), (int, float)):
        effective_cfg = max(effective_cfg, float(cfg_spec["minimum"]))

    inp = {
        fields["prompt"]: full_prompt,
        fields["negative_prompt"]: negative_prompt,
        fields["image"]: f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
        fields["cfg"]: effective_cfg,
    }
    for key, value in (("seed", seed), ("frames", frames), ("fps", fps)):
        name = fields.get(key)
        if value is not None and name in schema_properties:
            inp[name] = int(value)
    return inp


# --- backends ---------------------------------------------------------------------------------


class RunPodBackend:
    name = "runpod"

    def __init__(self, session, api_key, endpoint_id, sleep=time.sleep):
        self.session = session
        self.sleep = sleep
        self.base = f"{RUNPOD_API}/{endpoint_id}"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def submit(self, inp, execution_timeout_s):
        body = {"input": inp, "policy": {"executionTimeout": int(execution_timeout_s * 1000)}}
        data = _post_json(self.session, f"{self.base}/run", self.headers, body, self.name)
        job_id = data.get("id")
        if not job_id:
            raise CloudJobFailed(f"RunPod 沒有回傳工作 ID：{data}", provider=self.name, detail=data)
        return job_id

    def poll(self, job_id):
        data = _get_json(self.session, f"{self.base}/status/{job_id}", self.headers, self.name, sleep=self.sleep)
        status = data.get("status", "")
        state = {"IN_QUEUE": "queued", "IN_PROGRESS": "running", "COMPLETED": "succeeded",
                 "FAILED": "failed", "CANCELLED": "cancelled", "TIMED_OUT": "timed_out"}.get(status, "running")
        label = {"queued": "排隊中（冷啟動可能要幾分鐘）", "running": "雲端生成中"}.get(state, status)
        output = data.get("output")
        if state == "running" and isinstance(output, dict) and output.get("stage"):
            label = f"雲端{output['stage']} {output.get('current', '?')}/{output.get('total', '?')}"
        if state == "succeeded" and isinstance(output, dict) and output.get("error"):
            state = "failed"   # the worker returns {"error": ...} instead of raising
        return state, label, data

    def cancel(self, job_id):
        try:
            self.session.post(f"{self.base}/cancel/{job_id}", headers=self.headers, timeout=HTTP_TIMEOUT_SECONDS)
        except Exception:
            pass   # best effort; the caller is already on an error path

    def result(self, data, dest_dir, stem):
        output = data.get("output") or {}
        url = output.get("outputUrl") if isinstance(output, dict) else None
        if not url:
            raise CloudJobFailed(f"RunPod 工作完成但沒有輸出連結：{output}", provider=self.name, detail=output)
        # The worker keys the upload as generated/<job>.<ext>; video jobs give mp4, workflow jobs can
        # also give png or webm. Only a known extension is trusted - it becomes part of a local path.
        ext = os.path.splitext(str(output.get("outputKey") or ""))[1].lower().lstrip(".")
        ext = ext if ext in ("mp4", "png", "webm") else "mp4"
        path = _download(self.session, url, os.path.join(dest_dir, f"{stem}.{ext}"))
        queue = data.get("delayTime")
        execution = data.get("executionTime")
        # `is not None`, not a truthiness test: a provider reporting 0 ms of queue wait is
        # information, and folding that into None makes "the job started instantly" and "RunPod
        # told us nothing" indistinguishable. The distinction only became load-bearing when
        # job_contracts started treating these as a nullable tri-state whose consumers are
        # forbidden from inferring an unknown value.
        return (path,
                (queue / 1000.0 if queue is not None else None),
                (execution / 1000.0 if execution is not None else None))


class ReplicateBackend:
    name = "replicate"

    def __init__(self, session, token, model, sleep=time.sleep):
        self.session = session
        self.sleep = sleep
        self.model = model
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self._version = None

    def fetch_schema(self):
        """Returns the model's Input schema properties, pinning the version the check ran against so
        the prediction runs on exactly the model that passed the safety gate."""
        if not self.model or "/" not in self.model:
            raise gc.UsageError(f"Replicate 模型名稱格式應為 owner/name 或 owner/name:version，收到 {self.model!r}")
        owner_name, _, version = self.model.partition(":")
        if version:
            url = f"{REPLICATE_API}/models/{owner_name}/versions/{version}"
            data = _get_json(self.session, url, self.headers, self.name, sleep=self.sleep)
            version_data = data
        else:
            data = _get_json(self.session, f"{REPLICATE_API}/models/{owner_name}", self.headers, self.name,
                             sleep=self.sleep)
            version_data = data.get("latest_version") or {}
        self._version = version_data.get("id")
        try:
            props = version_data["openapi_schema"]["components"]["schemas"]["Input"]["properties"]
        except (KeyError, TypeError) as exc:
            raise gc.UsageError(f"讀不到 Replicate 模型 {self.model} 的輸入格式") from exc
        if not self._version:
            raise gc.UsageError(f"Replicate 模型 {self.model} 沒有可用的版本")
        return props

    def submit(self, inp, execution_timeout_s):
        body = {"version": self._version, "input": inp}
        data = _post_json(self.session, f"{REPLICATE_API}/predictions", self.headers, body, self.name)
        if not data.get("id"):
            raise CloudJobFailed(f"Replicate 沒有回傳 prediction ID：{data}", provider=self.name, detail=data)
        return data["id"]

    def poll(self, job_id):
        data = _get_json(self.session, f"{REPLICATE_API}/predictions/{job_id}", self.headers, self.name,
                         sleep=self.sleep)
        status = data.get("status", "")
        state = {"starting": "queued", "processing": "running", "succeeded": "succeeded",
                 "failed": "failed", "canceled": "cancelled"}.get(status, "running")
        label = {"queued": "啟動中（冷啟動可能要幾分鐘）", "running": "雲端生成中"}.get(state, status)
        return state, label, data

    def cancel(self, job_id):
        try:
            self.session.post(f"{REPLICATE_API}/predictions/{job_id}/cancel", headers=self.headers,
                              timeout=HTTP_TIMEOUT_SECONDS)
        except Exception:
            pass

    def result(self, data, dest_dir, stem):
        url = pick_output_url(data.get("output"))
        if not url:
            raise CloudJobFailed(f"Replicate 工作完成但沒有輸出：{data.get('output')}", provider=self.name)
        # api.replicate.com file URLs need the token; replicate.delivery links are pre-signed.
        headers = {"Authorization": self.headers["Authorization"]} if url.startswith(REPLICATE_API) else None
        path = _download(self.session, url, os.path.join(dest_dir, f"{stem}.mp4"), headers=headers)
        metrics = data.get("metrics") or {}
        return path, None, metrics.get("predict_time")


def pick_output_url(output):
    """Replicate outputs are a URL string or a list of them; prefer an .mp4 in a list."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        urls = [u for u in output if isinstance(u, str)]
        for u in urls:
            if u.split("?")[0].lower().endswith(".mp4"):
                return u
        return urls[0] if urls else None
    return None


# --- orchestration ----------------------------------------------------------------------------


def wait_for_job(backend, job_id, *, max_wait_s, on_status=None, cancel_event=None,
                 clock=time.monotonic, sleep=time.sleep):
    """Poll until the job finishes. Every way of walking away - timeout, cancel button, Ctrl-C -
    sends the remote cancel first, because an abandoned cloud job keeps billing."""
    start = clock()
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                backend.cancel(job_id)
                raise CloudJobCancelled("已取消雲端工作", provider=backend.name, job_id=job_id)
            state, label, data = backend.poll(job_id)
            elapsed = clock() - start
            if on_status is not None:
                try:
                    on_status(label, int(elapsed), int(max_wait_s))
                except Exception:
                    pass
            if state == "succeeded":
                return data
            if state in ("failed", "cancelled", "timed_out"):
                detail = data.get("error") or (data.get("output") or {}) if isinstance(data, dict) else data
                raise CloudJobFailed(f"{backend.name} 工作沒有成功（{state}）：{detail}",
                                     provider=backend.name, job_id=job_id, detail=data)
            if elapsed >= max_wait_s:
                backend.cancel(job_id)
                raise CloudJobFailed(f"等了 {int(elapsed)} 秒還沒完成，已取消雲端工作（上限 {int(max_wait_s)} 秒）",
                                     provider=backend.name, job_id=job_id)
            sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        backend.cancel(job_id)
        raise


def make_backend(provider, *, session=None, settings=None, env=None, sleep=time.sleep):
    settings = settings or load_cloud_settings()
    env = os.environ if env is None else env
    session = session or requests.Session()
    problems = config_status(provider, settings, env)
    if problems:
        raise gc.UsageError("雲端設定還沒完成：" + "；".join(problems))
    if provider == "runpod":
        return RunPodBackend(session, env["RUNPOD_API_KEY"], settings["runpod"]["endpoint_id"], sleep=sleep)
    if provider == "replicate":
        return ReplicateBackend(session, env["REPLICATE_API_TOKEN"], settings["replicate"]["model"], sleep=sleep)
    raise gc.UsageError(f"unknown provider {provider!r}; choices: {list(PROVIDERS)}")


def _normalise_tier(tier):
    return tier if tier in ("safe", "suggestive") else "safe"


def _validate_request(prompt, trigger):
    """The checks that need no settings and no network, so a bad request never gets as far as a
    configuration error (or, in a batch, a half-submitted queue)."""
    if not (prompt or "").strip():
        raise gc.UsageError("請輸入 prompt（描述要發生的動作）")
    if len(prompt) > gc.MAX_PROMPT_CHARS:
        raise gc.UsageError(f"prompt 太長（上限 {gc.MAX_PROMPT_CHARS} 字）")
    if trigger:
        gc.get_character(trigger)


def _build_input(provider, job_type, backend, settings, *, prompt, extra_negative, tier, trigger, image_path, seed,
                 params, replicate_props=None):
    if provider == "runpod":
        return build_runpod_input(job_type, prompt=prompt, extra_negative=extra_negative, tier=tier, trigger=trigger,
                                  image_path=image_path, seed=seed, params=params)
    if job_type != "video_wan_i2v":
        raise gc.UsageError("AnimateDiff 雲端高畫質只能跑在 RunPod（需要本專案自己的 FaceID 流程）")
    if not image_path:
        raise gc.UsageError("Replicate 需要一張第一幀圖片")
    fields = _merge(DEFAULT_REPLICATE_FIELDS, settings["replicate"].get("fields") or {})
    props = replicate_props if replicate_props is not None else backend.fetch_schema()
    return build_replicate_input(props, fields, prompt=prompt, extra_negative=extra_negative, tier=tier,
                                 trigger=trigger, image_path=image_path, seed=seed,
                                 cfg=params.get("cfg", client.WAN_CFG), frames=params.get("frames"),
                                 fps=params.get("fps"))


def _report(on_status, label, elapsed, limit):
    if on_status is not None:
        try:
            on_status(label, int(elapsed), int(limit))
        except Exception:
            pass


def _random_seed():
    return int(time.time()) % (2**31)


def run_cloud_video(provider, job_type, *, prompt, extra_negative="", tier="safe", trigger=None, image_path=None,
                    seed=None, params=None, out_dir, max_wait_s=None, on_status=None, cancel_event=None,
                    session=None, settings=None, env=None, sleep=time.sleep, clock=time.monotonic):
    """Submit one video job, wait for it, download the mp4. Returns a CloudResult."""
    tier = _normalise_tier(tier)
    _validate_request(prompt, trigger)
    params = dict(params or {})
    settings = settings or load_cloud_settings()
    backend = make_backend(provider, session=session, settings=settings, env=env, sleep=sleep)
    seed = int(seed) if seed is not None else _random_seed()
    max_wait_s = max_wait_s or settings["max_wait_s"]
    inp = _build_input(provider, job_type, backend, settings, prompt=prompt, extra_negative=extra_negative, tier=tier,
                       trigger=trigger, image_path=image_path, seed=seed, params=params)

    started = clock()
    job_id = backend.submit(inp, settings["runpod"]["execution_timeout_s"])
    _report(on_status, f"已送出（{provider} 工作 {job_id}）", 0, max_wait_s)
    data = wait_for_job(backend, job_id, max_wait_s=max_wait_s, on_status=on_status, cancel_event=cancel_event,
                        clock=clock, sleep=sleep)
    os.makedirs(out_dir, exist_ok=True)
    short = "wan" if job_type == "video_wan_i2v" else "animatediff"
    # Keyed by the provider's job id: the local GUI's gui_seed<seed> names collide across sessions.
    stem = f"cloud_{provider}_{short}_seed{seed}_{str(job_id)[:8]}"
    path, queue_s, execution_s = backend.result(data, out_dir, stem)
    return CloudResult(path=path, provider=provider, job_id=job_id, seed=seed,
                       elapsed_s=round(clock() - started, 1), queue_s=queue_s, execution_s=execution_s, raw=data)


def run_cloud_batch(provider, job_type, specs, *, out_dir, max_wait_s=None, on_status=None, cancel_event=None,
                    session=None, settings=None, env=None, sleep=time.sleep, clock=time.monotonic):
    """Submit every job first, then poll them together, and return [(spec, CloudResult or
    CloudJobFailed)] in spec order.

    Queuing everything up front is the point: the endpoint's warm worker works through the queue
    back to back, so only the first job pays the cold start (18 GB of Wan weights). Submitting one,
    waiting, then submitting the next lets the worker idle out in between and pay it every time.

    Every spec is validated and turned into a payload before the first POST, so a bad prompt or an
    oversized image can't leave half a batch billing. One job failing doesn't stop the others. The
    local deadline is max_wait_s per job times the batch size, because an endpoint capped at one
    worker runs the queue in sequence; each job's own execution is capped server-side by the
    endpoint policy regardless. Any way of walking away - deadline, cancel_event, Ctrl-C, an HTTP
    error mid-batch - cancels every job that hasn't finished."""
    specs = list(specs)
    if not specs:
        return []
    stems = [s.stem for s in specs]
    if len(set(stems)) != len(stems):
        raise gc.UsageError("同一批工作的輸出檔名（stem）不能重複")
    for spec in specs:
        _validate_request(spec.prompt, spec.trigger)
    settings = settings or load_cloud_settings()
    backend = make_backend(provider, session=session, settings=settings, env=env, sleep=sleep)
    max_wait_s = max_wait_s or settings["max_wait_s"]
    props = backend.fetch_schema() if provider == "replicate" and job_type == "video_wan_i2v" else None
    prepared = []
    for spec in specs:
        seed = int(spec.seed) if spec.seed is not None else _random_seed()
        inp = _build_input(provider, job_type, backend, settings, prompt=spec.prompt,
                           extra_negative=spec.extra_negative, tier=_normalise_tier(spec.tier), trigger=spec.trigger,
                           image_path=spec.image_path, seed=seed, params=dict(spec.params or {}),
                           replicate_props=props)
        prepared.append((spec, seed, inp))

    os.makedirs(out_dir, exist_ok=True)
    results = [None] * len(specs)
    pending = {}   # job_id -> (index, spec, seed)
    deadline = max_wait_s * len(specs)
    started = clock()
    try:
        for index, (spec, seed, inp) in enumerate(prepared):
            job_id = backend.submit(inp, settings["runpod"]["execution_timeout_s"])
            pending[job_id] = (index, spec, seed)
        _report(on_status, f"已送出 {len(specs)} 支（{provider}），雲端會依序處理", 0, deadline)
        while pending:
            if cancel_event is not None and cancel_event.is_set():
                raise CloudJobCancelled("已取消雲端工作", provider=backend.name)
            running = queued = 0
            for job_id in list(pending):
                index, spec, seed = pending[job_id]
                state, _label, data = backend.poll(job_id)
                if state == "succeeded":
                    del pending[job_id]
                    try:
                        path, queue_s, execution_s = backend.result(data, out_dir, spec.stem)
                    except CloudJobFailed as exc:
                        results[index] = exc
                        continue
                    results[index] = CloudResult(path=path, provider=provider, job_id=job_id, seed=seed,
                                                 elapsed_s=round(clock() - started, 1), queue_s=queue_s,
                                                 execution_s=execution_s, raw=data)
                elif state in ("failed", "cancelled", "timed_out"):
                    del pending[job_id]
                    detail = (data.get("error") or data.get("output") or {}) if isinstance(data, dict) else data
                    results[index] = CloudJobFailed(f"{backend.name} 工作沒有成功（{state}）：{detail}",
                                                    provider=backend.name, job_id=job_id, detail=data)
                elif state == "queued":
                    queued += 1
                else:
                    running += 1
            elapsed = clock() - started
            done = len(specs) - len(pending)
            _report(on_status, f"完成 {done}/{len(specs)}（生成中 {running}、排隊 {queued}）", elapsed, deadline)
            if not pending:
                break
            if elapsed >= deadline:
                for job_id, (index, _spec, _seed) in pending.items():
                    backend.cancel(job_id)
                    results[index] = CloudJobFailed(
                        f"等了 {int(elapsed)} 秒還沒完成，已取消雲端工作（整批上限 {int(deadline)} 秒）",
                        provider=backend.name, job_id=job_id)
                pending.clear()
                break
            sleep(POLL_INTERVAL_SECONDS)
    except BaseException:
        for job_id in pending:
            backend.cancel(job_id)
        raise
    return list(zip(specs, results))


# --- CLI --------------------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description="Run video jobs on RunPod Serverless or Replicate")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show what is configured and what is missing")

    run = sub.add_parser("run", help="submit one video job and download the result")
    run.add_argument("--provider", choices=PROVIDERS, required=True)
    run.add_argument("--job", choices=("wan-i2v", "animatediff"), required=True)
    run.add_argument("--image", help="first frame (Wan) / face reference (AnimateDiff): a generated fictional "
                                     "character image, never a real person's photo")
    run.add_argument("--character", help="character id; also lets the RunPod worker fall back to its anchor")
    run.add_argument("--prompt", required=True)
    run.add_argument("--negative", default="", help="extra negative terms (the safety terms are always added)")
    run.add_argument("--tier", choices=("safe", "suggestive"), default="safe")
    run.add_argument("--seed", type=int)
    run.add_argument("--frames", type=int)
    run.add_argument("--width", type=int)
    run.add_argument("--height", type=int)
    run.add_argument("--fps", type=int)
    run.add_argument("--cfg", type=float)
    run.add_argument("--steps", type=int)
    run.add_argument("--preview", action="store_true",
                     help="cheap Wan draft: 480x832, 49 frames, 12 steps (about a tenth of a full render)")
    run.add_argument("--landscape", action="store_true", help="with --preview: 832x480 instead of 480x832")
    run.add_argument("--count", type=int, default=1,
                     help="submit this many seeds (seed, seed+1, ...) as one batch so the warm worker runs "
                          "them back to back and only the first pays the cold start")
    run.add_argument("--timeout", type=int, help="max seconds to wait per job before cancelling")
    run.add_argument("--out", default=os.path.join(TRAINING_DIR, "reference_candidates", "videos"))

    cancel = sub.add_parser("cancel", help="cancel a remote job by id")
    cancel.add_argument("--provider", choices=PROVIDERS, required=True)
    cancel.add_argument("--job-id", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "status":
        problems = config_status()
        settings = load_cloud_settings()
        print(f"settings file: {SETTINGS_FILE}")
        print(f"runpod endpoint: {settings['runpod']['endpoint_id'] or '(未設定)'}")
        print(f"replicate model: {settings['replicate']['model'] or '(未設定)'}")
        print("\n".join(problems) if problems else "全部設定完成")
        return 0 if not problems else 1
    if args.cmd == "cancel":
        backend = make_backend(args.provider)
        backend.cancel(args.job_id)
        print(f"已送出取消：{args.provider} {args.job_id}")
        return 0

    job_type = "video_wan_i2v" if args.job == "wan-i2v" else "video_animatediff"
    params = cli_params(args, job_type)
    on_status = print_status
    if args.count < 1:
        raise gc.UsageError("--count 至少要 1")
    if args.count == 1:
        result = run_cloud_video(
            args.provider, job_type, prompt=args.prompt, extra_negative=args.negative, tier=args.tier,
            trigger=args.character, image_path=args.image, seed=args.seed, params=params, out_dir=args.out,
            max_wait_s=args.timeout, on_status=on_status,
        )
        print(f"saved {result.path}")
        print(f"job {result.job_id} | total {result.elapsed_s}s | queue {result.queue_s}s | "
              f"execution {result.execution_s}s")
        return 0

    base_seed = args.seed if args.seed is not None else _random_seed()
    short = "wan" if job_type == "video_wan_i2v" else "animatediff"
    batch_tag = time.strftime("%Y%m%d%H%M%S")
    specs = [CloudJobSpec(stem=f"cloud_{args.provider}_{short}_seed{base_seed + i}_{batch_tag}", prompt=args.prompt,
                          image_path=args.image, trigger=args.character, seed=base_seed + i, params=params,
                          extra_negative=args.negative, tier=args.tier)
             for i in range(args.count)]
    results = run_cloud_batch(args.provider, job_type, specs, out_dir=args.out, max_wait_s=args.timeout,
                              on_status=on_status)
    failed = 0
    for spec, outcome in results:
        if isinstance(outcome, CloudResult):
            print(f"seed {spec.seed}: saved {outcome.path} (execution {outcome.execution_s}s)")
        else:
            failed += 1
            print(f"seed {spec.seed}: 失敗 - {outcome}")
    return 1 if failed else 0


def print_status(label, elapsed, limit):
    print(f"[{elapsed:>4}s/{limit}s] {label}", flush=True)


def cli_params(args, job_type):
    """The params dict for `run`. --preview only fills in what wasn't given explicitly."""
    params = {"frames": args.frames, "width": args.width, "height": args.height, "fps": args.fps, "cfg": args.cfg,
              "steps": args.steps}
    if args.preview:
        if job_type != "video_wan_i2v":
            raise gc.UsageError("--preview 只適用於 Wan 圖生影片")
        preset = WAN_PREVIEW_LANDSCAPE if args.landscape else WAN_PREVIEW_PORTRAIT
        for key, value in preset.items():
            if params.get(key) is None:
                params[key] = value
    return params


if __name__ == "__main__":
    try:
        sys.exit(main())
    except gc.UsageError as exc:
        raise SystemExit(str(exc))
    except CloudJobFailed as exc:
        raise SystemExit(f"雲端工作失敗：{exc}")
