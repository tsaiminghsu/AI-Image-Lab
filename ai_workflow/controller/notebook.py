"""What the notebooks call. A notebook cell holds a form and one call into this module; the logic stays here
and in the rest of the package, so the notebooks never need to change when the core does.

Manual mode and the tool interface meet in generate(): it calls tools.create_job() - the same function an
assistant's tool call reaches - and then lets the worker run the job in this kernel.

Everything printed here is for the person at the notebook, so it is in Traditional Chinese.
"""

import json
import shutil
import subprocess
from pathlib import Path

from . import WorkflowError, jobs, tools
from .environment import Environment
from .storage import DEFAULT_RUNTIME_ROOT, DriveFolderStorage
from .worker import Worker

MOUNT_POINT = "/content/drive"
OK, BAD = "✓", "✗"


class Session:
    def __init__(self, storage, env, worker):
        self.storage = storage
        self.env = env
        self.worker = worker


def mount_drive():
    """Mount Google Drive when running on Colab; a no-op anywhere else."""
    try:
        from google.colab import drive
    except ImportError:
        return False
    drive.mount(MOUNT_POINT)
    return True


def _flush_drive():
    from google.colab import drive

    drive.flush_and_unmount()


def _release_runtime():
    from google.colab import runtime

    runtime.unassign()


def connect(root=DEFAULT_RUNTIME_ROOT, *, mount=True, env=None, worker=None):
    if mount:
        mount_drive()
    path = Path(root)
    if not (path / "workflows" / "registry.json").is_file():
        raise WorkflowError(
            "WORKSPACE_NOT_FOUND",
            "找不到工作區：%s 裡沒有 workflows/registry.json。" % path,
            hint="先在本機執行 scripts/deploy.py 把平台部署到 Google Drive 的 AI-Workflow 資料夾。",
        )
    storage = DriveFolderStorage(path)
    storage.ensure_layout()
    config = storage.read_json(tools.PLATFORM_CONFIG, {}) or {}
    environment = env or Environment(storage, config)
    session = Session(
        storage, environment, worker or Worker(storage, environment, flush_fn=_flush_drive, release_fn=_release_runtime)
    )
    print("工作區：%s" % path)
    return session


def _table(rows):
    width = max((len(str(r[0])) for r in rows), default=0)
    for name, value, ok in rows:
        print("  %s %s  %s" % (OK if ok else BAD, str(name).ljust(width), value))


# --- environment -----------------------------------------------------------------------------------


def check_environment(session):
    print("環境檢查")
    rows = session.env.report()
    _table(rows)
    return all(ok for name, _, ok in rows if name in ("Python", "Workspace", "Local disk"))


def setup(session):
    """Environment check, then ComfyUI and its packages: extracted from the workspace when they are already
    there, installed (and archived into the workspace) the first time."""
    env = session.env
    env.check_workspace()
    check_environment(session)
    env.timed("comfyui", env.ensure_comfyui, session.worker.ctx.registry.custom_nodes())
    env.timed("deps", env.ensure_deps)
    env.save_environment()
    print("ComfyUI 與套件")
    words = {"installed": "首次安裝（已存進工作區）", "extracted": "從工作區解壓", "present": "本次 session 已就緒",
             "migrated": "環境版本改變，重新安裝", "reinstalled": "重新安裝"}  # fmt: skip
    _table([(key, words.get(value, value), True) for key, value in env.actions.items() if not key.startswith("model:")])
    return dict(env.actions)


def start_comfyui(session):
    """Start ComfyUI (setting it up first if needed) and show that its API answers."""
    session.worker.boot()
    stats = session.env.http.system_stats().get("system", {})
    print("ComfyUI 已啟動")
    _table(
        [
            ("ComfyUI", stats.get("comfyui_version", "?"), True),
            ("Python", str(stats.get("python_version", "?")).split()[0], True),
            ("API", session.env.http.base_url, True),
        ]
    )
    return stats


def gpu_report(session):
    if shutil.which("nvidia-smi"):
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout)
    else:
        print("這個 runtime 沒有 GPU（nvidia-smi 不存在）。到「執行階段 > 變更執行階段類型」選 GPU。")
    return session.env.report()


def check_models(session, workflow=None):
    """Which model files each workflow needs and whether a verified copy is already in the workspace."""
    registry = session.worker.ctx.registry
    all_ok = True
    for name in [workflow] if workflow else registry.names():
        wf = registry.get(name)
        print("%s（%s）" % (name, wf.get("title", "")))
        if not wf.models:
            print("  不需要模型")
            continue
        rows = []
        for spec, ok in session.env.model_status(wf.models):
            all_ok = all_ok and ok
            note = "已在工作區" if ok else "尚未下載（第一次執行時會下載並存進工作區）"
            rows.append((spec["name"], "%.2f GiB  %s/  %s" % (spec["size"] / 2**30, spec["store"], note), ok))
        _table(rows)
    return all_ok


def check_workflows(session):
    """Every enabled workflow against the running ComfyUI: are its nodes installed, with these inputs?"""
    session.worker.boot()
    registry = session.worker.ctx.registry
    all_ok = True
    rows = []
    for name in registry.names():
        wf = registry.get(name)
        graph = wf.template()
        problems = session.env.node_check(graph)
        all_ok = all_ok and not problems
        rows.append((name, "節點齊全" if not problems else "; ".join(problems), not problems))
    for name, exc in registry.problems.items():
        all_ok = False
        rows.append((name, "registry 設定有誤：%s" % exc, False))
    print("Workflow 檢查")
    _table(rows)
    return all_ok


# --- jobs --------------------------------------------------------------------------------------------


def _clean_form(workflow, values):
    """A form is shared by every workflow of its kind, so fields the chosen workflow does not declare are left
    out, and so are the values that mean "not set": empty text, and 0 where the parameter's minimum is above 0."""
    cleaned = {}
    for key, value in (values or {}).items():
        spec = workflow.parameters.get(key)
        if spec is None or value is None or (isinstance(value, str) and not value.strip()):
            continue
        if value == 0 and not isinstance(value, bool) and spec.get("min", 0) > 0:
            continue
        cleaned[key] = value
    return cleaned


def input_ref(source, job_id="", path="", attested=False):
    """A file input from the form's fields: where the file comes from."""
    if source == "job":
        return {"source": "job", "job_id": job_id.strip()}
    if source == "file":
        return {"source": "file", "path": path.strip(), "attested": bool(attested)}
    return {"source": source}


def comfy_log(session, chars=4000):
    try:
        print(session.env.comfy_log.read_text(encoding="utf-8", errors="replace")[-chars:])
    except OSError:
        print("還沒有 ComfyUI log（ComfyUI 尚未在這個 session 啟動）。")


def _show(storage, output):
    try:
        from IPython.display import Image, Video, display
    except ImportError:
        return
    path = storage.local_path(output["path"])
    if path is None or not path.is_file():
        return
    if output["kind"] == "image":
        display(Image(filename=str(path)))
    else:
        display(Video(str(path), embed=True))


def generate(session, workflow, simple=None, advanced=None, inputs=None, extra=None, show=True):
    """Create a job from the form and run it here. `advanced` and `extra` (a JSON object of further
    parameters) are None unless the Advanced switch is on."""
    ctx = session.worker.ctx
    try:
        wf = ctx.registry.get(workflow)
        values = _clean_form(wf, simple)
        values.update(_clean_form(wf, advanced))
        if extra and extra.strip():
            more = json.loads(extra)
            if not isinstance(more, dict):
                raise WorkflowError("INVALID_INPUT", '「其他參數」必須是 JSON 物件，例如 {"steps": 12}')
            values.update(more)
        if values.get("width") and values.get("height"):
            values.pop("aspect", None)  # a custom size replaces the aspect preset
        values.update({k: v for k, v in (inputs or {}).items() if k in wf.inputs})
        created = tools.create_job(workflow, values, source="notebook", context=ctx)
    except ValueError as exc:
        print("%s 「其他參數」不是合法的 JSON：%s" % (BAD, exc))
        return None
    except WorkflowError as exc:
        print("%s 沒有建立 job：[%s] %s" % (BAD, exc.code, exc.message))
        if exc.hint:
            print("  %s" % exc.hint)
        return None
    job_id = created["job_id"]
    print(
        "Job ID：%s%s"
        % (
            job_id,
            "（另外建立了 %s）" % ", ".join(j for j in created["jobs"] if j != job_id)
            if len(created["jobs"]) > 1
            else "",
        )
    )
    session.worker.drain(until=job_id)
    return report(session, job_id, show=show)


def report(session, job_id, show=True):
    job = session.worker.ctx.store.get(job_id)
    rows = [("Job ID", job_id, True), ("Workflow", job["workflow"], True)]
    status = job["status"]
    rows.append(("Status", status, status == jobs.COMPLETED))
    if status == jobs.COMPLETED:
        out = job["output"]
        rows.append(("Generation Time", "%s 秒" % job.get("generation_seconds"), True))
        rows.append(("Output Path", "AI-Workflow/%s" % out["path"], True))
        size = "%sx%s" % (out.get("width"), out.get("height"))
        if out.get("duration"):
            size += "，%.2f 秒，%s fps" % (out["duration"], out.get("fps"))
        rows.append(("Output", "%s，%.1f KB" % (size, out["bytes"] / 1024), True))
        if job.get("vram_peak_mib"):
            rows.append(("VRAM 峰值", "%s MiB（%s）" % (job["vram_peak_mib"], job.get("gpu")), True))
        values = dict(job["parameters"])
        if "seed" in values:
            rows.append(("Seed", values["seed"], True))
    elif status in (jobs.FAILED, jobs.CANCELLED):
        error = job.get("error") or {}
        rows.append(("Error", "[%s] %s" % (error.get("code"), error.get("message")), False))
    else:
        waiting = session.worker.state.data.get("waiting", {}).get(job_id)
        rows.append(("等待中", waiting or "尚未開始", False))
    _table(rows)
    if show and status == jobs.COMPLETED:
        _show(session.storage, job["output"])
    return tools.get_result(job_id, context=session.worker.ctx)


def api_test(session):
    """ComfyUI API round trip with the model-free test workflow: /prompt, /history, /view, output stored."""
    print("ComfyUI API 測試（test-generation，不需要模型）")
    return generate(session, "test-generation", {})


def list_jobs(session, status=None, limit=20):
    found = tools.list_jobs(status, limit, context=session.worker.ctx)
    print("Jobs：%s" % (found["counts"] or "沒有 job"))
    _table(
        [
            (j["job_id"], "%s  %s  %s" % (j["status"], j["workflow"], j["created_at"]), j["status"] != jobs.FAILED)
            for j in found["jobs"]
        ]
    )
    return found


def describe(session, workflow):
    """The parameters a workflow takes (what the Advanced form's 'extra' field may contain)."""
    wf = session.worker.ctx.registry.get(workflow)
    info = wf.describe()
    print("%s：%s" % (workflow, info["title"]))
    _table(
        [
            (p["name"], "%s  預設 %s  %s" % (p["type"], p.get("default", "-"), p.get("label", "")), True)
            for p in info["parameters"]
        ]
    )
    return info


def serve(session, idle_minutes=10, release_runtime=True):
    """Wait for jobs made through the tool interface and run them. After idle_minutes with an empty queue the
    worker stops; with release_runtime the runtime is given back so it stops using compute units."""
    print(
        "等待 job 中（jobs/pending）。佇列空了 %s 分鐘後%s。"
        % (idle_minutes, "會釋放 runtime" if release_runtime else "停止等待（runtime 不會自動釋放，仍在計費）")
    )
    reason = session.worker.serve(float(idle_minutes) * 60)
    summary = session.worker.shutdown(reason, release=release_runtime)
    print("worker 已停止：%s，共完成 %d 個 job。" % (reason, summary["jobs"]))
    if not release_runtime:
        print("提醒：runtime 還開著。不用時到「執行階段 > 中斷連線並刪除執行階段」。")
    return summary


def stop(session, release_runtime=False):
    return session.worker.shutdown("manual", release=release_runtime)


__all__ = [
    "connect", "mount_drive", "check_environment", "setup", "start_comfyui", "gpu_report", "check_models",
    "check_workflows", "generate", "report", "api_test", "list_jobs", "describe", "serve", "stop", "input_ref",
    "comfy_log",
]  # fmt: skip
