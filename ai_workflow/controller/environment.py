"""The compute runtime's environment: check it, and make ComfyUI, its Python packages and the models
available - from the workspace when they are already there, from the pinned public source when they are not.

Nothing expensive is repeated across sessions. The workspace keeps:

    cache/comfyui-<commit>.tar.gz        the pinned ComfyUI source
    cache/node-<name>-<commit>.tar.gz    each pinned custom node
    cache/pydeps-<fingerprint>.tar       the pip packages the runtime image lacks (a PYTHONUSERBASE tree)
    <store>/<file>                       model files (models/<model>/..., loras/)
    models/manifest.json                 what each model file is: source, revision, size, sha256
    configs/environment.json             what was installed, with which versions

A new session extracts the archives and copies the models to the runtime's local disk. The runtime's own disk
(local_root) is a session-local cache and nothing else: everything under it may vanish.

Archives rather than folders, because a mounted Drive is slow with many small files. A model's manifest entry
is written only after the file is completely in the workspace, so a half-copied file is never trusted.

Every `actions[...]` value says what happened this session: installed / downloaded (first time), extracted /
staged (reused from the workspace), present (already on local disk). The session-restart test reads them.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from . import WorkflowError
from .comfy import Comfy, ComfyProcess, check_against_object_info, combo_options
from .validate import sha256_file

MODEL_FOLDERS = (
    "checkpoints", "diffusion_models", "text_encoders", "vae", "loras", "clip_vision", "controlnet",
    "upscale_models", "embeddings",
)  # fmt: skip
MANIFEST_FILE = "models/manifest.json"
ENVIRONMENT_FILE = "configs/environment.json"


def copy_large(src, dst, block=64 * 2**20):
    """Sequential large-block copy to a .partial name, then rename: a half-copied file never carries the final
    name, so a size check can trust it."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".partial")
    with open(src, "rb") as fi, open(partial, "wb") as fo:
        shutil.copyfileobj(fi, fo, block)
    os.replace(partial, dst)


def extract_tarball(archive, dest, strip=True):
    """Extract into dest. With strip, drop the single top-level directory GitHub tarballs carry."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".extracting")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="data")
    src = tmp
    if strip:
        entries = list(tmp.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            src = entries[0]
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dest)
    shutil.rmtree(tmp, ignore_errors=True)


def make_tarball(src_dir, archive):
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".partial")
    with tarfile.open(partial, "w") as tar:
        for child in sorted(Path(src_dir).iterdir()):
            tar.add(str(child), arcname=child.name)
    os.replace(partial, archive)


def url_download(url, dest, timeout=600):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(partial, "wb") as f:
        shutil.copyfileobj(resp, f, 8 * 2**20)
    os.replace(partial, dest)
    return dest


def hf_download(repo, filename, revision, local_dir):
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    # token=False: the pinned repositories are public, and without it every call first waits on a token lookup.
    return hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=str(local_dir), token=False)


def run_cmd(cmd, env=None, timeout=3600):
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def probe_gpu():
    try:
        import torch
    except ImportError:
        return {"available": False, "torch": None}
    if not torch.cuda.is_available():
        return {"available": False, "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "vram_gib": round(props.total_memory / 2**30, 1),
        "capability": "%d.%d" % (props.major, props.minor),
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


class Environment:
    def __init__(
        self,
        storage,
        config,
        *,
        local_root=None,
        clock=time.time,
        sleep=time.sleep,
        echo=print,
        gpu_probe=probe_gpu,
        hf_download_fn=hf_download,
        url_download_fn=url_download,
        runner=run_cmd,
        comfy_factory=ComfyProcess,
        comfy_url=None,
        base_freeze=None,
        copy_fn=copy_large,
    ):
        self.storage = storage
        self.config = config
        self.clock = clock
        self.sleep = sleep
        self.echo = echo
        self.gpu_probe = gpu_probe
        self.hf_download = hf_download_fn
        self.url_download = url_download_fn
        self.runner = runner
        self.comfy_factory = comfy_factory
        self.base_freeze = base_freeze
        self.copy_to_workspace = copy_fn
        workspace = storage.local_path("cache")
        if workspace is None:
            raise WorkflowError("STORAGE_UNSUPPORTED", "the runtime needs a storage backend with real file paths")
        self.workspace = workspace.parent
        self.local = Path(local_root or config.get("runtime", {}).get("local_root", "/content/aiwf"))
        self.comfy_dir = self.local / "ComfyUI"
        self.pyuser = self.local / "pyuser"
        self.local_models = self.local / "models"
        self.stage = self.local / "stage"
        self.comfy_out = self.local / "comfy_out"
        self.comfy_log = self.local / "comfyui.log"
        self.extra_paths = self.local / "extra_model_paths.yaml"
        self.cache = self.workspace / "cache"
        port = config.get("comfyui", {}).get("port", 8188)
        self.http = Comfy(comfy_url or "http://127.0.0.1:%d" % port)
        self.gpu = {}
        self.actions = {}
        self.phase_seconds = {}
        self.model_seconds = {}
        self.process = None
        self.process_args = None
        self.sync_threads = []
        self.deps_fingerprint = None
        self.deps_parts = None
        self.deps_installed_this_session = False

    def log(self, message):
        self.echo("[%s] %s" % (time.strftime("%H:%M:%S"), message))

    def act(self, key, value):
        """Record what happened to a component this session. "present" (already on the runtime's disk) never
        replaces an earlier answer, so the session record still says installed / extracted / downloaded / staged."""
        if value != "present" or key not in self.actions:
            self.actions[key] = value

    def timed(self, name, fn, *args):
        t = self.clock()
        try:
            return fn(*args)
        finally:
            self.phase_seconds[name] = round(self.phase_seconds.get(name, 0) + self.clock() - t, 1)

    # -- checks -----------------------------------------------------------------------------------

    def report(self):
        """The environment check: Python, CUDA, GPU, VRAM, the workspace and the local disk. Never raises;
        each row says ok or what is wrong."""
        self.gpu = self.gpu_probe()
        rows = [("Python", platform.python_version(), True)]
        rows.append(("PyTorch", self.gpu.get("torch") or "not installed", bool(self.gpu.get("torch"))))
        rows.append(("CUDA", self.gpu.get("cuda") or "not available", bool(self.gpu.get("available"))))
        rows.append(("GPU", self.gpu.get("name") or "none (CPU runtime)", bool(self.gpu.get("available"))))
        vram = self.gpu.get("vram_gib")
        rows.append(("VRAM", "%.1f GiB" % vram if vram else "-", bool(vram)))
        writable = False
        try:
            self.storage.write_text("logs/.write_probe", str(self.clock()))
            writable = bool(self.storage.read_text("logs/.write_probe"))
        except OSError:
            pass
        rows.append(("Workspace", str(self.workspace), writable))
        try:
            self.local.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(str(self.local)).free / 2**30
            need = self.config.get("runtime", {}).get("min_free_disk_gib", 10)
            rows.append(("Local disk", "%.1f GiB free" % free, free >= need))
        except OSError as exc:
            rows.append(("Local disk", str(exc), False))
        return rows

    def check_workspace(self):
        if not self.workspace.is_dir():
            raise WorkflowError("WORKSPACE_NOT_FOUND", "%s is not there - is Google Drive mounted?" % self.workspace)
        self.storage.ensure_layout()
        for d in (self.stage, self.comfy_out, self.local_models):
            d.mkdir(parents=True, exist_ok=True)
        for folder in MODEL_FOLDERS:
            (self.local_models / folder).mkdir(parents=True, exist_ok=True)

    # -- ComfyUI source and custom nodes ---------------------------------------------------------------

    def ensure_comfyui(self, custom_nodes=()):
        c = self.config["comfyui"]
        wanted = {"comfyui": c["commit"]}
        for name in custom_nodes:
            if name not in self.config.get("custom_nodes", {}):
                raise WorkflowError(
                    "INVALID_CONFIG", "custom node %r has no pinned entry in configs/platform.json" % name
                )
            wanted["custom_node:" + name] = self.config["custom_nodes"][name]["commit"]
        marker = self.comfy_dir / ".aiwf_installed.json"
        try:
            installed = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            installed = {}
        if installed == wanted:
            for key in wanted:
                self.act(key, "present")
            return
        if installed.get("comfyui") != c["commit"]:
            tarball = self.cache / ("comfyui-%s.tar.gz" % c["commit"])
            self.act("comfyui", self._fetch_archive(tarball, c["tarball_url"], "ComfyUI %s" % c["ref"]))
            extract_tarball(tarball, self.comfy_dir)
        else:
            self.act("comfyui", "present")
        for name in custom_nodes:
            node = self.config["custom_nodes"][name]
            tarball = self.cache / ("node-%s-%s.tar.gz" % (name, node["commit"]))
            self.act("custom_node:" + name, self._fetch_archive(tarball, node["tarball_url"], name))
            extract_tarball(tarball, self.comfy_dir / "custom_nodes" / name)
        marker.write_text(json.dumps(wanted), encoding="utf-8")

    def _fetch_archive(self, tarball, url, what):
        if tarball.is_file() and tarball.stat().st_size > 0:
            return "extracted"
        self.log("%s is not in the workspace yet: downloading the source archive once" % what)
        staged = self.url_download(url, self.stage / tarball.name)
        copy_large(staged, tarball)
        return "installed"

    # -- Python packages -------------------------------------------------------------------------------

    def requirement_files(self):
        files = [self.comfy_dir / "requirements.txt"]
        nodes = self.comfy_dir / "custom_nodes"
        if nodes.is_dir():
            for node in sorted(nodes.iterdir()):
                if (node / "requirements.txt").is_file():
                    files.append(node / "requirements.txt")
        return [f for f in files if f.is_file()]

    @staticmethod
    def kernel_env():
        env = dict(os.environ)
        env.pop("PYTHONUSERBASE", None)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return env

    def runtime_fingerprint(self):
        """Changes whenever the archived packages could stop matching: the runtime image's own packages, the
        Python / torch / CUDA versions, the ComfyUI commit, or any requirements file."""
        if self.base_freeze is None:
            code, out = self.runner([sys.executable, "-m", "pip", "freeze"], env=self.kernel_env())
            self.base_freeze = out if code == 0 else ""
        reqs = "\n".join(p.read_text(encoding="utf-8") for p in self.requirement_files())
        parts = {
            "python": platform.python_version(),
            "torch": self.gpu.get("torch"),
            "cuda": self.gpu.get("cuda"),
            "image": os.environ.get("COLAB_RELEASE_TAG"),
            "base_freeze": hashlib.sha256(self.base_freeze.encode("utf-8")).hexdigest(),
            "comfyui_commit": self.config["comfyui"]["commit"],
            "requirements": hashlib.sha256(reqs.encode("utf-8")).hexdigest(),
        }
        digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:16], parts

    @staticmethod
    def base_pins():
        """Pin the GPU stack to what the runtime image ships, so no requirement can archive a second torch."""
        from importlib import metadata

        pins = []
        for name in ("torch", "torchvision", "torchaudio", "numpy"):
            try:
                pins.append("%s==%s" % (name, metadata.version(name)))
            except metadata.PackageNotFoundError:
                pass
        return pins

    def ensure_deps(self, force=False):
        fingerprint, parts = self.runtime_fingerprint()
        self.deps_fingerprint, self.deps_parts = fingerprint, parts
        archive = self.cache / ("pydeps-%s.tar" % fingerprint)
        done = self.pyuser / ".aiwf_fingerprint"
        if not force and done.is_file() and done.read_text(encoding="utf-8").strip() == fingerprint:
            self.act("deps", "present")
            return
        if not force and archive.is_file() and archive.stat().st_size > 0:
            shutil.rmtree(self.pyuser, ignore_errors=True)
            extract_tarball(archive, self.pyuser, strip=False)
            done.write_text(fingerprint, encoding="utf-8")
            self.act("deps", "extracted")
            return
        previous = (self.storage.read_json(ENVIRONMENT_FILE, {}) or {}).get("deps_fingerprint")
        self.log("installing ComfyUI's Python packages once for fingerprint %s (was %s)" % (fingerprint, previous))
        shutil.rmtree(self.pyuser, ignore_errors=True)
        self.pyuser.mkdir(parents=True)
        constraints = self.stage / "constraints.txt"
        constraints.parent.mkdir(parents=True, exist_ok=True)
        constraints.write_text("\n".join(self.base_pins()) + "\n", encoding="utf-8")
        env = self.kernel_env()
        env["PYTHONUSERBASE"] = str(self.pyuser)
        cmd = [sys.executable, "-m", "pip", "install", "--user", "--no-warn-script-location", "-c", str(constraints)]
        for req in self.requirement_files():
            cmd += ["-r", str(req)]
        code, out = self.runner(cmd, env=env)
        if code != 0:
            raise WorkflowError("DEPS_INSTALL_FAILED", "pip install failed (exit %s): %s" % (code, out[-3000:]))
        done.write_text(fingerprint, encoding="utf-8")
        self.deps_installed_this_session = True
        staged = self.stage / archive.name
        make_tarball(self.pyuser, staged)
        copy_large(staged, archive)
        staged.unlink()
        changed = previous and previous != fingerprint
        self.act("deps", "reinstalled" if force else ("migrated" if changed else "installed"))

    # -- models ----------------------------------------------------------------------------------------

    def model_status(self, specs):
        """For each model file: is a complete, verified copy in the workspace? (manifest + size)."""
        manifest = self.storage.read_json(MANIFEST_FILE, {}) or {}
        rows = []
        for spec in specs:
            key = "%s/%s" % (spec["store"], spec["name"])
            path = self.workspace / spec["store"] / spec["name"]
            entry = manifest.get(key) or {}
            ok = (
                path.is_file()
                and path.stat().st_size == spec["size"]
                and entry.get("revision") == spec["revision"]
                and entry.get("sha256") == spec["sha256"]
            )
            rows.append((spec, ok))
        return rows

    def ensure_models(self, specs):
        """Stage the model files on the runtime's local disk: copied from the workspace when a verified copy is
        there, downloaded from the pinned source (and then copied into the workspace in the background) when
        not. Only the files that are missing or changed are fetched.

        The workspace copy is the one that counts; the copy on the runtime's disk is working material. So a
        file that is on this disk but neither in the workspace nor on its way there (an earlier copy failed)
        is sent again, not left for the next session to download a second time."""
        missing, present = [], []
        for spec, ok in self.model_status(specs):
            (present if ok else missing).append(spec)
        todo = [s for s in missing if not self._local_ok(s)]
        if todo:
            self.check_space(sum(s["size"] for s in todo))
            self._start_sync(self._download(todo))
        in_flight = {item["name"] for item in self.sync_status()["pending"]}
        retry = [s for s in missing if s not in todo and s["name"] not in in_flight]
        if retry:
            self._start_sync([(s, self._local_path(s)) for s in retry])
        for spec in missing:
            self.act("model:" + spec["name"], "downloaded" if spec in todo else "present")
        for spec in present:
            if self._local_ok(spec):
                self.act("model:" + spec["name"], "present")
                continue
            t = self.clock()
            copy_large(self.workspace / spec["store"] / spec["name"], self._local_path(spec))
            self.model_seconds[spec["name"]] = {"staged": round(self.clock() - t, 1)}
            self.log("staged %s from the workspace in %.1f s" % (spec["name"], self.clock() - t))
            self.act("model:" + spec["name"], "staged")

    def _local_path(self, spec):
        return self.local_models / spec["folder"] / spec["name"]

    def _local_ok(self, spec):
        local = self._local_path(spec)
        return local.is_file() and local.stat().st_size == spec["size"]

    def check_space(self, need_bytes):
        free = shutil.disk_usage(str(self.local)).free
        # The local copy, plus the Drive client's own local cache of what is written to the workspace, plus headroom.
        need = 2 * need_bytes + 10 * 2**30
        if free < need:
            raise WorkflowError(
                "DISK_FULL", "%.1f GiB free on the runtime, the models need %.1f GiB" % (free / 2**30, need / 2**30)
            )

    def _download_one(self, spec):
        """Download one file from its pinned source to the runtime's disk and verify it. Returns its path."""
        t = self.clock()
        got = Path(self.hf_download(spec["repo"], spec["path_in_repo"], spec["revision"], self.stage))
        seconds = self.clock() - t
        digest = sha256_file(got)
        if digest != spec["sha256"] or got.stat().st_size != spec["size"]:
            raise WorkflowError("MODEL_CHECKSUM", "%s: sha256 %s, expected %s" % (spec["name"], digest, spec["sha256"]))
        local = self._local_path(spec)
        local.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, local)
        self.model_seconds[spec["name"]] = {"downloaded": round(seconds, 1)}
        self.log("downloaded %s: %.2f GiB in %.1f s, sha256 ok" % (spec["name"], spec["size"] / 2**30, seconds))
        return local

    def _download(self, specs):
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
            return list(zip(specs, pool.map(self._download_one, specs)))

    def _start_sync(self, results):
        sync = ModelSync(self, results)
        sync.start()
        self.sync_threads.append(sync)

    def save_to_workspace(self, spec, local):
        """Copy one verified model file into the workspace, then record it in the manifest - in that order, so
        a manifest entry always means a complete file."""
        t = self.clock()
        self.copy_to_workspace(local, self.workspace / spec["store"] / spec["name"])
        with ModelSync.lock:
            manifest = self.storage.read_json(MANIFEST_FILE, {}) or {}
            manifest["%s/%s" % (spec["store"], spec["name"])] = {
                "name": spec["name"],
                "revision": spec["revision"],
                "source": "https://huggingface.co/%s/resolve/%s/%s"
                % (spec["repo"], spec["revision"], spec["path_in_repo"]),
                "size": spec["size"],
                "sha256": spec["sha256"],
                "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            self.storage.write_json(MANIFEST_FILE, manifest)
        seconds = round(self.clock() - t, 1)
        self.model_seconds.setdefault(spec["name"], {})["copied_to_workspace"] = seconds
        self.log("copied %s into the workspace in %s s" % (spec["name"], seconds))
        return seconds

    def sync_status(self):
        """Where the freshly downloaded models are on their way into the workspace: still being copied, saved,
        or failed. `pending` is what would be lost if the runtime were deleted right now."""
        pending, saved, failed = [], {}, {}
        for thread in self.sync_threads:
            for spec, _ in thread.results:
                result = thread.report.get(spec["name"])
                if result is None:
                    pending.append({"name": spec["name"], "bytes": spec["size"]})
                elif result.startswith("failed"):
                    failed[spec["name"]] = result
                    saved.pop(spec["name"], None)
                else:
                    saved[spec["name"]] = result
                    failed.pop(spec["name"], None)
        return {"pending": pending, "pending_bytes": sum(p["bytes"] for p in pending), "saved": saved, "failed": failed}

    def wait_for_sync(self, progress=None, every_seconds=30):
        """Block until every freshly downloaded model is in the workspace (or its copy has failed). Called
        before the mounted folder is flushed and the runtime released. `progress` is called with sync_status()
        every `every_seconds` while it waits - tens of GB take a while and silence looks like a hang."""
        for thread in self.sync_threads:
            while thread.is_alive():
                thread.join(every_seconds)
                if progress is not None and thread.is_alive():
                    progress(self.sync_status())
        status = self.sync_status()
        return dict(status["saved"], **status["failed"])

    def prefetch_models(self, specs):
        """Put model files into the workspace without generating anything: one file at a time - download,
        verify, copy into the workspace, record, delete the runtime's copy - so the runtime needs room for the
        largest single file, not for the whole set, and no GPU. Files already verified in the workspace are
        skipped. Returns {file name: what happened}."""
        done = {}
        for spec, ok in self.model_status(specs):
            name = spec["name"]
            if ok:
                done[name] = "already in the workspace"
                self.act("model:" + name, "in workspace")
                continue
            had_local = self._local_ok(spec)
            if not had_local:
                self.check_space(spec["size"])
                self._download_one(spec)
            self.save_to_workspace(spec, self._local_path(spec))
            if not had_local:
                self._local_path(spec).unlink()
            done[name] = "saved to the workspace" if had_local else "downloaded and saved to the workspace"
            self.act("model:" + name, "prefetched")
        return done

    # -- the ComfyUI process ---------------------------------------------------------------------------

    def write_extra_model_paths(self):
        lines = ["aiwf:", "  base_path: %s" % self.local_models]
        lines += ["  %s: %s" % (folder, folder) for folder in MODEL_FOLDERS]
        self.extra_paths.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def comfy_args(self, workflow_args=()):
        args = list(self.config["comfyui"].get("extra_args", []))
        if not self.gpu.get("available"):
            args.append("--cpu")
        for rule in workflow_args:
            if (self.gpu.get("vram_gib") or 0) < rule.get("when_vram_below_gib", 0):
                args += [a for a in rule["args"] if a not in args]
        return args

    def ensure_comfy_running(self, workflow_args=()):
        """Start ComfyUI, or restart it when it has died or the job needs a launch argument it was not started
        with. Arguments are only ever added: a restart reloads every model, so a queue that alternates between
        two workflows must not restart ComfyUI between each pair of jobs."""
        args = self.comfy_args(workflow_args)
        alive = self.process is not None and self.process.running() and self.http.alive()
        if alive and set(args) <= set(self.process_args):
            return "running"
        if self.process_args:
            args += [a for a in self.process_args if a not in args]
        if self.process is not None:
            self.process.stop()
        self.write_extra_model_paths()
        self.process = self.comfy_factory(
            self.comfy_dir,
            port=self.config["comfyui"].get("port", 8188),
            extra_paths_yaml=self.extra_paths,
            output_dir=self.comfy_out,
            log_path=self.comfy_log,
            pyuser=self.pyuser,
            extra_args=args,
        )
        self.process_args = args
        try:
            self._wait_for_comfy()
        except WorkflowError:
            tail = self.process.log_tail()
            if self.deps_installed_this_session or not any(s in tail for s in ("ModuleNotFoundError", "ImportError")):
                raise
            # An archive with the same fingerprint that does not import: rebuild it once.
            self.log("ComfyUI failed to import with the archived packages; reinstalling them once")
            self.ensure_deps(force=True)
            self._wait_for_comfy()
        return "started"

    def _wait_for_comfy(self):
        self.process.start()
        deadline = self.clock() + self.config["comfyui"].get("startup_timeout_seconds", 600)
        while self.clock() < deadline:
            if self.http.alive():
                return
            if not self.process.running():
                raise WorkflowError(
                    "COMFYUI_START_FAILED", "ComfyUI exited during startup:\n" + self.process.log_tail()
                )
            self.sleep(1)
        self.process.stop()
        raise WorkflowError("COMFYUI_START_FAILED", "ComfyUI did not answer within the startup timeout")

    def stop_comfy(self):
        if self.process is not None:
            self.process.stop()

    def node_check(self, graph, specs=()):
        """The graph's nodes exist with these inputs, and each loader lists its model file."""
        info = self.http.object_info()  # asked fresh each time: a file staged a moment ago must be listed
        problems = check_against_object_info(graph, info)
        for spec in specs:
            loader = spec.get("loader")
            options = combo_options(info, loader[0], loader[1]) if loader else None
            if options is not None and spec["name"] not in options:
                problems.append("%s does not list %s" % (loader[0], spec["name"]))
        return problems

    def save_environment(self):
        record = self.storage.read_json(ENVIRONMENT_FILE, {}) or {}
        record.update(
            comfyui_ref=self.config["comfyui"]["ref"],
            comfyui_commit=self.config["comfyui"]["commit"],
            custom_nodes={k: v["commit"] for k, v in self.config.get("custom_nodes", {}).items()},
            python_version=platform.python_version(),
            torch_version=self.gpu.get("torch"),
            cuda_version=self.gpu.get("cuda"),
            runtime_image=os.environ.get("COLAB_RELEASE_TAG"),
            deps_fingerprint=self.deps_fingerprint,
            deps_parts=self.deps_parts,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        record.setdefault("installed_at", record["updated_at"])
        history = list(record.get("history") or [])
        history.append({"at": record["updated_at"], "actions": dict(self.actions)})
        record["history"] = history[-30:]
        self.storage.write_json(ENVIRONMENT_FILE, record)


class ModelSync(threading.Thread):
    """Copies freshly downloaded models into the workspace in the background while the first job already runs
    from the runtime's copy. `report` gains an entry per file when its copy has ended, one way or the other."""

    lock = threading.Lock()

    def __init__(self, env, results):
        super().__init__(name="aiwf-model-sync", daemon=True)
        self.env = env
        self.results = list(results)
        self.report = {}

    def run(self):
        for spec, local in self.results:
            try:
                seconds = self.env.save_to_workspace(spec, local)
            except OSError as exc:
                self.report[spec["name"]] = "failed: %s" % exc
                self.env.log("could not copy %s into the workspace: %s" % (spec["name"], exc))
                continue
            self.report[spec["name"]] = "copied in %s s" % seconds
