"""The notebooks are entry points and nothing else, the deploy script only touches what it should, and the
platform's code names no assistant and needs no credential.
"""

import ast
import json
import os
import re
import subprocess
import sys

import pytest
from aiwf_fakes import PLATFORM_DIR, load_script, make_workspace
from controller import notebook as nb
from controller import tools

build = load_script("build_notebooks")
deploy = load_script("deploy")
NOTEBOOKS = ("00_setup", "01_comfyui", "02_image_generation", "03_video_generation", "99_manual_debug")
# Nothing in a notebook may do more than this: mount, put the workspace on sys.path, and call controller.notebook.
ALLOWED_IMPORTS = {"from google.colab import drive", "import sys", "from controller import notebook as nb"}


def cells(name):
    data = json.loads((PLATFORM_DIR / "notebooks" / (name + ".ipynb")).read_text(encoding="utf-8"))
    return data, ["".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code"]


def test_the_committed_notebooks_are_what_the_builder_writes():
    rendered = build.render_all()
    assert sorted(rendered) == sorted(n + ".ipynb" for n in NOTEBOOKS)
    for name, text in rendered.items():
        on_disk = (PLATFORM_DIR / "notebooks" / name).read_bytes()
        assert on_disk == text.encode("utf-8"), "%s is stale: run ai_workflow/scripts/build_notebooks.py" % name
        assert b"\r\n" not in on_disk


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_a_notebook_holds_forms_and_calls_and_no_logic(name):
    data, code = cells(name)
    assert data["nbformat"] == 4 and all(c.get("outputs", []) == [] for c in data["cells"])
    assert code[0].count("drive.mount") == 1 and "nb.connect(ROOT, mount=False)" in code[0]
    for source in code:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            assert not isinstance(
                node, (ast.FunctionDef, ast.ClassDef, ast.For, ast.While, ast.Try, ast.With, ast.Lambda)
            )
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert ast.unparse(node) in ALLOWED_IMPORTS
            if isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                assert target.startswith("nb.") or target in ("dict", "drive.mount", "sys.path.insert"), target
        statements = [n for n in tree.body if not isinstance(n, (ast.Assign, ast.Import, ast.ImportFrom))]
        assert len(statements) <= 3, "a cell is a form plus a call, not a program"
    called = set(re.findall(r"\bnb\.(\w+)", "\n".join(code)))
    assert called <= set(nb.__all__), called - set(nb.__all__)


def test_the_generation_forms_offer_what_the_registry_declares():
    registry = json.loads((PLATFORM_DIR / "workflows" / "registry.json").read_text(encoding="utf-8"))["workflows"]
    _, image = cells("02_image_generation")
    form = image[2]
    assert 'workflow = "z-image-basic"  #@param ["z-image-basic", "test-generation"] {allow-input: true}' in form
    for aspect in registry["z-image-basic"]["parameters"]["aspect"]["sizes"]:
        assert '"%s"' % aspect in form
    assert "cfg = 2.0" in form and "steps = 8" in form and "seed = -1" in form and "use_advanced = False" in form
    assert (
        "result = nb.generate(session, workflow, simple, advanced, extra=extra_json if use_advanced else None)" in form
    )
    _, video = cells("03_video_generation")
    form = video[2]
    assert 'first_frame_source = "generate"  #@param ["generate", "job", "file"]' in form
    assert "first_frame_attested = False" in form and "不是真人照片" in form
    assert "duration = 5" in form and 'inputs={"first_frame": first_frame}' in form
    data, _ = cells("03_video_generation")
    assert data["metadata"]["colab"]["gpuType"] == "A100" and data["metadata"]["accelerator"] == "GPU"


def test_run_all_with_the_default_form_values_creates_valid_jobs(tmp_path):
    """Run All must work untouched: the forms' default values have to pass validation."""
    storage, _ = make_workspace(tmp_path)
    ctx = tools.open_context(storage=storage)
    for notebook, cell in (("02_image_generation", 2), ("03_video_generation", 2)):
        _, code = cells(notebook)
        created = []

        class Recorder:
            input_ref = staticmethod(nb.input_ref)

            @staticmethod
            def generate(session, workflow, simple, advanced, extra=None, inputs=None):
                wf = ctx.registry.get(workflow)
                values = nb._clean_form(wf, simple)
                values.update(inputs or {})
                created.append(tools.create_job(workflow, values, context=ctx))

        exec(code[cell], {"nb": Recorder, "session": None})  # noqa: S102 - our own generated cell
        assert created and created[0]["status"] == "pending"


# --- deploy -------------------------------------------------------------------------------------------


def test_deploy_creates_the_tree_and_is_idempotent(tmp_path):
    target = tmp_path / "AI-Workflow"
    planned = deploy.deploy(target, dry_run=True)
    assert planned and not target.exists()
    deploy.deploy(target)
    for rel in (
        "controller/tools.py",
        "scripts/aiwf.py",
        "notebooks/03_video_generation.ipynb",
        "configs/platform.json",
    ):
        assert (target / rel).is_file(), rel
    for rel in ("models", "loras", "inputs", "outputs/videos", "jobs/pending", "jobs/failed", "cache", "logs"):
        assert (target / rel).is_dir() and not any((target / rel).iterdir()), rel
    assert not (target / "configs" / "local.json").exists() and not list(target.rglob("__pycache__"))
    assert deploy.deploy(target) == []


def test_deploy_keeps_what_belongs_to_the_workspace(tmp_path):
    target = tmp_path / "AI-Workflow"
    deploy.deploy(target)
    (target / "jobs" / "pending" / "job-1.json").write_text("{}")
    (target / "models" / "big.safetensors").write_bytes(b"weights")
    (target / "workflows" / "image" / "mine.json").write_text("{}")
    (target / "controller" / "removed_module.py").write_text("x = 1\n")
    registry = json.loads((target / "workflows" / "registry.json").read_text(encoding="utf-8"))
    registry["workflows"]["mine"] = {"type": "image", "workflow_file": "image/mine.json"}
    registry["workflows"]["z-image-basic"]["title"] = "edited in the workspace"
    (target / "workflows" / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    actions = deploy.deploy(target)
    assert sorted(actions) == [("merge", "workflows/registry.json"), ("remove", "controller/removed_module.py")]
    merged = json.loads((target / "workflows" / "registry.json").read_text(encoding="utf-8"))["workflows"]
    assert (
        "mine" in merged and merged["z-image-basic"]["title"] == "Z-Image Turbo 文字生圖"
    )  # the repository's entry wins
    assert (target / "workflows" / "image" / "mine.json").is_file()
    assert (target / "jobs" / "pending" / "job-1.json").is_file() and (target / "models" / "big.safetensors").is_file()


def test_the_deployed_copy_runs_from_the_workspace_with_no_configuration(tmp_path, capsys):
    """The command line inside the workspace finds the workspace by itself - any machine that syncs the folder
    has the tool interface."""
    target = tmp_path / "AI-Workflow"
    deploy.deploy(target)
    env = {k: v for k, v in os.environ.items() if k != "AIWF_ROOT" and not k.upper().endswith("_API_KEY")}
    env.pop("PYTHONPATH", None)
    run = subprocess.run(
        [sys.executable, str(target / "scripts" / "aiwf.py"), "create-job", "test-generation", "--set", "width=128"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(tmp_path), env=env, timeout=120,
    )  # fmt: skip
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["ok"] and out["status"] == "pending"
    assert (target / "jobs" / "pending" / (out["job_id"] + ".json")).is_file()


# --- no assistant, no credential -------------------------------------------------------------------------

PROVIDER_WORDS = re.compile(r"anthropic|openai|gemini|claude|chatgpt|\bgpt\b|copilot|api[_-]?key", re.IGNORECASE)


def test_the_platform_code_names_no_assistant_and_no_credential():
    """Test C: swapping the assistant must not touch the core. The simplest proof is that the core, the scripts,
    the notebooks, the workflows and the configs do not know any assistant exists. (AGENTS.md and README.md
    are prose for people and assistants; the rule is about what runs.)"""
    offenders = []
    for folder in ("controller", "scripts", "notebooks", "workflows", "configs"):
        for path in sorted((PLATFORM_DIR / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".ipynb") and "__pycache__" not in path.parts:
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if PROVIDER_WORDS.search(line) and "no credential" not in line:
                        offenders.append("%s:%d: %s" % (path.relative_to(PLATFORM_DIR), number, line.strip()[:80]))
    assert offenders == []


def test_the_core_imports_only_the_standard_library():
    allowed = set(sys.stdlib_module_names) | {"controller"}
    optional = {"torch", "huggingface_hub", "PIL", "google", "IPython"}  # imported inside functions, on the runtime
    for path in sorted((PLATFORM_DIR / "controller").glob("*.py")) + sorted((PLATFORM_DIR / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module.split(".")[0]]
            assert set(names) <= allowed, "%s imports %s at module level" % (path.name, names)
        every = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
        every |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
        assert every <= allowed | optional, "%s: %s" % (path.name, every - allowed - optional)
