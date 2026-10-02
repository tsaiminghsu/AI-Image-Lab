"""Generate the five notebooks in notebooks/ from the registry.

    python ai_workflow/scripts/build_notebooks.py            (add --check to only verify they are up to date)

The notebooks are entry points and nothing else: every code cell is a form (Colab `#@param` fields) followed by
one call into controller.notebook. The form choices and defaults are read from workflows/registry.json, so the
notebooks are regenerated - never hand-edited - when a workflow changes; tests assert the committed files equal
this script's output. A workflow added later in the workspace still works without regenerating: the workflow
field accepts typed input, and the Advanced form has an "extra" field for any parameter.

Standard library only. The output is deterministic (no timestamps, no ids).
"""

import argparse
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = SOURCE / "notebooks"
ROOT_DEFAULT = "/content/drive/MyDrive/AI-Workflow"
SAMPLE_IMAGE_PROMPT = "A white ceramic mug on a plain white background, centered, soft studio light."
SAMPLE_VIDEO_PROMPT = (
    "A cup of hot coffee on a wooden table beside a window in soft morning light. Thin steam rises slowly from "
    "the cup. The camera remains static in a locked-off shot."
)
ATTESTATION_ZH = "這張圖是本專案生成的虛構成人角色，或畫面中沒有人物；**不是真人照片**。"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _source(lines)}


def code(title, *lines):
    head = '#@title %s { display-mode: "form" }' % title
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"cellView": "form"},
        "outputs": [],
        "source": _source((head,) + lines),
    }


def _source(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def notebook(cells, gpu=None, high_mem=False):
    colab = {"provenance": []}
    metadata = {
        "colab": colab,
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if gpu:
        colab["gpuType"] = gpu
        metadata["accelerator"] = "GPU"
    if high_mem:
        colab["machine_shape"] = "hm"
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 0}


def lit(value):
    """A Python literal for a form default."""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else repr(value)


def choices(values):
    return "[%s]" % ", ".join(json.dumps(v, ensure_ascii=False) for v in values)


BOOTSTRAP = code(
    "1. 掛載 Google Drive，載入 Core",
    'ROOT = "%s"  #@param {type:"string"}' % ROOT_DEFAULT,
    "from google.colab import drive",
    'drive.mount("/content/drive")',
    "import sys",
    "sys.path.insert(0, ROOT)",
    "from controller import notebook as nb",
    "session = nb.connect(ROOT, mount=False)",
)

SERVE_LINES = (
    "#@markdown 佇列空了這麼多分鐘後停止等待：",
    'idle_minutes = 10  #@param {type:"integer"}',
    "#@markdown 停止後釋放 runtime（不再消耗運算單元）。取消勾選的話 runtime 會繼續計費，要自己中斷。",
    'release_runtime = True  #@param {type:"boolean"}',
)
RELEASE_LINES = (
    "#@markdown 存完後釋放 runtime（不再消耗運算單元）。取消勾選的話 runtime 會繼續計費，要自己中斷。",
    'release_runtime = True  #@param {type:"boolean"}',
)
STORAGE_NOTE = (
    "**存放位置**：模型、生成結果、job 記錄都在 Google Drive 的 `AI-Workflow/`；Colab 這台機器上只有運算過程中的暫存。"
    "模型第一次下載後會在背景存入 Drive，**最後一格「儲存並結束」會等它存完並確實寫入 Drive**——"
    "在那之前刪除 runtime，還沒存完的模型下次要重新下載。"
)


def _prefetch_cell(number, reg):
    lines, pairs = [], []
    for name, wf in reg["workflows"].items():
        if not wf.get("models") or not wf.get("enabled", True):
            continue
        variable = "fetch_" + name.replace("-", "_")
        gigabytes = sum(m["size"] for m in wf["models"]) / 1e9
        lines += [
            "#@markdown `%s`（%s，約 %.1f GB）" % (name, wf.get("title", name), gigabytes),
            '%s = False  #@param {type:"boolean"}' % variable,
        ]
        pairs.append("%s: %s" % (lit(name), variable))
    return code(
        "%d. （選用）預先下載模型到 Google Drive" % number,
        "#@markdown 勾選要先備好的 workflow。一次下載一個檔、驗證後直接存進 Drive，不需要 GPU，也不會生成任何東西。",
        "#@markdown 已經在 Drive 上的檔案會略過。",
        *lines,
        "saved = nb.prefetch_models(session, {%s})" % ", ".join(pairs),
    )


def _finish_cell(number, with_queue=True):
    if not with_queue:
        return code(
            "%d. 儲存並結束" % number,
            "#@markdown 確認這個 session 寫入的東西都已進到 Google Drive（flush）。",
            *RELEASE_LINES,
            "summary = nb.finish(session, release_runtime)",
        )
    return code(
        "%d. 儲存並結束" % number,
        "#@markdown 等模型存入 Google Drive、把模型與生成結果確實寫入 Drive（flush），然後釋放 runtime。",
        "#@markdown 想連續生成多張：先不要跑這一格，重複執行上一格；全部做完再跑這一格。",
        "#@markdown ---",
        "#@markdown （選用）結束前先繼續等待 Agent 建立的 job，像 01 一樣：",
        'wait_for_jobs = False  #@param {type:"boolean"}',
        *SERVE_LINES,
        "if wait_for_jobs:",
        "    summary = nb.serve(session, idle_minutes, release_runtime)",
        "else:",
        "    summary = nb.finish(session, release_runtime)",
    )


def setup_notebook(reg):
    return notebook(
        [
            md(
                "# 00 環境初始化",
                "",
                "第一次使用、或換了新的 Colab session 時執行。**執行階段 > 全部執行** 即可。",
                "",
                "- 掛載 Google Drive（工作區 `AI-Workflow/`）",
                "- 檢查 Python／CUDA／GPU／VRAM／Drive／磁碟",
                "- 準備 ComfyUI 與 Python 套件：工作區已有就解壓，沒有才安裝並存回工作區",
                "",
                "這本 notebook 不啟動 ComfyUI。模型預設在第一次執行對應的 workflow 時下載；"
                "想先把模型備好，勾選第 4 格的 workflow 再執行。",
                "",
                STORAGE_NOTE,
            ),
            BOOTSTRAP,
            code("2. 環境檢查與 ComfyUI 安裝", "actions = nb.setup(session)"),
            code("3. 模型檢查（哪些已經在 Google Drive）", "nb.check_models(session)"),
            _prefetch_cell(4, reg),
            _finish_cell(5, with_queue=False),
            md(
                "下一步：`01_comfyui.ipynb`（啟動與 API 測試）、`02_image_generation.ipynb`、`03_video_generation.ipynb`。"
            ),
        ],
        gpu="T4",
    )


def comfyui_notebook(reg):
    return notebook(
        [
            md(
                "# 01 ComfyUI：啟動、健康檢查、等待 Agent 的 job",
                "",
                "**執行階段 > 全部執行**：啟動 ComfyUI → API 測試（不需要模型）→ 開始等待 `jobs/pending` 裡的 job。",
                "",
                "AI Agent 模式就是讓這本 notebook 開著：Agent 用工具建立 job，這裡負責執行並把結果寫回 Google Drive。",
                "",
                "GPU 要自己選（執行階段 > 變更執行階段類型）：Z-Image 需要 **L4** 以上，MiniMax H3 需要 **A100**。"
                "GPU 不夠的 job 會留在佇列裡等，不會被標成失敗。",
            ),
            BOOTSTRAP,
            code("2. 啟動 ComfyUI 與健康檢查", "stats = nb.start_comfyui(session)"),
            code("3. ComfyUI API 測試", "test = nb.api_test(session)"),
            code(
                "4. 等待並執行 job（Agent 模式）",
                "#@markdown 取消勾選就只做上面的啟動與測試，不進入等待。",
                'wait_for_jobs = True  #@param {type:"boolean"}',
                *SERVE_LINES,
                "if wait_for_jobs:",
                "    summary = nb.serve(session, idle_minutes, release_runtime)",
            ),
        ],
        gpu="L4",
    )


def _advanced_fields(wf, skip):
    lines, names = [], []
    for name, p in wf["parameters"].items():
        if name in skip or p.get("simple"):
            continue
        kind = p["type"]
        label = p.get("label", name)
        default = p.get("default")
        if kind in ("string", "negative"):
            field = '%s = %s  #@param {type:"string"}' % (name, lit(default or ""))
        elif kind == "bool":
            field = '%s = %s  #@param {type:"boolean"}' % (name, lit(bool(default)))
        elif kind == "choice":
            field = "%s = %s  #@param %s" % (name, lit(default), choices(p["choices"]))
        elif kind == "seed":
            field = '%s = -1  #@param {type:"integer"}' % name
        elif kind == "int":
            field = '%s = %s  #@param {type:"integer"}' % (name, lit(default if default is not None else 0))
        elif kind == "float":
            field = '%s = %s  #@param {type:"number"}' % (name, lit(default if default is not None else 0.0))
        else:
            continue
        lines += ["#@markdown %s" % label, field]
        names.append(name)
    return lines, names


def _generate_cell(title, reg, kind, default_workflow, simple_lines, simple_names, extra_lines=(), call_extra=""):
    wf = reg["workflows"][default_workflow]
    names = [n for n, w in reg["workflows"].items() if w["type"] == kind and w.get("enabled", True)]
    names.sort(key=lambda n: (n != default_workflow, "test" in reg["workflows"][n].get("task_types", [])))
    advanced_lines, advanced_names = _advanced_fields(wf, set(simple_names))
    return code(
        title,
        "#@markdown ### Workflow",
        "workflow = %s  #@param %s {allow-input: true}" % (lit(default_workflow), choices(names)),
        "#@markdown ### Simple",
        *simple_lines,
        *extra_lines,
        "#@markdown ---",
        "#@markdown ### Advanced（打勾後，下面的欄位才會生效；0 或空白 = 用預設值）",
        'use_advanced = False  #@param {type:"boolean"}',
        *advanced_lines,
        '#@markdown 其他參數（JSON，例如 `{"steps": 12}`；可用的名稱見 `nb.describe(session, workflow)`）',
        'extra_json = "{}"  #@param {type:"string"}',
        "simple = dict(%s)" % ", ".join("%s=%s" % (n, n) for n in simple_names),
        "advanced = dict(%s) if use_advanced else None" % ", ".join("%s=%s" % (n, n) for n in advanced_names),
        "result = nb.generate(session, workflow, simple, advanced, extra=extra_json if use_advanced else None%s)"
        % call_extra,
    )


def image_notebook(reg):
    wf = reg["workflows"]["z-image-basic"]
    aspect = wf["parameters"]["aspect"]
    simple_lines = [
        "#@markdown %s" % wf["parameters"]["prompt"]["label"],
        'prompt = %s  #@param {type:"string"}' % lit(SAMPLE_IMAGE_PROMPT),
        "#@markdown %s" % aspect["label"],
        "aspect = %s  #@param %s" % (lit(aspect["default"]), choices(list(aspect["sizes"]))),
    ]
    return notebook(
        [
            md(
                "# 02 圖片生成（Z-Image）",
                "",
                "**執行階段 > 全部執行**：掛載 Drive → 啟動 ComfyUI → 用下面表單的內容生成一張圖 → PNG 存到 "
                "`AI-Workflow/outputs/images/`。",
                "",
                "「全部執行」會生成一張然後存檔並釋放 runtime。要連續生成多張：依序執行第 1–3 格，"
                "改表單、重跑第 3 格幾次都可以，最後再執行第 4 格。需要 **L4** 以上的 GPU。"
                "第一次執行會下載模型（約 21 GB）並存進 Google Drive，之後的 session 直接從 Drive 讀取。",
                "",
                STORAGE_NOTE,
                "",
                "安全負面詞與 CFG 下限（1.5）由 Core 套用，表單無法關閉。",
            ),
            BOOTSTRAP,
            code("2. 啟動 ComfyUI", "stats = nb.start_comfyui(session)", 'nb.check_models(session, "z-image-basic")'),
            _generate_cell("3. 生成圖片", reg, "image", "z-image-basic", simple_lines, ["prompt", "aspect"]),
            _finish_cell(4),
        ],
        gpu="L4",
    )


def video_notebook(reg):
    wf = reg["workflows"]["minimax-h3-basic"]
    p = wf["parameters"]
    simple_lines = [
        "#@markdown %s" % p["prompt"]["label"],
        'prompt = %s  #@param {type:"string"}' % lit(SAMPLE_VIDEO_PROMPT),
        "#@markdown %s" % p["duration"]["label"],
        'duration = %s  #@param {type:"slider", min:%s, max:%s, step:1}'
        % (p["duration"]["default"], p["duration"]["min"], p["duration"]["max"]),
        "#@markdown %s" % p["aspect"]["label"],
        "aspect = %s  #@param %s" % (lit(p["aspect"]["default"]), choices(p["aspect"]["choices"])),
    ]
    frame = wf["inputs"]["first_frame"]
    frame_lines = [
        "#@markdown ### 首幀",
        "#@markdown `generate`：先用 Z-Image 依 Prompt 生一張首幀。`job`：用已完成的圖片 job。`file`：用 `AI-Workflow/inputs/` 裡的圖。",
        "first_frame_source = %s  #@param %s" % (lit(frame["default_source"]), choices(frame["sources"])),
        "#@markdown 來源是 `job` 時填 Job ID：",
        'first_frame_job_id = ""  #@param {type:"string"}',
        "#@markdown 來源是 `file` 時填路徑（例如 `inputs/frame.png`），並且**必須**勾選下面的聲明：",
        'first_frame_file = ""  #@param {type:"string"}',
        "#@markdown %s" % ATTESTATION_ZH,
        'first_frame_attested = False  #@param {type:"boolean"}',
        "first_frame = nb.input_ref(first_frame_source, first_frame_job_id, first_frame_file, first_frame_attested)",
    ]
    return notebook(
        [
            md(
                "# 03 影片生成（MiniMax H3）",
                "",
                "**執行階段 > 全部執行**：掛載 Drive → 啟動 ComfyUI → 依 Prompt 生成首幀（Z-Image）→ H3 生成影片（含音訊）→ "
                "MP4 存到 `AI-Workflow/outputs/videos/`，預覽圖在 `outputs/previews/`。",
                "",
                "「全部執行」會生成一支然後存檔並釋放 runtime。要連續生成多支：依序執行第 1–3 格，"
                "重跑第 3 格，最後再執行第 4 格。",
                "",
                "需要 **A100**。第一次執行會下載模型（H3 約 43 GB、Z-Image 約 21 GB）並存進 Google Drive；"
                "這麼大的檔案存進 Drive 需要一段時間，第 4 格會等它完成。",
                "",
                STORAGE_NOTE,
                "",
                "H3 沒有負面詞，所以 Prompt 會經過關鍵字過濾（命中就拒絕，沒有略過的選項），"
                "首幀只能是本專案生成的圖，或由你聲明來源的 `inputs/` 圖片。不能使用真人照片。",
            ),
            BOOTSTRAP,
            code(
                "2. 啟動 ComfyUI", "stats = nb.start_comfyui(session)", 'nb.check_models(session, "minimax-h3-basic")'
            ),
            _generate_cell(
                "3. 生成影片",
                reg,
                "video",
                "minimax-h3-basic",
                simple_lines,
                ["prompt", "duration", "aspect"],
                extra_lines=frame_lines,
                call_extra=', inputs={"first_frame": first_frame}',
            ),  # fmt: skip
            _finish_cell(4),
        ],
        gpu="A100",
        high_mem=True,
    )


def debug_notebook(reg):
    return notebook(
        [
            md(
                "# 99 手動除錯",
                "",
                "每一格是一項獨立的檢查，可以全部執行，也可以只跑需要的那一格。這本 notebook 不會釋放 runtime。",
            ),
            BOOTSTRAP,
            code("2. GPU／VRAM", "rows = nb.gpu_report(session)"),
            code("3. 環境檢查", "ok = nb.check_environment(session)"),
            code("4. 模型檢查", "ok = nb.check_models(session)"),
            code(
                "5. 啟動 ComfyUI、Workflow 節點檢查",
                "stats = nb.start_comfyui(session)",
                "ok = nb.check_workflows(session)",
            ),
            code("6. ComfyUI API 測試", "test = nb.api_test(session)"),
            code("7. Job 列表", "found = nb.list_jobs(session)"),
            code("8. ComfyUI log（最後 4000 字）", "nb.comfy_log(session)"),
        ],
        gpu="T4",
    )


NOTEBOOKS = {
    "00_setup.ipynb": setup_notebook,
    "01_comfyui.ipynb": comfyui_notebook,
    "02_image_generation.ipynb": image_notebook,
    "03_video_generation.ipynb": video_notebook,
    "99_manual_debug.ipynb": debug_notebook,
}


def render_all():
    reg = json.loads((SOURCE / "workflows" / "registry.json").read_text(encoding="utf-8"))
    return {name: json.dumps(build(reg), indent=1, ensure_ascii=False) + "\n" for name, build in NOTEBOOKS.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the notebooks from the registry.")
    parser.add_argument("--check", action="store_true", help="exit 1 if a committed notebook differs")
    args = parser.parse_args(argv)
    stale = []
    for name, text in render_all().items():
        path = NOTEBOOK_DIR / name
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == text:
            continue
        stale.append(name)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    if args.check and stale:
        print("out of date: %s (run build_notebooks.py)" % ", ".join(stale))
        return 1
    print("%d notebook(s) %s" % (len(stale), "need rebuilding" if args.check else "written"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
