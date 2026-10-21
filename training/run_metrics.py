"""Machine metrics sampled while a local picture / video job runs, a line chart of them, and a plain-language reading.

The numbers are the ones this project's notes say matter on this machine: GPU temperature (throttling starts near
84 C), GPU utilisation and power (is the card computing, or waiting for weights?), VRAM used against the card's total,
free system RAM, the page file (swap) in use, and the ComfyUI process's own memory. A job records one row every few
seconds; the rows are saved next to the job's record as <name>.metrics.json and the GUI draws them.

Pure stdlib plus an optional psutil (already in the ComfyUI venv). The chart is plain SVG built here, so it needs no
plotting library and the tests can read it.
"""

import json
import subprocess
import time

THROTTLE_C = 84  # CLAUDE.md: the card throttles in hardware at 84 C
WARM_C = 80
FIELDS = ("gpu_temp", "gpu_util", "gpu_power", "vram_used", "vram_total", "ram_free", "swap_used", "comfy_rss")


def _num(text):
    try:
        return float(str(text).strip())
    except ValueError:
        return None  # nvidia-smi prints [N/A] or [Not Supported] for what a card cannot report


def _comfy_rss_gb(psutil):
    """Resident memory of the ComfyUI server, in GB. The venv's python.exe is a launcher that starts the real
    interpreter with the same command line, so every match is read and the largest one is the server."""
    best = None
    for p in psutil.process_iter(["cmdline", "memory_info"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or []).replace("/", "\\")
            if "main.py" in cmd and "--port" in cmd:  # the server is started as main.py --listen ... --port N
                rss = p.info["memory_info"].rss
                best = rss if best is None else max(best, rss)
        except Exception:  # a process can vanish between listing and reading it
            continue
    return None if best is None else round(best / 2**30, 2)


def sample(run=subprocess.run):
    """One reading of the machine; a value the system cannot report is None."""
    row = dict.fromkeys(FIELDS)
    try:
        out = run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout  # fmt: skip
        temp, util, power, used, total = (_num(x) for x in out.strip().splitlines()[0].split(","))
        row.update(gpu_temp=temp, gpu_util=util, gpu_power=power,
                   vram_used=None if used is None else round(used / 1024, 2),
                   vram_total=None if total is None else round(total / 1024, 2))  # fmt: skip
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        import psutil

        row["ram_free"] = round(psutil.virtual_memory().available / 2**30, 2)
        row["swap_used"] = round(psutil.swap_memory().used / 2**30, 2)
        row["comfy_rss"] = _comfy_rss_gb(psutil)
    except ImportError:
        pass
    return row


class Recorder:
    """Collects rows with the seconds since it was created; add() takes one reading."""

    def __init__(self, label="", sampler=sample, clock=time.time):
        self.label, self.sampler, self.clock = label, sampler, clock
        self.t0 = clock()
        self.rows = []

    def add(self):
        row = dict(self.sampler())
        row["t"] = round(self.clock() - self.t0, 1)
        self.rows.append(row)
        return row

    def save(self, path):
        data = {"label": self.label, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": self.rows}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return path


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("label", ""), data.get("rows", [])


# ---------------------------------------------------------------- reading the numbers


def _values(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def summarize(rows):
    """Key figures and a list of plain-language findings (each starts with ✔, ⚠ or ℹ). Empty rows -> ({}, [])."""
    if not rows:
        return {}, []
    seconds = rows[-1]["t"]
    step = seconds / (len(rows) - 1) if len(rows) > 1 else 0.0
    temps, vram, ram = _values(rows, "gpu_temp"), _values(rows, "vram_used"), _values(rows, "ram_free")
    swap, rss, util = _values(rows, "swap_used"), _values(rows, "comfy_rss"), _values(rows, "gpu_util")
    power = _values(rows, "gpu_power")
    total = max(_values(rows, "vram_total") or [0]) or None
    # Loading ends at the first two readings in a row with the card at least half busy.
    load_s = None
    for a, b in zip(rows, rows[1:]):
        if (a.get("gpu_util") or 0) >= 50 and (b.get("gpu_util") or 0) >= 50:
            load_s = a["t"]
            break
    busy = [r["gpu_util"] for r in rows if load_s is not None and r["t"] >= load_s and r.get("gpu_util") is not None]
    key = {
        "seconds": seconds,
        "samples": len(rows),
        "peak_temp": max(temps) if temps else None,
        "seconds_over_warm": round(sum(1 for t in temps if t >= WARM_C) * step) if temps else None,
        "vram_peak": max(vram) if vram else None,
        "vram_total": total,
        "ram_free_min": min(ram) if ram else None,
        "swap_peak": max(swap) if swap else None,
        "comfy_rss_peak": max(rss) if rss else None,
        "load_seconds": load_s,
        "util_after_load": round(sum(busy) / len(busy)) if busy else None,
        "power_avg": round(sum(power) / len(power)) if power else None,
    }
    notes = []
    if key["peak_temp"] is not None:
        t = key["peak_temp"]
        if t >= THROTTLE_C:
            notes.append(f"⚠ GPU 最高 {t:.0f}°C，到了硬體降頻線（{THROTTLE_C}°C）：這次的時間不能直接跟冷機那次比")
        elif t >= WARM_C:
            notes.append(f"⚠ GPU 最高 {t:.0f}°C，有 {key['seconds_over_warm']} 秒在 {WARM_C}°C 以上，接近降頻線")
        else:
            notes.append(f"✔ GPU 最高 {t:.0f}°C，離降頻線（{THROTTLE_C}°C）還有餘裕")
    if key["vram_peak"] is not None and total:
        room = total - key["vram_peak"]
        if room < 0.5:
            notes.append(
                f"⚠ 顯存峰值 {key['vram_peak']:.1f}/{total:.1f} GB，幾乎滿：再多一點（別的程式、更大的圖）就會失敗或被迫搬到 RAM"
            )
        else:
            notes.append(f"✔ 顯存峰值 {key['vram_peak']:.1f}/{total:.1f} GB，還剩 {room:.1f} GB")
    if key["swap_peak"] is not None and key["swap_peak"] >= 1:
        if key["ram_free_min"] is not None and key["ram_free_min"] >= 4:
            notes.append(
                f"ℹ 分頁檔有 {key['swap_peak']:.1f} GB 在用，但可用 RAM 一直很充裕（最低 {key['ram_free_min']:.1f} GB），影響不大"
            )
        else:
            notes.append(f"⚠ 分頁檔用到 {key['swap_peak']:.1f} GB 而且 RAM 吃緊：權重被換到硬碟，這是變慢的主因之一")
    if key["ram_free_min"] is not None and key["ram_free_min"] < 2:
        notes.append(f"⚠ 可用 RAM 最低只剩 {key['ram_free_min']:.1f} GB（跑的時候別開瀏覽器大量分頁或別的大程式）")
    elif key["ram_free_min"] is not None:
        notes.append(f"✔ 可用 RAM 最低 {key['ram_free_min']:.1f} GB")
    if key["comfy_rss_peak"] is not None:
        notes.append(f"ℹ ComfyUI 本身最多用了 {key['comfy_rss_peak']:.1f} GB 的 RAM")
    if load_s is not None and seconds:
        share = round(100 * load_s / seconds)
        line = f"ℹ 前 {load_s:.0f} 秒（{share}%）GPU 還沒開始滿載＝在載入／搬移模型，之後 GPU 平均使用率 {key['util_after_load']}%"
        if share >= 40:
            line += "。載入佔了大半，瓶頸在模型搬移（匯流排、RAM、硬碟），不是取樣本身"
        notes.append(line)
    elif util and max(util) < 50:
        notes.append("ℹ GPU 使用率一直沒超過 50%：這段時間幾乎都在等（載入或搬移），不是在算")
    return key, notes


# ---------------------------------------------------------------- the chart

_W, _LEFT, _RIGHT, _PANEL_H, _GAP, _TOP = 900, 62, 18, 84, 24, 8


def _nice_step(span):
    for step in (10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if span / step <= 8:
            return step
    return 3600


def _path(points):
    """An SVG path through (x, y) points; None breaks the line (a reading that is missing)."""
    out, pen = [], False
    for p in points:
        if p is None:
            pen = False
            continue
        out.append(f"{'L' if pen else 'M'}{p[0]:.1f},{p[1]:.1f}")
        pen = True
    return " ".join(out)


def chart_svg(rows):
    """Stacked line charts sharing one time axis: temperature, GPU busy, VRAM, RAM free & page file, ComfyUI memory.
    Returns an HTML string (an inline SVG); a message when there is nothing to draw."""
    if len(rows) < 2:
        return "<div style='padding:14px;opacity:.7'>這次沒有足夠的指標資料（至少要兩筆取樣）</div>"
    span = max(rows[-1]["t"], 1.0)
    total = max(_values(rows, "vram_total") or [0]) or None
    panels = [
        (
            "GPU 溫度 °C",
            [("溫度", "gpu_temp", "#e5484d")],
            (20, max(90, max(_values(rows, "gpu_temp") or [0]) + 2)),
            [(THROTTLE_C, "降頻 84°C", "#e5484d")],
        ),
        ("GPU 使用率 %", [("使用率", "gpu_util", "#3e63dd")], (0, 100), []),
        ("GPU 功率 W", [("功率", "gpu_power", "#f5a524")], None, []),
        (
            "顯存 GB",
            [("已用", "vram_used", "#8e4ec6")],
            (0, (total or max(_values(rows, "vram_used") or [1])) * 1.05),
            [(total, f"總量 {total:.1f} GB", "#8e4ec6")] if total else [],
        ),
        ("系統記憶體 GB", [("可用 RAM", "ram_free", "#30a46c"), ("分頁檔已用", "swap_used", "#e5484d")], None, []),
        ("ComfyUI 程序記憶體 GB", [("ComfyUI", "comfy_rss", "#0091ff")], None, []),
    ]
    panels = [p for p in panels if any(_values(rows, k) for _, k, _c in p[1])]
    height = _TOP + len(panels) * (_PANEL_H + _GAP) + 8
    plot_w = _W - _LEFT - _RIGHT
    step = _nice_step(span)
    parts = [f"<svg viewBox='0 0 {_W} {height}' width='100%' xmlns='http://www.w3.org/2000/svg' "
             "font-family='sans-serif' font-size='11' fill='currentColor'>"]  # fmt: skip
    for n, (title, series, domain, refs) in enumerate(panels):
        top = _TOP + n * (_PANEL_H + _GAP)
        vals = [v for _, k, _c in series for v in _values(rows, k)]
        lo, hi = domain or (0, max(vals + [1]) * 1.1)
        if hi <= lo:
            hi = lo + 1
        y = lambda v, top=top, lo=lo, hi=hi: top + _PANEL_H - (v - lo) / (hi - lo) * _PANEL_H  # noqa: E731
        x = lambda t: _LEFT + t / span * plot_w  # noqa: E731
        parts.append(f"<text x='{_LEFT}' y='{top - 4}' font-weight='600'>{title}</text>")
        parts.append(f"<rect x='{_LEFT}' y='{top}' width='{plot_w}' height='{_PANEL_H}' fill='none' stroke='currentColor' "
                     "stroke-opacity='.18'/>")  # fmt: skip
        for tick in range(0, int(span) + 1, step):
            parts.append(f"<line x1='{x(tick):.1f}' y1='{top}' x2='{x(tick):.1f}' y2='{top + _PANEL_H}' "
                         "stroke='currentColor' stroke-opacity='.08'/>")  # fmt: skip
            if n == len(panels) - 1:
                parts.append(
                    f"<text x='{x(tick):.1f}' y='{top + _PANEL_H + 14}' text-anchor='middle' opacity='.7'>{tick} 秒</text>"
                )
        for v in (lo, hi):
            parts.append(f"<text x='{_LEFT - 6}' y='{y(v) + 4:.1f}' text-anchor='end' opacity='.7'>{v:.0f}</text>")
        for value, label, color in refs:
            parts.append(f"<line x1='{_LEFT}' y1='{y(value):.1f}' x2='{_LEFT + plot_w}' y2='{y(value):.1f}' stroke='{color}' "
                         "stroke-dasharray='5 4' stroke-opacity='.7'/>"
                         f"<text x='{_LEFT + plot_w - 4}' y='{y(value) - 4:.1f}' text-anchor='end' fill='{color}'>{label}</text>")  # fmt: skip
        legend_x = _LEFT + 150
        for name, key, color in series:
            points = [None if r.get(key) is None else (x(r["t"]), y(min(max(r[key], lo), hi))) for r in rows]
            if any(points):
                parts.append(
                    f"<path d='{_path(points)}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'/>"
                )
                last = [v for v in _values(rows, key)][-1]
                parts.append(
                    f"<text x='{legend_x}' y='{top - 4}' fill='{color}'>● {name}（最高 {max(_values(rows, key)):.1f}，結束 {last:.1f}）</text>"
                )
                legend_x += 190
    parts.append("</svg>")
    return "<div style='overflow-x:auto'>" + "".join(parts) + "</div>"
