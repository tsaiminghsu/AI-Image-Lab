"""training/run_metrics.py: the machine readings taken while a job runs, the plain-language reading of them and the
SVG line chart. No GPU needed: nvidia-smi is replaced by a fake and the rows are written by hand."""

import json
import subprocess

import run_metrics as rm


def fake_smi(text):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    return run


def row(t, **kw):
    base = dict.fromkeys(rm.FIELDS)
    base.update(t=t, **kw)
    return base


def test_sample_reads_the_card_and_tolerates_what_it_cannot_report():
    got = rm.sample(fake_smi("71, 96, 118.40, 13400, 16311\n"))
    assert got["gpu_temp"] == 71 and got["gpu_util"] == 96 and got["gpu_power"] == 118.4
    assert got["vram_used"] == round(13400 / 1024, 2) and got["vram_total"] == round(16311 / 1024, 2)
    na = rm.sample(fake_smi("65, 3, [N/A], 900, 16311\n"))
    assert na["gpu_power"] is None and na["gpu_temp"] == 65  # one unreadable field does not lose the rest

    def broken(cmd, **kw):
        raise OSError("no nvidia-smi")

    assert rm.sample(broken)["gpu_temp"] is None  # no card: the keys still exist


def test_the_recorder_stamps_rows_with_seconds_and_round_trips_through_a_file(tmp_path):
    ticks = iter([100.0, 100.0, 105.2, 110.4])
    rec = rm.Recorder("H3 test", sampler=lambda: {"gpu_temp": 60.0}, clock=lambda: next(ticks))
    for _ in range(3):
        rec.add()
    assert [r["t"] for r in rec.rows] == [0.0, 5.2, 10.4]
    path = rec.save(str(tmp_path / "a.metrics.json"))
    label, rows = rm.load(path)
    assert label == "H3 test" and rows == rec.rows
    assert json.load(open(path, encoding="utf-8"))["rows"][0]["gpu_temp"] == 60.0


def test_a_hot_tight_swapping_loading_heavy_run_is_called_out():
    rows = [row(0, gpu_temp=50, gpu_util=2, vram_used=2, vram_total=16, ram_free=9, swap_used=3, comfy_rss=30)]
    rows += [row(10 * i, gpu_temp=50 + 4 * i, gpu_util=2, vram_used=8, vram_total=16, ram_free=0.9, swap_used=14,
                 comfy_rss=40) for i in range(1, 6)]  # fmt: skip
    rows += [row(60 + 10 * i, gpu_temp=86, gpu_util=95, vram_used=15.8, vram_total=16, ram_free=1.5, swap_used=12,
                 gpu_power=150, comfy_rss=38) for i in range(0, 5)]  # fmt: skip
    key, notes = rm.summarize(rows)
    text = "\n".join(notes)
    assert key["peak_temp"] == 86 and key["vram_peak"] == 15.8 and key["swap_peak"] == 14
    assert key["load_seconds"] == 60 and key["util_after_load"] == 95
    assert "降頻線" in text and "顯存" in text and "分頁檔" in text and "可用 RAM 最低只剩 0.9 GB" in text
    assert "載入佔了大半" in text  # 60 s of a 100 s run is loading, so the weights moving is the bottleneck


def test_a_comfortable_run_gets_only_good_news():
    rows = [row(10 * i, gpu_temp=60, gpu_util=99, vram_used=10, vram_total=16, ram_free=20, swap_used=0.2)
            for i in range(6)]  # fmt: skip
    key, notes = rm.summarize(rows)
    assert not any(n.startswith("⚠") for n in notes)
    assert key["load_seconds"] == 0 and key["seconds_over_warm"] == 0
    assert rm.summarize([]) == ({}, [])


def test_the_chart_draws_a_panel_per_metric_with_the_throttle_and_vram_limits():
    rows = [row(10 * i, gpu_temp=60 + i, gpu_util=50, gpu_power=100, vram_used=12, vram_total=16, ram_free=20,
                swap_used=1, comfy_rss=None) for i in range(7)]  # fmt: skip
    html = rm.chart_svg(rows)
    assert html.count("<svg") == 1 and "<path" in html
    for title in ("GPU 溫度", "GPU 使用率", "GPU 功率", "顯存 GB", "系統記憶體"):
        assert title in html
    assert "降頻 84°C" in html and "總量 16.0 GB" in html
    assert "ComfyUI 程序記憶體" not in html  # a metric with no readings gets no empty panel
    assert "60 秒" in html  # the time axis
    assert "至少要兩筆" in rm.chart_svg([row(0)])


def test_a_missing_reading_breaks_the_line_instead_of_drawing_through_zero():
    rows = [row(0, gpu_temp=60), row(10, gpu_temp=None), row(20, gpu_temp=62), row(30, gpu_temp=63)]
    html = rm.chart_svg(rows)
    assert html.count("M") >= 2  # two separate strokes: before and after the gap


def test_a_leftover_page_file_is_not_blamed_when_ram_is_plentiful():
    rows = [
        row(10 * i, gpu_temp=50, gpu_util=2, vram_used=1, vram_total=16, ram_free=22, swap_used=1.3) for i in range(5)
    ]
    _, notes = rm.summarize(rows)
    assert not any(n.startswith("⚠") for n in notes)
    assert any("影響不大" in n for n in notes)
