"""Duration/frames, resolution and the ffprobe rules every downloaded MiniMax H3 clip must pass."""

import math

import pytest

from minimax_h3_fakes import good_probe, h3

VIDEO = {"short_side": 768, "max_megapixels": 1.2}


@pytest.mark.parametrize("seconds,frames", [(4, 107), (5, 124), (8, 192), (12, 294), (14.375, 345), (15, 362)])
def test_frames_snap_to_the_17k_plus_5_grid(seconds, frames):
    """The table from ComfyUI's H3 node tooltip and the reference notebook."""
    assert h3.frames_for(seconds) == frames
    assert (frames - 5) % 17 == 0


@pytest.mark.parametrize(
    "image,expected",
    [
        ((1920, 1080), (1376, 768)),
        ((1080, 1920), (768, 1376)),
        ((1024, 768), (1024, 768)),
        ((1024, 1024), (768, 768)),
        ((1536, 1024), (1152, 768)),
    ],
)
def test_auto_resolution_keeps_the_aspect_ratio_at_768p(image, expected):
    assert h3.auto_resolution(*image, short_side=768, max_megapixels=1.2) == expected


def test_auto_resolution_shrinks_to_the_pixel_budget():
    w, h = h3.auto_resolution(2100, 900, short_side=768, max_megapixels=1.2)
    assert (w, h) == (1632, 704)
    assert w * h <= 1.2e6 and w % 32 == 0 and h % 32 == 0


def test_explicit_resolution_close_to_the_image_aspect_is_accepted():
    assert h3.resolve_resolution("1344x768", 1920, 1080, VIDEO) == (1344, 768)  # 1.6 % off 16:9


@pytest.mark.parametrize(
    "requested,match",
    [
        ("1024x1024", "stretched"),
        ("1350x768", "multiples of 32"),
        ("1600x900", "multiples of 32"),
        ("1536x864", "budget"),
        ("192x128", "256-2048"),
        ("big", "look like"),
    ],
)
def test_bad_explicit_resolutions_are_rejected_with_a_suggestion(requested, match):
    with pytest.raises(h3.InputError, match=match):
        h3.resolve_resolution(requested, 1920, 1080, VIDEO)


def test_good_clip_passes():
    assert h3.check_probe(good_probe(), width=1376, height=768, frames=192) == []


@pytest.mark.parametrize(
    "probe,problem",
    [
        ({"streams": [], "format": {}}, "no video stream"),
        (good_probe(codec="hevc"), "codec"),
        (good_probe(pix_fmt="yuv444p"), "pixel format"),
        (good_probe(width=1344), "resolution"),
        (good_probe(fps="30/1"), "frame rate"),
        (good_probe(frames=124), "duration"),
        (good_probe(audio=False), "no audio"),
    ],
)
def test_bad_clips_fail(probe, problem):
    problems = h3.check_probe(probe, width=1376, height=768, frames=192)
    assert problems and any(problem in p for p in problems), problems


def test_empty_file_fails():
    probe = good_probe()
    probe["format"]["size"] = "0"
    assert "file is empty" in h3.check_probe(probe, width=1376, height=768, frames=192)


def test_duration_tolerance_is_two_frames_or_a_tenth_of_a_second():
    probe = good_probe()
    probe["streams"][0]["duration"] = str(192 / 24 + 0.09)
    assert h3.check_probe(probe, width=1376, height=768, frames=192) == []
    probe["streams"][0]["duration"] = str(192 / 24 + 0.2)
    assert h3.check_probe(probe, width=1376, height=768, frames=192)


def test_duration_falls_back_to_frame_count():
    probe = good_probe()
    del probe["streams"][0]["duration"]
    assert h3.check_probe(probe, width=1376, height=768, frames=192) == []


def test_summary_reports_the_fields_in_the_job_record():
    s = h3.summarize_probe(good_probe())
    assert s["video_codec"] == "h264" and s["audio_codec"] == "aac"
    assert (s["width"], s["height"], s["nb_frames"]) == (1376, 768, 192)
    assert math.isclose(s["fps"], 24.0) and math.isclose(s["video_duration"], 8.0)
