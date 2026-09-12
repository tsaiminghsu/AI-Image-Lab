"""Coverage for _submit_and_wait's HTTP/polling contract, the LCM<->non-LCM mode-switch reset
logic it drives, and the retry/cleanup/concurrency behaviour around both.

The regression this file started out pinning down: status["completed"] never flips True on a
server-side node error, so a naive "poll until completed" loop would spin for the full
timeout_seconds (up to 2400s for the AnimateDiff path) on a job that failed in under a second.
_submit_and_wait checks status_str == "error" explicitly to surface that failure immediately.

The second theme is resilience, and it hinges on one asymmetry: GET /history, GET /view and the
overwrite=true upload can be repeated for free, but POST /prompt queues a whole extra generation
on an 8GB card. The tests below hold that line - a flaky poll must never cost a second job, and
abandoning a job must never leave it rendering (or kill somebody else's).
"""

import os
import threading
import time
from urllib.parse import urlparse

import pytest
import requests

import comfyui_client as client
from helpers import FakeComfy, FakeResponse

_real_sleep = time.sleep  # kept out of reach of the no_sleep fixture (see the threading test)

# Grabbed before any test monkeypatches client._submit_and_wait (see the LCM tests below), so
# these tests always exercise the real function regardless of what other fixtures replace it with.
_real_submit_and_wait = client._submit_and_wait


def test_node_errors_raise_before_any_history_get(fake_comfy):
    fake_comfy.node_errors = {"3": ["required input missing"]}
    with pytest.raises(RuntimeError, match="workflow validation"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=5)
    # "/history" (no trailing slash, no id) is _reset_if_mode_switch's own last-executed-prompt
    # check and fires before every POST /prompt regardless of node_errors; "/history/<id>" is the
    # completion-polling GET this test guards against ever happening.
    assert not any(path.startswith("/history/") for path, _ in fake_comfy.gets)


def test_execution_error_surfaces_immediately(fake_comfy):
    fake_comfy.history_sequence = [{}, FakeComfy.failed("KSampler", "3", "boom")]
    with pytest.raises(RuntimeError, match=r"KSampler \(node 3\): boom"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=5)


def test_completed_downloads_output_and_queries_view_correctly(fake_comfy, monkeypatch):
    raw_dir = os.environ["COMFYUI_RAW_OUTPUT_DIR"]
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]
    fake_comfy.view_bytes = b"PNGDATA"

    path = _real_submit_and_wait({"3": {}}, timeout_seconds=5)

    assert os.path.dirname(path) == raw_dir
    with open(path, "rb") as f:
        assert f.read() == b"PNGDATA"
    view_calls = [params for p, params in fake_comfy.gets if p == "/view"]
    assert view_calls == [{"filename": "out.png", "subfolder": "", "type": "output"}]


def test_output_node_id_is_respected(fake_comfy):
    fake_comfy.history_sequence = [FakeComfy.done("7", "vid.mp4")]
    path = _real_submit_and_wait({"3": {}}, output_node_id="7", timeout_seconds=5)
    assert os.path.basename(path) == "vid.mp4"


def test_empty_history_forever_times_out_naming_the_prompt_id(fake_comfy):
    fake_comfy.prompt_id = "p42"
    # history_sequence left empty: GET /history/<id> always comes back with nothing for this id.
    with pytest.raises(TimeoutError, match="p42"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=0)


def test_client_id_included_when_supplied(fake_comfy):
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]
    _real_submit_and_wait({"3": {}}, timeout_seconds=5, client_id="abc123")
    (path, payload) = next(p for p in fake_comfy.posted if p[0] == "/prompt")
    assert payload["client_id"] == "abc123"


def test_client_id_omitted_when_not_supplied(fake_comfy):
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]
    _real_submit_and_wait({"3": {}}, timeout_seconds=5)
    (path, payload) = next(p for p in fake_comfy.posted if p[0] == "/prompt")
    assert "client_id" not in payload


# -- LCM <-> non-LCM mode-switch reset ------------------------------------------------------


def _build_workflow_via_animatediff(lcm_preset=None):
    """Get a realistic AnimateDiff workflow dict out of the real submit_generation_animatediff,
    without it ever touching the network: swap _submit_and_wait for a recorder for the duration
    of the call only, then hand back the exact wf dict it built."""
    calls = []

    def fake(wf, output_node_id="9", timeout_seconds=0, client_id=None):
        calls.append(wf)
        return os.path.join("fake", "out.mp4")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(client, "_submit_and_wait", fake)
        client.submit_generation_animatediff(
            prompt="p",
            negative_prompt="n",
            seed=1,
            face_ref_image_filename="face.png",
            filename_prefix="test",
            lcm_preset=lcm_preset,
        )
    return calls[0]


def _history_with_last_prompt(wf):
    """Shape _last_executed_workflow() expects from GET /history?max_items=1: prompt is a
    [num, prompt_id, workflow_dict, extra_data, outputs] tuple/list, workflow_dict at index 2."""
    return {"some-id": {"prompt": [1, "some-id", wf, {}, []]}}


def test_lcm_mode_switch_frees_before_prompt_when_mode_differs(fake_comfy):
    non_lcm_wf = _build_workflow_via_animatediff(lcm_preset=None)
    lcm_wf = _build_workflow_via_animatediff(lcm_preset="lcm_lora")
    assert client._is_lcm_workflow(non_lcm_wf) is False
    assert client._is_lcm_workflow(lcm_wf) is True

    fake_comfy.last_history = _history_with_last_prompt(non_lcm_wf)
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.mp4")]

    _real_submit_and_wait(lcm_wf, timeout_seconds=5)

    assert fake_comfy.paths("post") == ["/free", "/prompt"]


def test_same_mode_does_not_free(fake_comfy):
    non_lcm_wf = _build_workflow_via_animatediff(lcm_preset=None)
    fake_comfy.last_history = _history_with_last_prompt(non_lcm_wf)
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.mp4")]

    _real_submit_and_wait(non_lcm_wf, timeout_seconds=5)

    assert fake_comfy.paths("post") == ["/prompt"]


def test_fresh_server_does_not_free(fake_comfy):
    lcm_wf = _build_workflow_via_animatediff(lcm_preset="lcm_lora")
    fake_comfy.last_history = {}  # no prior prompt at all
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.mp4")]

    _real_submit_and_wait(lcm_wf, timeout_seconds=5)

    assert fake_comfy.paths("post") == ["/prompt"]


def test_mode_switch_skips_free_when_the_queue_is_busy(fake_comfy):
    """Another client's job is running: /free would unload the models it is sampling with, so
    the reset is skipped even though the mode differs. Ours is not queued yet at this point,
    so anything the queue reports is by definition somebody else's."""
    non_lcm_wf = _build_workflow_via_animatediff(lcm_preset=None)
    lcm_wf = _build_workflow_via_animatediff(lcm_preset="lcm_lora")
    fake_comfy.last_history = _history_with_last_prompt(non_lcm_wf)
    fake_comfy.queue_running = ["someone-elses-job"]
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.mp4")]

    _real_submit_and_wait(lcm_wf, timeout_seconds=5)

    assert fake_comfy.paths("post") == ["/prompt"]


def test_mode_switch_skips_free_when_something_is_merely_pending(fake_comfy):
    non_lcm_wf = _build_workflow_via_animatediff(lcm_preset=None)
    lcm_wf = _build_workflow_via_animatediff(lcm_preset="lcm_lora")
    fake_comfy.last_history = _history_with_last_prompt(non_lcm_wf)
    fake_comfy.queue_pending = ["someone-elses-queued-job"]
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.mp4")]

    _real_submit_and_wait(lcm_wf, timeout_seconds=5)

    assert fake_comfy.paths("post") == ["/prompt"]


def test_queue_ids_reads_comfyuis_own_entry_shape(fake_comfy):
    fake_comfy.queue_running, fake_comfy.queue_pending = ["a"], ["b", "c"]
    assert client._queue_ids() == (["a"], ["b", "c"])


# -- E1: polling retries (GET /history/<id> is idempotent - retry generously) -------------------


def test_poll_survives_transient_failures_without_resubmitting(fake_comfy, no_sleep):
    """Two dead polls then a good one: the job still completes, and - the whole point of
    splitting the policy - POST /prompt was never repeated."""
    fake_comfy.get_failures["/history/"] = [
        requests.exceptions.ConnectionError("connection aborted"),
        requests.exceptions.ReadTimeout("read timed out"),
    ]
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    path = _real_submit_and_wait({"3": {}}, timeout_seconds=60)

    assert os.path.basename(path) == "out.png"
    assert fake_comfy.paths("post").count("/prompt") == 1
    assert no_sleep[:2] == [2, 4]  # the documented 2/4/8/16 backoff


def test_poll_retries_http_500_and_non_json_bodies(fake_comfy, no_sleep):
    """ComfyUI restarting serves its aiohttp error page, so .json() raises - and a 5xx never
    reaches .json() at all. Both are "ask again", not "the job is gone"."""
    fake_comfy.get_failures["/history/"] = [
        FakeResponse(status_code=503),
        FakeResponse(status_code=200, body=None),  # body None -> .json() raises ValueError
    ]
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    assert os.path.basename(_real_submit_and_wait({"3": {}}, timeout_seconds=60)) == "out.png"


def test_poll_gives_up_after_the_retry_budget(fake_comfy, no_sleep):
    fake_comfy.get_failures["/history/"] = [requests.exceptions.ConnectionError("down")] * 30

    with pytest.raises(RuntimeError, match="沒有回應"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=600)

    assert sum(no_sleep) > client.POLL_RETRY_BUDGET_SECONDS
    # Bounded, not endless: 2+4+8+16+16... crosses 120s on the 11th consecutive failure.
    assert len([p for p, _ in fake_comfy.gets if p.startswith("/history/")]) == 11


def test_a_good_poll_resets_the_retry_budget(fake_comfy, no_sleep):
    """10 failures, one "still running" answer, 10 more failures, then done. Twice the budget
    in total slept, but never 11 consecutive failures - so the job is never abandoned."""
    fail = requests.exceptions.ConnectionError("down")
    in_progress = FakeResponse(body={})  # a real answer, just not a finished job
    fake_comfy.get_failures["/history/"] = [fail] * 10 + [in_progress] + [fail] * 10
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    path = _real_submit_and_wait({"3": {}}, timeout_seconds=600)

    assert os.path.basename(path) == "out.png"
    assert sum(no_sleep) > 2 * client.POLL_RETRY_BUDGET_SECONDS


# -- E1: POST /prompt is NOT idempotent -------------------------------------------------------


def test_read_timeout_on_post_prompt_raises_immediately(fake_comfy, no_sleep):
    """The request was sent; ComfyUI may already be rendering it. Retrying would queue a
    second generation on an 8GB card, so stop and tell the user to look at the queue."""
    fake_comfy.post_failures["/prompt"] = [requests.exceptions.ReadTimeout("read timed out")]

    with pytest.raises(RuntimeError, match="佇列"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=60)

    assert fake_comfy.paths("post").count("/prompt") == 1
    assert no_sleep == []


def test_refused_connection_on_post_prompt_is_retried(fake_comfy, no_sleep):
    """Connection refused = nothing reached ComfyUI, so resending cannot double-queue."""
    fake_comfy.post_failures["/prompt"] = [FakeComfy.refused(), FakeComfy.refused()]
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    path = _real_submit_and_wait({"3": {}}, timeout_seconds=60)

    assert os.path.basename(path) == "out.png"
    assert fake_comfy.paths("post").count("/prompt") == 3
    assert no_sleep == [2, 4]
    polled = {p for p, _ in fake_comfy.gets if p.startswith("/history/")}
    assert polled == {f"/history/{fake_comfy.prompt_id}"}  # one prompt id, one generation


def test_connect_timeout_on_post_prompt_is_retried(fake_comfy, no_sleep):
    fake_comfy.post_failures["/prompt"] = [requests.exceptions.ConnectTimeout("connect timed out")]
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    _real_submit_and_wait({"3": {}}, timeout_seconds=60)

    assert fake_comfy.paths("post").count("/prompt") == 2


def test_post_prompt_gives_up_after_three_attempts(fake_comfy, no_sleep):
    fake_comfy.post_failures["/prompt"] = [FakeComfy.refused()] * 5

    # ComfyUIUnavailable, not the raw ConnectionError: it still IS one by __cause__, but the
    # message the user sees has to be a sentence naming the server, not a urllib3 chain.
    with pytest.raises(client.ComfyUIUnavailable) as excinfo:
        _real_submit_and_wait({"3": {}}, timeout_seconds=60)
    assert isinstance(excinfo.value.__cause__, requests.exceptions.ConnectionError)

    assert fake_comfy.paths("post").count("/prompt") == client.PROMPT_POST_MAX_ATTEMPTS


def test_a_plain_connection_error_is_not_treated_as_refused():
    """ "Connection aborted" mid-body could mean the request landed - not retriable."""
    assert client._is_connection_refused(requests.exceptions.ConnectionError("Connection aborted")) is False
    assert client._is_connection_refused(FakeComfy.refused()) is True


# -- E1: orphan cleanup -----------------------------------------------------------------------


def _cleanup_posts(fake_comfy):
    return [(p, payload) for p, payload in fake_comfy.posted if p in ("/queue", "/interrupt")]


def test_timeout_dequeues_our_pending_prompt(fake_comfy, no_sleep):
    fake_comfy.queue_pending = ["p1"]

    with pytest.raises(TimeoutError):
        _real_submit_and_wait({"3": {}}, timeout_seconds=0)

    assert _cleanup_posts(fake_comfy) == [("/queue", {"delete": ["p1"]})]  # queued, not running


def test_timeout_interrupts_only_when_our_prompt_is_the_running_one(fake_comfy, no_sleep):
    fake_comfy.queue_running = ["p1"]

    with pytest.raises(TimeoutError):
        _real_submit_and_wait({"3": {}}, timeout_seconds=0)

    assert _cleanup_posts(fake_comfy) == [
        ("/queue", {"delete": ["p1"]}),
        ("/interrupt", {"prompt_id": "p1"}),
    ]


def test_no_interrupt_when_a_different_prompt_is_running(fake_comfy, no_sleep):
    """Never kill somebody else's job while tidying up after ours - older ComfyUI builds
    ignore the prompt_id in the body and interrupt whatever is executing."""
    fake_comfy.queue_running = ["another-clients-job"]

    with pytest.raises(TimeoutError):
        _real_submit_and_wait({"3": {}}, timeout_seconds=0)

    assert "/interrupt" not in fake_comfy.paths("post")


def test_keyboard_interrupt_cleans_up_and_still_propagates(fake_comfy, no_sleep):
    fake_comfy.queue_running = ["p1"]
    fake_comfy.get_failures["/history/"] = [KeyboardInterrupt]

    with pytest.raises(KeyboardInterrupt):
        _real_submit_and_wait({"3": {}}, timeout_seconds=60)

    assert [p for p, _ in _cleanup_posts(fake_comfy)] == ["/queue", "/interrupt"]


def test_a_failing_cleanup_never_masks_the_original_error(fake_comfy, no_sleep):
    fake_comfy.queue_running = ["p1"]
    fake_comfy.post_failures["/queue"] = [RuntimeError("cleanup exploded")]
    fake_comfy.post_failures["/interrupt"] = [RuntimeError("interrupt exploded")]
    fake_comfy.get_failures["/queue"] = [requests.exceptions.ConnectionError("queue unreachable")]

    with pytest.raises(TimeoutError, match="p1"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=0)


def test_no_cleanup_when_the_job_simply_fails(fake_comfy):
    """A node error is the server's final word - nothing is left queued to clean up."""
    fake_comfy.history_sequence = [FakeComfy.failed("KSampler", "3", "boom")]

    with pytest.raises(RuntimeError, match="boom"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=60)

    assert _cleanup_posts(fake_comfy) == []


# -- E1: the same bounded retry on the other two idempotent calls -------------------------------


def test_view_download_is_retried(fake_comfy, no_sleep):
    fake_comfy.get_failures["/view"] = [requests.exceptions.ReadTimeout("read timed out")]
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    path = _real_submit_and_wait({"3": {}}, timeout_seconds=60)

    with open(path, "rb") as f:
        assert f.read() == b"PNGDATA"


def test_upload_reference_image_is_retried(fake_comfy, no_sleep, tmp_path):
    ref = tmp_path / "anchor.png"
    ref.write_bytes(b"IMG")
    fake_comfy.post_failures["/upload/image"] = [FakeComfy.refused()]

    assert client.upload_reference_image(str(ref)) == "uploaded.png"
    assert fake_comfy.paths("post").count("/upload/image") == 2


# -- E2: one generation at a time ---------------------------------------------------------------


def test_two_threads_do_not_interleave_their_submissions(fake_comfy, monkeypatch):
    """Drives the real _CLIENT_LOCK: the fake holds POST /prompt open long enough that an
    unlocked _submit_and_wait would let the second thread's POST land inside the first one's."""
    events = []

    class SlowPromptComfy(FakeComfy):
        def post(self, url, json=None, files=None, data=None, timeout=None):
            if urlparse(url).path == "/prompt":
                events.append(("enter", threading.current_thread().name))
                _real_sleep(0.05)
                events.append(("leave", threading.current_thread().name))
            return super().post(url, json=json, files=files, data=data, timeout=timeout)

    slow = SlowPromptComfy(history_sequence=[FakeComfy.done("9", "out.png")] * 2)
    monkeypatch.setattr(client, "requests", slow)

    threads = [
        threading.Thread(target=_real_submit_and_wait, args=({"3": {}},), kwargs={"timeout_seconds": 60}, name=name)
        for name in ("first", "second")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads)

    assert len(events) == 4
    assert events[0][1] == events[1][1], f"the two POSTs interleaved: {events}"
    assert events[2][1] == events[3][1], f"the two POSTs interleaved: {events}"
    assert events[0][1] != events[2][1]


def test_submit_and_wait_is_reentrant_for_one_thread():
    """RLock, not Lock: a nested call on the same thread must not self-deadlock."""
    assert client._CLIENT_LOCK.acquire(blocking=False)
    try:
        assert client._CLIENT_LOCK.acquire(blocking=False)
        client._CLIENT_LOCK.release()
    finally:
        client._CLIENT_LOCK.release()


# -- _is_lcm_workflow detectors ---------------------------------------------------------------


def test_is_lcm_workflow_detects_beta_schedule():
    wf = {"2": {"class_type": "ADE_AnimateDiffLoaderGen1", "inputs": {"beta_schedule": "lcm[sqrt_linear]"}}}
    assert client._is_lcm_workflow(wf) is True


def test_is_lcm_workflow_detects_lcm_lora():
    lora_file = next(iter(client.LCM_LORA_FILES))
    for class_type in ("LoraLoaderModelOnly", "LoraLoader"):
        wf = {"61": {"class_type": class_type, "inputs": {"lora_name": lora_file}}}
        assert client._is_lcm_workflow(wf) is True


def test_is_lcm_workflow_detects_lcm_sampler():
    wf = {"3": {"class_type": "KSampler", "inputs": {"sampler_name": "lcm"}}}
    assert client._is_lcm_workflow(wf) is True


def test_is_lcm_workflow_false_otherwise():
    wf = {"3": {"class_type": "KSampler", "inputs": {"sampler_name": "dpmpp_2m"}}}
    assert client._is_lcm_workflow(wf) is False
    assert client._is_lcm_workflow({}) is False


# -- is_server_running / has_node swallow connection errors -------------------------------------


def test_is_server_running_false_when_down(fake_comfy):
    fake_comfy.down = True
    assert client.is_server_running() is False


def test_has_node_false_when_down(fake_comfy):
    fake_comfy.down = True
    assert client.has_node("KSampler") is False


# -- _loader_options parses both combo shapes ----------------------------------------------


def test_loader_options_classic_list_shape(fake_comfy):
    fake_comfy.object_info = {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}}}
    }
    assert client._loader_options("CheckpointLoaderSimple", "ckpt_name") == ["a.safetensors", "b.safetensors"]


def test_loader_options_combo_dict_shape(fake_comfy):
    fake_comfy.object_info = {
        "UNETLoader": {"input": {"required": {"unet_name": ["COMBO", {"options": ["x.safetensors"]}]}}}
    }
    assert client._loader_options("UNETLoader", "unet_name") == ["x.safetensors"]


def test_loader_options_malformed_spec_returns_none(fake_comfy):
    fake_comfy.object_info = {"Foo": {"input": {"required": {"bar": "not-a-list-or-combo"}}}}
    assert client._loader_options("Foo", "bar") is None


def test_loader_options_missing_node_returns_none(fake_comfy):
    assert client._loader_options("DoesNotExist", "field") is None


# --- instrumentation and path handling must not be able to lose a finished render --------------


def test_log_gpu_memory_returns_nan_instead_of_raising(fake_comfy, capsys):
    """generate_character's video path calls this AFTER the mp4 is saved. An unguarded raise
    there would throw away a finished render over a telemetry call."""
    fake_comfy.down = True
    used, total = client.log_gpu_memory("after_video")
    assert used != used and total != total, "expected nan, nan"  # nan is the only value != itself
    assert "after_video" in capsys.readouterr().out


def test_log_gpu_memory_reports_real_numbers_when_the_server_answers(fake_comfy):
    fake_comfy.system_stats = {"devices": [{"vram_total": 8 * 1024**3, "vram_free": 2 * 1024**3}]}
    used, total = client.log_gpu_memory("warm")
    assert round(total) == 8192
    assert round(used) == 6144


def test_download_output_cannot_be_written_outside_the_output_dir(fake_comfy, tmp_path, monkeypatch):
    """The filename comes from ComfyUI's own response, but it is still untrusted input being
    joined into a local path: a traversal component in it would write outside out_dir."""
    out_dir = tmp_path / "raw"
    monkeypatch.setenv("COMFYUI_RAW_OUTPUT_DIR", str(out_dir))
    path = client._download_output({"filename": "../../escaped.png", "subfolder": "", "type": "output"})
    assert os.path.dirname(os.path.abspath(path)) == str(out_dir)
    assert os.path.basename(path) == "escaped.png"


# --- "ComfyUI is down" must read as one line, not a urllib3 traceback --------------------------
# Measured before this type existed: pointing the CLI at a dead port printed a full
# requests/urllib3 stack ending in NewConnectionError, burying the one fact the user needs.
# ComfyUIUnavailable gives every entry point something specific to catch: the CLI exits 1 with
# the message, the GUI shows a toast, image_api records it as the job error.


def test_unavailable_is_a_runtime_error_not_a_usage_error():
    """It is an environment problem, so it must not be confused with a caller mistake - but it
    must still be an ordinary Exception, or the boundaries that catch Exception miss it."""
    assert issubclass(client.ComfyUIUnavailable, RuntimeError)
    assert issubclass(client.ComfyUIUnavailable, Exception)


def test_exhausted_post_retries_name_the_server(fake_comfy, no_sleep):
    fake_comfy.post_failures = {"/prompt": [fake_comfy.refused() for _ in range(client.PROMPT_POST_MAX_ATTEMPTS)]}
    with pytest.raises(client.ComfyUIUnavailable) as excinfo:
        client._submit_and_wait({"9": {"class_type": "SaveImage", "inputs": {}}}, timeout_seconds=5)
    message = str(excinfo.value)
    assert client.COMFYUI_URL in message
    assert "NewConnectionError" not in message, "the raw urllib3 chain must not be the message"
    assert len([p for p, _ in fake_comfy.posted if p == "/prompt"]) == client.PROMPT_POST_MAX_ATTEMPTS


def test_a_non_retriable_connection_failure_also_names_the_server(fake_comfy, no_sleep):
    """A ConnectionError that is NOT a refused connection is not resent (it may have arrived),
    but the user still gets a sentence rather than a traceback."""
    fake_comfy.post_failures = {"/prompt": [requests.exceptions.ConnectionError("chunked encoding broke")]}
    with pytest.raises(client.ComfyUIUnavailable) as excinfo:
        client._submit_and_wait({"9": {"class_type": "SaveImage", "inputs": {}}}, timeout_seconds=5)
    assert client.COMFYUI_URL in str(excinfo.value)
    assert len([p for p, _ in fake_comfy.posted if p == "/prompt"]) == 1, "must not be resent"


def test_exhausted_poll_budget_raises_unavailable(fake_comfy, no_sleep):
    fake_comfy.get_failures = {"/history/": [requests.exceptions.ConnectionError("dropped")] * 40}
    with pytest.raises(client.ComfyUIUnavailable) as excinfo:
        client._submit_and_wait({"9": {"class_type": "SaveImage", "inputs": {}}}, timeout_seconds=600)
    assert client.COMFYUI_URL in str(excinfo.value)
