"""Shared test doubles and graph helpers.

FakeComfy stands in for the `requests` module inside comfyui_client. That module calls
`requests.post` / `requests.get` as module attributes, so `monkeypatch.setattr(client,
"requests", FakeComfy())` is enough to intercept every HTTP call without adding a
responses/requests-mock dependency.
"""

from urllib.parse import urlparse

import requests


class FakeResponse:
    def __init__(self, status_code=200, body=None, content=b""):
        self.status_code = status_code
        self._body = body
        self.content = content

    def json(self):
        if self._body is None:
            raise ValueError("no JSON object could be decoded")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeComfy:
    """Minimal ComfyUI HTTP surface: POST /prompt, /free, /queue, /interrupt, /upload/image;
    GET /system_stats, /history, /history/<id>, /queue, /object_info/<node>, /view."""

    # The REAL requests exception hierarchy, not a stand-in: comfyui_client's retry policy
    # turns on the differences between ConnectTimeout / ReadTimeout / ConnectionError (see
    # _post_prompt_with_retry), and a hand-rolled tree would let a wrong isinstance() pass here
    # and fail in production.
    exceptions = requests.exceptions

    class utils:
        @staticmethod
        def quote(s):
            return s

    def __init__(
        self,
        prompt_id="p1",
        node_errors=None,
        history_sequence=None,
        object_info=None,
        view_bytes=b"PNGDATA",
        last_history=None,
        queue_running=None,
        queue_pending=None,
    ):
        self.posted = []  # [(path, json_payload)]
        self.gets = []  # [(path, params)]
        self.prompt_id = prompt_id
        self.node_errors = node_errors or {}
        self.history_sequence = list(history_sequence or [])
        self.object_info = object_info or {}
        self.view_bytes = view_bytes
        # /system_stats body; log_gpu_memory reads devices[0].vram_total/vram_free out of it.
        self.system_stats = {}
        self.last_history = last_history or {}
        self.down = False
        # Prompt ids GET /queue reports as running / pending (see queue_entry below).
        self.queue_running = list(queue_running or [])
        self.queue_pending = list(queue_pending or [])
        # Failures to inject, keyed by request path prefix: {"/history/": [exc, exc, ...]}.
        # Each list is consumed left to right, one item per matching request, before the
        # normal answer is produced - an exception instance/class is raised, a FakeResponse is
        # returned as-is (for HTTP 5xx). Empty/exhausted lists fall through to normal service.
        self.get_failures = {}
        self.post_failures = {}

    # -- helpers for building history bodies ------------------------------------------------
    @staticmethod
    def done(output_node_id="9", filename="out.png"):
        return {
            "status": {"status_str": "success", "completed": True},
            "outputs": {output_node_id: {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}},
        }

    @staticmethod
    def failed(node_type="KSampler", node_id="3", message="boom"):
        return {
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [
                    ["execution_error", {"node_type": node_type, "node_id": node_id, "exception_message": message}]
                ],
            }
        }

    @staticmethod
    def queue_entry(prompt_id):
        """One GET /queue entry, in ComfyUI's own shape: the queue tuple
        (number, prompt_id, prompt, extra_data, outputs_to_execute) with the sensitive 6th
        element already stripped by the server (ComfyUI/server.py:69-71, :1064-1070)."""
        return [0, prompt_id, {}, {}, []]

    @staticmethod
    def refused(message="Failed to establish a new connection: [WinError 10061] refused"):
        """A ConnectionError that comfyui_client._is_connection_refused recognises as
        "never left this machine" - what a stopped ComfyUI looks like on this Windows box."""
        return requests.exceptions.ConnectionError(message)

    def _inject(self, table, path):
        """Pop and apply the next injected failure for this path, if any."""
        for key, queued in table.items():
            if not (path == key or path.startswith(key)) or not queued:
                continue
            item = queued.pop(0)
            if isinstance(item, type) and issubclass(item, BaseException):
                raise item()
            if isinstance(item, BaseException):
                raise item
            return item
        return None

    # -- the requests surface ---------------------------------------------------------------
    def post(self, url, json=None, files=None, data=None, timeout=None):
        if self.down:
            raise self.exceptions.RequestException("connection refused")
        path = urlparse(url).path
        self.posted.append((path, json))  # recorded BEFORE injection, so failed attempts count
        injected = self._inject(self.post_failures, path)
        if injected is not None:
            return injected
        if path == "/prompt":
            return FakeResponse(body={"prompt_id": self.prompt_id, "node_errors": self.node_errors})
        if path == "/free":
            return FakeResponse(body={})
        if path == "/queue":
            for pid in (json or {}).get("delete", []):
                if pid in self.queue_pending:
                    self.queue_pending.remove(pid)
            return FakeResponse(body={})
        if path == "/interrupt":
            return FakeResponse(body={})
        if path == "/upload/image":
            return FakeResponse(body={"name": "uploaded.png"})
        raise AssertionError(f"unexpected POST {path}")

    def get(self, url, params=None, timeout=None):
        if self.down:
            raise self.exceptions.RequestException("connection refused")
        path = urlparse(url).path
        self.gets.append((path, params))  # recorded BEFORE injection, so failed polls count
        injected = self._inject(self.get_failures, path)
        if injected is not None:
            return injected
        if path == "/system_stats":
            return FakeResponse(body=self.system_stats)
        if path == "/queue":
            return FakeResponse(
                body={
                    "queue_running": [self.queue_entry(p) for p in self.queue_running],
                    "queue_pending": [self.queue_entry(p) for p in self.queue_pending],
                }
            )
        if path == "/history":
            return FakeResponse(body=self.last_history)
        if path.startswith("/history/"):
            body = self.history_sequence.pop(0) if self.history_sequence else {}
            return FakeResponse(body={self.prompt_id: body} if body else {})
        if path.startswith("/object_info/"):
            node = path.rsplit("/", 1)[1]
            return FakeResponse(body={node: self.object_info[node]} if node in self.object_info else {})
        if path == "/view":
            return FakeResponse(content=self.view_bytes)
        raise AssertionError(f"unexpected GET {path}")

    # -- assertions helpers -----------------------------------------------------------------
    def posted_prompt(self):
        """The workflow dict of the single POST /prompt this fake received."""
        prompts = [payload["prompt"] for path, payload in self.posted if path == "/prompt"]
        assert len(prompts) == 1, f"expected exactly one POST /prompt, got {len(prompts)}"
        return prompts[0]

    def paths(self, method="post"):
        return [p for p, _ in (self.posted if method == "post" else self.gets)]


# -- workflow graph helpers ------------------------------------------------------------------


def refs(wf):
    """Yield (node_id, input_name, [src_node_id, output_index]) for every edge in the graph."""
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                yield nid, key, val


def dangling(wf):
    return [(nid, key, val) for nid, key, val in refs(wf) if val[0] not in wf]


def assert_no_dangling(wf):
    missing = dangling(wf)
    assert not missing, f"inputs reference nodes that are not in the graph: {missing}"


def upstream(wf, nid, seen=None):
    """Every node id reachable by walking input edges backwards from nid."""
    if seen is None:
        seen = set()
    for _, _, (src, _) in (
        (nid, k, v)
        for k, v in wf[nid].get("inputs", {}).items()
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)
    ):
        if src not in seen:
            seen.add(src)
            if src in wf:
                upstream(wf, src, seen)
    return seen
