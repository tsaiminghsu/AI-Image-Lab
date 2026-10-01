"""Where jobs run. The core asks a provider two things only: is a worker alive, and what should a person do
to get one. Phase 1 has a single provider - a notebook on the compute runtime that someone opens and runs.
A local GPU or a rented one is another class here later; nothing else changes, and there is no router yet.
"""

import time

WORKER_STATE_FILE = "logs/worker_state.json"


class ComputeProvider:
    name = "base"

    def worker_state(self, storage, now=None):
        raise NotImplementedError

    def start_hint(self):
        raise NotImplementedError

    def serve_hint(self):
        raise NotImplementedError


class ColabNotebookProvider(ComputeProvider):
    """The worker is a loop inside a notebook. It writes logs/worker_state.json every few seconds; a state
    older than stale_seconds means nobody is serving the queue. The age compares the runtime's clock with the
    local one and includes the sync delay of the folder, so the threshold is generous."""

    name = "colab-notebook"

    def __init__(self, config):
        self.notebook = config.get("compute", {}).get("worker_notebook", "notebooks/01_comfyui.ipynb")
        self.stale_seconds = config.get("worker", {}).get("stale_worker_seconds", 180)

    def worker_state(self, storage, now=None):
        state = storage.read_json(WORKER_STATE_FILE)
        if not isinstance(state, dict) or "updated_epoch" not in state:
            return {"alive": False, "serving": False, "state": None}
        age = (time.time() if now is None else now) - float(state["updated_epoch"])
        ended = state.get("worker_state") in ("stopped", "failed")
        return {
            "alive": not ended and age <= self.stale_seconds,
            "state": state.get("worker_state"),
            "serving": bool(state.get("serving")) and not ended and age <= self.stale_seconds,
            "age_seconds": round(age, 1),
            "session": state.get("session"),
            "gpu": (state.get("gpu") or {}).get("name"),
            "vram_gib": (state.get("gpu") or {}).get("vram_gib"),
            "current_job": state.get("current_job"),
            "waiting": state.get("waiting") or {},
        }

    def serve_hint(self):
        return (
            "A runtime is open but it is not taking jobs from the queue. In its notebook, run the cell that "
            "waits for jobs (the last cell of %s)." % self.notebook
        )

    def start_hint(self):
        return (
            "No worker is serving the queue. In Google Drive open AI-Workflow/%s with Google Colab and choose "
            "Runtime > Run all; the job will be picked up from jobs/pending." % self.notebook
        )


PROVIDERS = {ColabNotebookProvider.name: ColabNotebookProvider}


def make_provider(config):
    return PROVIDERS[config.get("compute", {}).get("provider", ColabNotebookProvider.name)](config)
