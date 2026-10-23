"""gpu_topology: which card does what on a two-card machine. Everything runs against a made-up nvidia-smi listing."""

import os

import pytest
import requests

import gpu_topology as gt
import h3_idea

TWO = "0, NVIDIA GeForce RTX 5060 Ti, GPU-aaaa, 16311, 488\n1, NVIDIA GeForce RTX 2070, GPU-bbbb, 8192, 0\n"


THREE = TWO + "2, NVIDIA GeForce RTX 3060, GPU-dddd, 12288, 0\n"


def up(url, timeout=None):
    return type("R", (), {"status_code": 200})()


def down(url, timeout=None):
    raise requests.ConnectionError("down")


class Out:
    def __init__(self, stdout):
        self.stdout = stdout


def listing(text=TWO):
    return lambda cmd, **kw: Out(text)


def gpus(text=TWO):
    return gt.detect(listing(text))


@pytest.fixture
def real_topology(monkeypatch):
    """Undo conftest's GPU_SPLIT=off and feed the cache from a made-up listing."""

    def use(text=TWO, **env):
        monkeypatch.delenv("GPU_SPLIT", raising=False)
        for key in ("IMAGE_GPU", "LLM_GPU"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        gt.reset()
        gt.plan(run=listing(text))

    return use


def test_detect_reads_every_card_and_skips_lines_it_cannot_read():
    found = gpus(TWO + "garbage\n2, only, three\n")
    assert [(g.index, g.label, g.uuid) for g in found] == [(0, "RTX 5060 Ti", "GPU-aaaa"), (1, "RTX 2070", "GPU-bbbb")]
    assert found[1].free_mib == 8192 and found[0].free_mib == 16311 - 488


def test_detect_returns_nothing_when_nvidia_smi_is_missing():
    def boom(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")

    assert gt.detect(boom) == []


def test_the_big_card_renders_and_the_other_one_hosts_qwen():
    plan = gt.make_plan(gpus(), env={})
    assert plan.split and plan.image.label == "RTX 5060 Ti" and plan.llm.label == "RTX 2070"
    assert "圖片／影片 → RTX 5060 Ti" in plan.reason and "Qwen → RTX 2070" in plan.reason


def test_the_order_nvidia_smi_lists_them_in_does_not_matter():
    reversed_listing = (
        "0, NVIDIA GeForce RTX 2070, GPU-bbbb, 8192, 0\n1, NVIDIA GeForce RTX 5060 Ti, GPU-aaaa, 16311, 0\n"
    )
    plan = gt.make_plan(gpus(reversed_listing), env={})
    assert plan.image.uuid == "GPU-aaaa" and plan.llm.uuid == "GPU-bbbb"


def test_overrides_pick_a_card_by_index_uuid_or_part_of_the_name():
    assert gt.make_plan(gpus(), env={"IMAGE_GPU": "2070"}).image.uuid == "GPU-bbbb"
    assert gt.make_plan(gpus(), env={"IMAGE_GPU": "GPU-bbbb"}).llm.uuid == "GPU-aaaa"
    assert gt.make_plan(gpus(THREE), env={}).llm.uuid == "GPU-dddd"  # of the others, the biggest
    assert gt.make_plan(gpus(THREE), env={"LLM_GPU": "2070"}).llm.uuid == "GPU-bbbb"
    assert (
        gt.make_plan(gpus(THREE), env={"LLM_GPU": "5060"}).llm.uuid == "GPU-dddd"
    )  # the picture card is never the Qwen card
    plan = gt.make_plan(gpus(), env={"IMAGE_GPU": "nonsense"})
    assert plan.image.uuid == "GPU-aaaa"  # an override that matches nothing falls back to the automatic choice


def test_one_card_means_no_split():
    plan = gt.make_plan(gpus("0, NVIDIA GeForce RTX 5060 Ti, GPU-aaaa, 16311, 0\n"), env={})
    assert plan.split is False and plan.llm is plan.image and "只有一張卡" in plan.reason


def test_a_second_card_too_small_for_qwen_is_not_used():
    small = "0, NVIDIA GeForce RTX 5060 Ti, GPU-aaaa, 16311, 0\n1, NVIDIA GeForce GTX 1060, GPU-cccc, 6144, 0\n"
    plan = gt.make_plan(gpus(small), env={})
    assert plan.split is False and "放不下 Qwen" in plan.reason


@pytest.mark.parametrize("value", ["off", "0", "false", "NO"])
def test_gpu_split_off_pins_nothing(value):
    assert gt.make_plan(gpus(), env={"GPU_SPLIT": value}) is None
    assert gt.make_plan([], env={}) is None


def test_the_processes_are_pinned_by_uuid_in_pci_order(real_topology):
    real_topology()
    assert gt.image_env() == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "GPU-aaaa"}
    assert gt.llm_env() == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "GPU-bbbb",
        "OLLAMA_VULKAN": "0",
    }  # Vulkan would still show the picture card
    assert gt.smi_command() == ["nvidia-smi", "-i", "0"]
    assert gt.planned_split() and "RTX 2070" in gt.describe()


def test_nothing_is_pinned_when_the_split_is_off():
    assert gt.image_env() == {} and gt.llm_env() == {} and gt.smi_command() == ["nvidia-smi"]
    assert gt.planned_split() is False and "沒有指定" in gt.describe()


def test_a_single_card_still_pins_the_picture_process_but_has_no_qwen_environment(real_topology):
    real_topology("0, NVIDIA GeForce RTX 5060 Ti, GPU-aaaa, 16311, 0\n")
    assert gt.image_env()["CUDA_VISIBLE_DEVICES"] == "GPU-aaaa" and gt.llm_env() == {}


def test_effective_split_needs_the_pinned_server_or_room_for_it(real_topology):
    real_topology()
    assert gt.effective_split(up) is True  # the pinned server answers: it is the one holding the card
    assert gt.effective_split(down) is True  # not running, but the 2070 has 8192 MiB free
    real_topology(
        "0, NVIDIA GeForce RTX 5060 Ti, GPU-aaaa, 16311, 0\n1, NVIDIA GeForce RTX 2070, GPU-bbbb, 8192, 3000\n"
    )
    assert gt.effective_split(down) is False  # something else holds 3 GB of it: back to sharing the picture card
    assert gt.effective_split(up) is True  # ...unless the pinned server is the one that holds it


def test_keep_alive_is_long_on_its_own_card_and_zero_when_shared(real_topology):
    real_topology()
    assert gt.llm_keep_alive(down) == gt.KEEP_ALIVE_SPLIT  # not running yet, the card is free: Qwen will stay loaded
    real_topology(
        "0, NVIDIA GeForce RTX 5060 Ti, GPU-aaaa, 16311, 0\n1, NVIDIA GeForce RTX 2070, GPU-bbbb, 8192, 3000\n"
    )
    assert gt.llm_keep_alive(down) == 0  # the second card is occupied: Qwen is unloaded after every call as before


def test_keep_alive_is_zero_without_a_split():
    assert gt.llm_keep_alive() == 0


# --- the pinned Ollama server (h3_idea) --------------------------------------------------------------------------------


class Resp:
    status_code = 200

    def json(self):
        return {}


class Proc:
    def __init__(self):
        self.stopped = False

    def poll(self):
        return None

    def terminate(self):
        self.stopped = True


def manifest_folder(tmp_path):
    folder = tmp_path / "models"
    manifest = folder / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5"
    manifest.mkdir(parents=True)
    (manifest / "9b").write_text("{}")
    return folder


def test_with_a_second_card_only_the_pinned_server_is_looked_for(real_topology):
    real_topology()
    asked = []

    def get(url, timeout=None):
        asked.append(url)
        raise requests.ConnectionError("down")

    with pytest.raises(h3_idea.IdeaError):
        h3_idea.find_ollama(get)
    assert {u.removesuffix("/api/tags") for u in asked} == {
        f"http://127.0.0.1:{gt.PINNED_PORT}"
    }  # the shared 21434 / 11434 servers are never tried
    assert h3_idea.own_url() == f"http://127.0.0.1:{gt.PINNED_PORT}"


def test_without_a_split_the_shared_servers_are_used_as_before():
    asked = []

    def get(url, timeout=None):
        asked.append(url)
        raise requests.ConnectionError("down")

    with pytest.raises(h3_idea.IdeaError):
        h3_idea.find_ollama(get)
    assert [u.removesuffix("/api/tags") for u in asked] == list(h3_idea.URL_CANDIDATES) and h3_idea.own_url().endswith(
        str(h3_idea.OWN_PORT)
    )


def test_the_pinned_server_sees_only_the_qwen_card_and_outlives_the_call(real_topology, tmp_path, monkeypatch):
    real_topology()
    h3_idea.stop_pinned()
    monkeypatch.setattr(h3_idea.shutil, "which", lambda name: "ollama.exe")
    seen = {}
    proc = Proc()

    def popen(cmd, env=None, **kw):
        seen["cmd"], seen["env"] = cmd, env
        return proc

    folder = manifest_folder(tmp_path)
    get = lambda url, timeout=None: Resp()  # noqa: E731
    handle = h3_idea.start_own_ollama("qwen3.5:9b", popen=popen, get=get, folders=[str(folder)])
    env = seen["env"]
    assert env["OLLAMA_HOST"] == f"127.0.0.1:{gt.PINNED_PORT}" and env["CUDA_VISIBLE_DEVICES"] == "GPU-bbbb"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and env["OLLAMA_VULKAN"] == "0"
    handle.terminate()  # the callers' finally-block: the server must stay up between chat turns
    assert proc.stopped is False
    again = h3_idea.start_own_ollama(
        "qwen3.5:9b", popen=lambda *a, **k: pytest.fail("started twice"), get=get, folders=[str(folder)]
    )
    assert again is handle
    h3_idea.stop_pinned()
    assert proc.stopped is True


def test_the_shared_server_is_still_stopped_by_its_caller(tmp_path, monkeypatch):
    monkeypatch.setattr(h3_idea.shutil, "which", lambda name: "ollama.exe")
    seen = {}

    def popen(cmd, env=None, **kw):
        seen["env"] = env
        return Proc()

    proc = h3_idea.start_own_ollama(
        "qwen3.5:9b", popen=popen, get=lambda u, timeout=None: Resp(), folders=[str(manifest_folder(tmp_path))]
    )
    assert isinstance(proc, Proc) and seen["env"]["OLLAMA_HOST"].endswith(str(h3_idea.OWN_PORT))
    assert seen["env"].get("CUDA_VISIBLE_DEVICES") == os.environ.get("CUDA_VISIBLE_DEVICES")  # nothing pinned
