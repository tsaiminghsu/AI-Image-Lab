"""comfyui_client.default_comfyui_url: Windows reserves ranges of ports after some restarts (8139-8338 on 2026-10-20), so
ComfyUI could not bind 8188 and every GUI job failed with "ComfyUI process exited early". The default URL now moves to
the first usable port; COMFYUI_URL still wins when set."""

import socket

import comfyui_client as client


def probe_from(states):
    """A probe that answers from a {port: state} table and records which ports were asked."""
    asked = []

    def probe(port):
        asked.append(port)
        return states.get(port, "blocked")

    probe.asked = asked
    return probe


def test_the_usual_port_is_used_when_it_is_free_or_a_comfyui_already_runs_there():
    for state in ("free", "up"):
        assert client.default_comfyui_url(probe_from({8188: state}), env={}) == "http://127.0.0.1:8188"


def test_a_system_reserved_port_is_skipped():
    probe = probe_from({8188: "blocked", 18188: "free"})
    assert client.default_comfyui_url(probe, env={}) == "http://127.0.0.1:18188"
    assert probe.asked == [8188, 18188]  # stops at the first usable one


def test_a_comfyui_already_running_on_the_fallback_port_is_reused_by_the_next_process():
    """The GUI started ComfyUI on 18188; a CLI run later must find it there, not pick 18288."""
    probe = probe_from({8188: "blocked", 18188: "up", 18288: "free"})
    assert client.default_comfyui_url(probe, env={}) == "http://127.0.0.1:18188"


def test_the_environment_variable_wins_and_nothing_is_probed():
    probe = probe_from({})
    assert (
        client.default_comfyui_url(probe, env={"COMFYUI_URL": "https://pod-8188.proxy.runpod.net"})
        == "https://pod-8188.proxy.runpod.net"
    )
    assert probe.asked == []


def test_when_no_port_is_usable_it_falls_back_to_the_usual_one_so_the_error_names_it():
    assert client.default_comfyui_url(probe_from({}), env={}) == "http://127.0.0.1:8188"


def test_the_real_probe_tells_free_up_and_taken_apart():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert client.probe_port(port) == "up"
    assert client.probe_port(port) in ("free", "blocked")  # closed again: bindable unless the OS holds it for a moment
    assert client.server_port() == int(
        client.COMFYUI_URL.rsplit(":", 1)[1]
    )  # start_server listens where the client talks
