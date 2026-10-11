"""Offline coordinate-space, observation, and endpoint response regressions."""
import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from osworld_agent.actions import Action
from osworld_agent.adapters.uitars_grounding import RealUItarsGrounding, _extract_coords as tars_coords
from osworld_agent.adapters.uivenus2_grounding import UIVenus2Grounding, _extract_coords as venus_coords
from osworld_agent.adapters.vmware_env import VMwareEnvironment, to_pyautogui
from osworld_agent.adapters.windows_env import WindowsEnvironment
from osworld_agent.env import Observation, OSWorldEnvironment
from osworld_agent.model.grounding import GroundingResult, resolve_coords, screenshot_size


def image_bytes(width=800, height=600):
    buf = BytesIO()
    Image.new("RGB", (width, height)).save(buf, "PNG")
    return buf.getvalue()


def fake_client(content):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=content))]))))


def test_actual_dimensions_from_every_supported_image_format(tmp_path):
    raw = image_bytes()
    path = tmp_path / "shot.png"
    path.write_bytes(raw)
    for shot in (raw, str(path), path, Image.new("RGB", (800, 600)),
                 "data:image/png;base64," + base64.b64encode(raw).decode()):
        assert screenshot_size(shot) == (800, 600)
    assert screenshot_size(b"not an image") is None


@pytest.mark.parametrize("text", ["[-1,-1]", "[-1,30]", "1 2", "step 2: (30, 40)",
                                 "[1,2,3,4]", "(1,2]", "(nan, 1)"])
def test_tars_rejects_ambiguous_or_negative_output(text):
    assert tars_coords(text) is None


@pytest.mark.parametrize("text", ["[1,2]", "(1, 2)", "1, 2"])
def test_tars_accepts_single_point(text):
    assert tars_coords(text) == (1, 2)


def test_tars_uses_actual_bytes_dimensions_by_default():
    model = RealUItarsGrounding("unused", "unused", "unused", width=1920, height=1080)
    model._client = fake_client("(400, 300)")
    assert resolve_coords(model.locate(image_bytes(), "target")) == (400, 300)


def test_tars_retains_explicit_endpoint_coordinate_space():
    model = RealUItarsGrounding("unused", "unused", "unused",
                               grounding_width=1000, grounding_height=1000)
    model._client = fake_client("[500,500]")
    assert resolve_coords(model.locate(image_bytes(), "target")) == (400, 300)


@pytest.mark.parametrize("text", ["[800,0]", "[0,600]", "[-1,-1]", "[2000,2000]"])
def test_tars_outside_actual_image_is_failure(text):
    model = RealUItarsGrounding("unused", "unused", "unused")
    model._client = fake_client(text)
    assert resolve_coords(model.locate(image_bytes(), "target")) is None


def test_tars_requires_complete_positive_protocol_dimensions():
    with pytest.raises(ValueError):
        RealUItarsGrounding("u", "k", "m", grounding_width=1000)
    with pytest.raises(ValueError):
        RealUItarsGrounding("u", "k", "m", grounding_width=0, grounding_height=1000)


@pytest.mark.parametrize("text", ["[-1,-1]", "[-5,0,100,100]", "[10,10,0,0]",
                                 "[1001,0]", "[0,0,1001,1000]", "point: [1,2]"])
def test_venus_rejects_invalid_points_and_boxes(text):
    assert venus_coords(text) is None


def test_venus_bytes_size_and_normalized_edge():
    model = UIVenus2Grounding("unused", "unused", "unused", width=1920, height=1080)
    model._client = fake_client("[500,500]")
    assert resolve_coords(model.locate(image_bytes(), "target")) == (400, 300)
    model._client = fake_client("[1000,1000]")
    assert resolve_coords(model.locate(image_bytes(), "target")) == (799, 599)


def test_logging_wrapper_does_not_change_coordinate_mapping(monkeypatch):
    from osworld_agent.viz import logging
    monkeypatch.setattr(logging, "_post", lambda payload: None)
    model = UIVenus2Grounding("unused", "unused", "unused")
    model._client = fake_client("[500,500]")
    wrapped = logging.LoggingGrounding(model)
    wrapped.width = 9999  # Historical pipeline code may still assign this.
    assert resolve_coords(wrapped.locate(image_bytes(), "target")) == (400, 300)


@pytest.mark.parametrize("result", [GroundingResult(float("nan"), 1),
                                   GroundingResult(float("inf"), 1),
                                   GroundingResult(1, 1, float("nan"))])
def test_resolve_rejects_nonfinite_coordinates(result):
    assert resolve_coords(result) is None


def test_base_observe_never_calls_reset(monkeypatch):
    env = OSWorldEnvironment("unused")
    monkeypatch.setattr(env, "reset", lambda task: pytest.fail("reset must not run"))
    with pytest.raises(NotImplementedError):
        env.observe()


def test_vmware_observe_is_read_only_and_marks_desktop_space():
    env = object.__new__(VMwareEnvironment)
    env._env = SimpleNamespace(_get_obs=lambda: {"screenshot": image_bytes()})
    env._collect_browser_snapshot = lambda: None
    env._focus_is_browser = lambda: False
    env._a11y_text = lambda raw: "desktop"
    env.reset = lambda task: pytest.fail("reset must not run")
    obs = env.observe()
    assert obs.info["screenshot_space"] == "screen_pixels"
    assert obs.info["screenshot_size"] == [800, 600]
    assert obs.info["scene"]["browser_use"] == 0
    assert env.evaluation_available is True


def test_windows_collects_a_single_browser_observation(monkeypatch):
    env = WindowsEnvironment()
    snapshot = {"kind": "browser_ax", "snapshot_id": "s", "url": "https://example.test"}
    metadata = {"screenshot_space": "css_viewport", "screenshot_size": [800, 600],
                "page_id": "p", "document_id": "d", "snapshot_id": "s"}
    calls = []
    def collect():
        calls.append(True)
        return image_bytes(), snapshot, metadata
    env._browser_session = SimpleNamespace(collect_observation=collect)
    monkeypatch.setattr(env, "_focus_is_browser", lambda: pytest.fail("extra CDP probe"))
    monkeypatch.setattr(env, "_take_screenshot", lambda: pytest.fail("extra screenshot"))
    obs = env.observe()
    assert len(calls) == 1
    assert obs.context == [snapshot]
    assert obs.info["screenshot_space"] == "css_viewport"
    assert obs.info["scene"]["snapshot_id"] == "s"


def test_windows_failed_viewport_capture_discards_browser_identity(monkeypatch):
    env = WindowsEnvironment()
    env._browser_session = SimpleNamespace(collect_observation=lambda: (
        None, {"kind": "browser_ax", "snapshot_id": "old"}, {"page_id": "old"}))
    monkeypatch.setattr(env, "_take_screenshot", lambda: image_bytes())
    obs = env.observe()
    assert obs.info["screenshot_space"] == "screen_pixels"
    assert obs.info["scene"]["browser_use"] == 0
    assert "page_id" not in obs.info
    assert obs.context == []


def test_windows_viewport_rejection_is_not_replayed_on_desktop(monkeypatch):
    env = WindowsEnvironment()
    calls = []
    def execute(action, metadata):
        calls.append((action.action, metadata))
        return {"status": "rejected", "dispatched": False, "error": "unsupported"}
    env._browser_session = SimpleNamespace(execute_viewport_action=execute)
    monkeypatch.setattr(env, "observe", lambda: Observation())
    monkeypatch.setattr(env, "_focus_is_browser", lambda: pytest.fail("focus must not determine space"))
    action = Action("select", {"x": 10, "y": 20, "option": "a", "coordinate_space": "css_viewport",
                               "observation_metadata": {"page_id": "seen"}})
    result = env.step(action)
    assert calls == [("select", {"page_id": "seen"})]
    assert result.info["error"] == "unsupported"
    assert result.info["dispatched"] is False


def test_windows_missing_coordinate_space_is_rejected(monkeypatch):
    env = WindowsEnvironment()
    monkeypatch.setattr(env, "observe", lambda: Observation())
    result = env.step(Action("click", {"x": 10, "y": 20}))
    assert result.info["status"] == "rejected"
    assert result.info["dispatched"] is False


def test_windows_explicit_screen_coordinates_do_not_use_an_open_browser(monkeypatch):
    from osworld_agent.adapters import windows_env
    env = WindowsEnvironment()
    env._browser_session = SimpleNamespace(execute_viewport_action=lambda *a, **kw: pytest.fail("wrong space"))
    monkeypatch.setattr(env, "observe", lambda: Observation())
    monkeypatch.setattr(env, "_focus_is_browser", lambda: pytest.fail("focus probe"))
    monkeypatch.setattr(windows_env, "to_pyautogui", lambda action: "pass")
    result = env.step(Action("click", {"x": 10, "y": 20, "coordinate_space": "screen_pixels"}))
    assert result.info["channel"] == "desktop"
    assert result.info["dispatched"] is True


def test_windows_execution_error_is_reported(monkeypatch):
    from osworld_agent.adapters import windows_env
    env = WindowsEnvironment()
    monkeypatch.setattr(env, "_build_observation", lambda: Observation())
    monkeypatch.setattr(windows_env, "to_pyautogui", lambda action: "raise ValueError('execution failed')")
    result = env.step(Action("click", {"x": 10, "y": 20, "coordinate_space": "screen_pixels"}))
    assert result.info["status"] == "failed"
    assert "execution failed" in result.info["error"]


@pytest.mark.parametrize("action", [Action("click", {"x": 5, "y": 6, "coordinate_space": "css_viewport"}),
                                    Action("type", {"x": -1, "y": 20, "text": "oops"}),
                                    Action("drag_and_drop", {"x1": 1, "y1": 1, "x2": -1, "y2": 2}),
                                    Action("select", {"option": "a"})])
def test_invalid_or_viewport_coordinates_never_become_pyautogui(action):
    assert to_pyautogui(action) == ""


def test_visual_logging_does_not_block_on_a_stalled_endpoint(monkeypatch):
    import queue
    import threading
    from osworld_agent.viz import logging
    started, release = threading.Event(), threading.Event()
    def post(*args, **kwargs):
        started.set()
        release.wait(2)
    monkeypatch.setattr(logging, "_session", SimpleNamespace(post=post))
    monkeypatch.setattr(logging, "_log_queue", queue.Queue(maxsize=1))
    monkeypatch.setattr(logging, "_log_worker", None)
    monkeypatch.setattr(logging, "_dropped_logs", 0)
    try:
        logging._post({"event": "first"})
        assert started.wait(1)
        logging._post({"event": "second"})
        logging._post({"event": "third"})
        assert logging._dropped_logs == 1
        assert logging.flush_logs(timeout=0) is False
    finally:
        release.set()
    assert logging.flush_logs(timeout=1) is True
