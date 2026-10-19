"""Workspace provenance and virtual-desktop mapping without native Windows input."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from osworld_agent import desktop_capture
from osworld_agent.actions import Action
from osworld_agent.adapters.vmware_env import to_pyautogui
from osworld_agent.adapters.windows_env import WindowsEnvironment
from osworld_agent.browser_feedback import workspace_facts
from osworld_agent.browser_session import BrowserSession
from osworld_agent.env import Observation
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.verifiers import CompletionVerifier
from test_browser_lifecycle import EventBrowser, session
from test_browser_operational_feedback import pipe
from test_execution_queue_review import Reviewer, PASSED
from test_pipeline_completion import observation


@pytest.fixture(autouse=True)
def no_viz(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)


def test_workspace_allocated_before_browser_and_kept_through_episode(session):
    browser = EventBrowser()
    session._connect = Mock(return_value=(None, browser))
    first = session.prepare_workspace('task')
    assert Path(first['workspace_dir']).is_dir()
    session._connect.assert_not_called()
    assert session.prepare_workspace('task') == first
    begun = session.begin_episode('task')
    assert begun['workspace_dir'] == first['workspace_dir'] == begun['downloads_dir']
    assert session.event_metadata()['initial_state']['workspace_dir'] == first['workspace_dir']
    marker = Path(first['workspace_dir']) / 'old.pdf'
    marker.write_bytes(b'old')
    second = session.begin_episode('task')
    assert second['workspace_dir'] == first['workspace_dir']
    assert second['episode_id'] == first['episode_id']
    assert marker.read_bytes() == b'old' and not session.event_metadata()['artifacts']


def test_workspace_config_environment_and_legacy_alias(tmp_path, monkeypatch):
    monkeypatch.setenv('OSWORLD_WORKSPACE_DIR', str(tmp_path / '默认目录'))
    default = BrowserSession()
    explicit = BrowserSession(workspace_dir=tmp_path / 'explicit')
    legacy = BrowserSession(downloads_dir=tmp_path / 'legacy')
    assert default._base_downloads_dir == str(tmp_path / '默认目录')
    assert explicit._base_downloads_dir == str(tmp_path / 'explicit')
    assert legacy._base_downloads_dir == str(tmp_path / 'legacy')
    for item in (default, explicit, legacy):
        item.close()
    with pytest.raises(ValueError):
        BrowserSession(workspace_dir=tmp_path, downloads_dir=tmp_path)


def test_planner_and_decision_both_receive_fixed_workspace(tmp_path):
    p = pipe()
    p.completion = CompletionVerifier('查看网页')
    p.env.initial_state = {'workspace_dir': str(tmp_path), 'workspace_location': 'target_desktop'}
    state = StepState(task='查看网页', memory=StructuredMemory())
    assert p._plan(state.task, state)
    assert str(tmp_path) in json.dumps(p.model.calls, ensure_ascii=False)
    state.obs = observation()
    state.obs.info['initial_state'] = copy.deepcopy(p.env.initial_state)
    p.build_input(state)
    assert str(tmp_path) in state.prompt_text and '默认操作目录' in state.prompt_text
    assert '只有原始任务明确指定' in state.prompt_text


def test_remote_workspace_is_not_presented_as_a_guest_path():
    obs = Observation(info={'initial_state': {'workspace_dir': '/host/workspace',
                                            'workspace_location': 'controller_host'}})
    assert '不要把控制器路径输入远程文件管理器' in '\n'.join(workspace_facts(obs))


def geometry(left=-1920, top=-200):
    return {'left': left, 'top': top, 'width': 4480, 'height': 1640}


def captured():
    return {'virtual_desktop': geometry(), 'screen_origin': [-1920, -200],
            'desktop_capture_id': 'capture-1', 'screenshot_size': [4480, 1640]}


def test_capture_includes_all_screens_and_negative_origin(monkeypatch):
    from PIL import ImageGrab
    grab = Mock(return_value=Image.new('RGB', (4480, 1640)))
    monkeypatch.setattr(ImageGrab, 'grab', grab)
    monkeypatch.setattr(desktop_capture, 'virtual_geometry', geometry)
    shot, metadata = desktop_capture.capture_virtual_desktop()
    grab.assert_called_once_with(all_screens=True)
    assert shot.size == (4480, 1640) and metadata['screen_origin'] == [-1920, -200]


def test_capture_rejects_monitor_change_and_wrong_dpi_size(monkeypatch):
    from PIL import ImageGrab
    monkeypatch.setattr(ImageGrab, 'grab', lambda **_: Image.new('RGB', (4480, 1640)))
    monkeypatch.setattr(desktop_capture, 'virtual_geometry', Mock(side_effect=[geometry(), geometry(0)]))
    with pytest.raises(OSError):
        desktop_capture.capture_virtual_desktop()
    monkeypatch.setattr(desktop_capture, 'virtual_geometry', geometry)
    monkeypatch.setattr(ImageGrab, 'grab', lambda **_: Image.new('RGB', (1920, 1080)))
    with pytest.raises(OSError):
        desktop_capture.capture_virtual_desktop()


@pytest.mark.parametrize('name,coords,expected', [
    ('click', {'x': 100, 'y': 100}, 'click(-1820, -100)'),
    ('double_click', {'x': 2000, 'y': 300}, 'doubleClick(80, 100)'),
    ('type', {'x': 10, 'y': 20, 'text': 'data'}, 'click(-1910, -180)'),
    ('drag_and_drop', {'x1': 100, 'y1': 100, 'x2': 2000, 'y2': 300}, 'dragTo(80, 100'),
])
def test_image_local_coordinates_map_once_to_physical_desktop(name, coords, expected):
    action = Action(name, dict(coords, coordinate_space='screen_pixels'))
    original = copy.deepcopy(action.args)
    mapped, origin = desktop_capture.validate_desktop_action(action, captured(), geometry())
    command = to_pyautogui(mapped, desktop_origin=origin)
    assert expected in command
    assert action.args == original


@pytest.mark.parametrize('coords', [{'x': -1, 'y': 1}, {'x': 4480, 'y': 1}, {'x': 1, 'y': 1640},
                                    {'x': float('nan'), 'y': 1}, {'x': True, 'y': 1}])
def test_invalid_image_coordinates_never_reach_native_input(coords):
    with pytest.raises(ValueError):
        desktop_capture.validate_desktop_action(Action('click', coords), captured(), geometry())


def test_monitor_change_or_old_capture_rejects_before_dispatch():
    with pytest.raises(ValueError):
        desktop_capture.validate_desktop_action(Action('click', {'x': 100, 'y': 100}), captured(), geometry(0))
    with pytest.raises(ValueError):
        desktop_capture.validate_desktop_action(Action('click', {'x': 100, 'y': 100,
            'observation_metadata': {'desktop_capture_id': 'old'}}), captured(), geometry())


def test_windows_adapter_uses_capture_origin_not_model_supplied_origin(monkeypatch):
    env = WindowsEnvironment()
    env._observation_surface = 'desktop'
    env._desktop_capture = captured()
    monkeypatch.setattr(env, 'observe', lambda: Observation())
    monkeypatch.setattr(desktop_capture, 'virtual_geometry', geometry)
    gui = SimpleNamespace(click=Mock())
    monkeypatch.setitem(sys.modules, 'pyautogui', gui)
    obs = Observation(info={'screenshot_space': 'screen_pixels', **captured()})
    action = Action('click', {'x': 100, 'y': 100, 'screen_origin': [999, 999]})
    Pipeline._attach_observation_space(action, obs)
    result = env.step(action)
    assert result.info['dispatched'] is True
    gui.click.assert_called_once_with(-1820, -100)
    monkeypatch.setattr(desktop_capture, 'virtual_geometry', lambda: geometry(0))
    result = env.step(action)
    assert not result.info['dispatched'] and gui.click.call_count == 1


def test_final_judge_gets_planner_hint_and_unresolved_requirements_not_self_report():
    p = pipe(verifier_model=Reviewer([PASSED]))
    p.completion = CompletionVerifier('整理错题', {'checks': [{'kind': 'unresolved_requirement', 'text': '错题按章节归档'}]})
    state = StepState(task='整理错题，按章节归档', obs=observation(), memory=StructuredMemory())
    state.memory.plan = {'final_check': '核对章节分组与每题归属'}
    ok, _ = p._verify_finish(state, '自报已经全部完成')
    assert ok
    messages = json.dumps(p.reviewer.model.calls, ensure_ascii=False)
    assert '核对章节分组与每题归属' in messages and '错题按章节归档' in messages
    assert '自报已经全部完成' not in messages
    assert '不能删减或放宽原始任务' in messages


def test_final_judge_cannot_override_a_missing_download(tmp_path):
    reviewer = Reviewer([PASSED])
    p = pipe(verifier_model=reviewer)
    p.completion = CompletionVerifier('下载文件并整理', {'checks': [
        {'kind': 'download', 'url': 'https://example.test/file.pdf'},
        {'kind': 'unresolved_requirement', 'text': '整理内容'}]})
    state = StepState(task='下载文件并整理', obs=observation(), memory=StructuredMemory())
    assert not p._verify_finish(state, '完成')[0]
    assert not reviewer.calls


def test_windows_unicode_typing_sends_paths_without_clipboard(monkeypatch):
    from osworld_agent import windows_text
    text = 'C:\\工作区\\论文😀.pdf'
    sent = []
    def send(count, inputs, size):
        assert size == (40 if windows_text.ctypes.sizeof(windows_text.ctypes.c_void_p) == 8 else 28)
        sent.extend((event.type, event.keyboard.vk, event.keyboard.scan, event.keyboard.flags) for event in inputs)
        return count
    transport = Mock(side_effect=send)
    monkeypatch.setattr(windows_text.ctypes, 'WinDLL', lambda *a, **k: SimpleNamespace(SendInput=transport), raising=False)
    fail_safe = Mock()
    monkeypatch.setitem(sys.modules, 'pyautogui', SimpleNamespace(failSafeCheck=fail_safe))
    windows_text.type_unicode(text)
    recovered = b''.join(scan.to_bytes(2, 'little') for _, _, scan, flags in sent if flags == 4).decode('utf-16-le')
    assert recovered == text and all(kind == 1 and vk == 0 for kind, vk, _, _ in sent)
    assert all(sent[i][3] == 4 and sent[i + 1][3] == 6 for i in range(0, len(sent), 2))
    fail_safe.assert_called_once()
    transport.side_effect = lambda *a: 0
    with pytest.raises(OSError):
        windows_text.type_unicode(text)


def test_thread_dpi_context_is_restored_even_on_input_failure(monkeypatch):
    import ctypes
    setter = Mock(return_value=123)
    with monkeypatch.context() as patch:
        patch.setattr(desktop_capture.sys, 'platform', 'win32')
        patch.setattr(ctypes, 'WinDLL', lambda *a, **k: SimpleNamespace(SetThreadDpiAwarenessContext=setter), raising=False)
        with pytest.raises(RuntimeError):
            with desktop_capture.physical_dpi_context():
                raise RuntimeError('input failed')
    assert setter.call_count == 2
    assert setter.call_args.args == (123,)
