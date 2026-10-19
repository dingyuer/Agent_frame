"""Studio edits must reach only the selected decision surface, not judges."""
import json

import pytest

from osworld_agent.decision_context import action_scope, system_sections
from osworld_agent.decision_modes import DecisionMode
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.prompt_settings import load_mode_prompts, prompt_cli_args, validate_mode_prompts
from osworld_agent.prompts import build_editable_mode_prompts
from test_mode_prompts import browser_observation, desktop_observation
from test_pipeline_completion import Env, Model


def test_saved_templates_reach_real_pipeline_inputs_and_switch_with_observation(tmp_path):
    templates = {'dom': 'CUSTOM DOM POLICY', 'gui_browser': 'CUSTOM BROWSER POLICY',
                 'gui_desktop': 'CUSTOM DESKTOP POLICY'}
    args = prompt_cli_args(templates, tmp_path)
    assert args[0] == '--mode-prompts-file'
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='legacy',
                    mode_prompts=load_mode_prompts(args[1]))
    assert pipe.run_manifest['decision_prompt_overrides'] == templates
    for mode, obs in [('dom', browser_observation()),
                      ('gui_browser', browser_observation(mode='browser_gui')),
                      ('gui_desktop', desktop_observation())]:
        if mode == 'gui_browser':
            obs.info['scene']['structure_available'] = False
        state = StepState(task='搜索论文 Alpha 并下载', obs=obs)
        messages = pipe.build_input(state)
        system = messages[0]['content']
        assert templates[mode] in system
        for other in templates.keys() - {mode}:
            assert templates[other] not in system
        assert '本轮动作' in system and '论文' in system
        assert 'custom_' + mode in state.prompt_metadata['prompt_rules']
        assert 'download' not in pipe._active_actions
        assert 'run_in_terminal' not in pipe._active_actions


def test_missing_override_keeps_default_and_dialog_recovery_cannot_be_replaced():
    profiles = [DecisionMode('dom', 'browser', 'test'), DecisionMode('dialog', 'browser', 'test'),
                DecisionMode('observe', 'desktop', 'test')]
    for profile in profiles:
        kwargs = dict(profile=profile, task='打开网页')
        names = action_scope(None, None, profile=profile)
        assert system_sections(names, None, **kwargs) == system_sections(
            names, None, mode_prompts={'gui_desktop': 'CUSTOM'}, **kwargs)


def test_editor_shows_policy_bodies_without_static_action_menus():
    templates = build_editable_mode_prompts(os_name='windows', browser_only=True)
    assert set(templates) == {'dom', 'gui_browser', 'gui_desktop', 'cli'}
    assert 'target_ref' in templates['dom']
    assert '当前截图是桌面' in templates['gui_desktop']
    assert '资源管理器' in templates['gui_desktop']
    assert '当前截图是浏览器网页视口' in templates['gui_browser']
    assert all('本轮动作（type / 必填参数）' not in body for body in templates.values())


@pytest.mark.parametrize('value', [[], {'invalid':'text'}, {'dom':''}, {'dom':False}, {'dom':'x'*40001}])
def test_invalid_templates_fail_before_start(value, tmp_path):
    with pytest.raises(ValueError):
        prompt_cli_args(value, tmp_path)
    assert not list(tmp_path.iterdir())


def test_unedited_run_does_not_write_configuration(tmp_path):
    assert prompt_cli_args({}, tmp_path) == []
    assert load_mode_prompts(None) == {}
    assert not list(tmp_path.iterdir())


def test_agent_passes_mode_configuration_to_pipeline(monkeypatch):
    import osworld_agent.agent as module
    captured = {}
    class Runner:
        decision_judge = None
        def __init__(self, **kwargs): captured.update(kwargs)
        def run(self, task): return task
    monkeypatch.setattr(module, 'Pipeline', Runner)
    agent = module.Agent(Model([]), None, Env(), mode_prompts={'dom':'custom policy'})
    assert agent.run('demo') == 'demo'
    assert captured['mode_prompts'] == {'dom':'custom policy'}


def test_server_launch_args_preserve_all_templates_without_real_process(tmp_path, monkeypatch):
    import osworld_agent.viz.server as server
    import osworld_agent.persistent_chrome as chrome
    monkeypatch.setattr(server, 'LOCAL_RUNS_DIR', str(tmp_path))
    settings = {'browser_session_mode':'isolated', 'browser_profile_dir':'', 'effective_profile_dir':''}
    monkeypatch.setattr(chrome, 'browser_settings', lambda: settings)
    monkeypatch.setattr(chrome, 'save_browser_settings', lambda *args: settings)
    monkeypatch.setattr(server, '_wait_cli_args', lambda cfg: [])
    captured = {}
    class CapturedLaunch(Exception): pass
    def popen(cmd, **kwargs):
        captured['cmd'] = cmd
        raise CapturedLaunch('no real process')
    monkeypatch.setattr(server.subprocess, 'Popen', popen)
    previous = server._agent_state.copy()
    try:
        (tmp_path / 'workspace').mkdir()
        result = server._start_local({'task':'test', 'mode_prompts':{'dom':'unique test policy'},
                             'workspace_dir': str(tmp_path / 'workspace'),
                             'result_dir':str(tmp_path / 'run')}, 8088)
        assert 'cmd' in captured, result
        cmd = captured['cmd']
        loaded = load_mode_prompts(cmd[cmd.index('--mode-prompts-file')+1])
        assert loaded == {'dom':'unique test policy'}
    finally:
        server._agent_state.clear()
        server._agent_state.update(previous)
