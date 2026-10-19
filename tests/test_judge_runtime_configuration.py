"""Runtime selection is explicit without changing the legacy judge API defaults."""
import argparse
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from osworld_agent.viz.server import _judge_cli_args
from osworld_agent.model.decision_judge import DecisionJudgeConfig, add_judge_arguments, judge_config_from_args


@pytest.mark.parametrize('policy', ['threshold', 'maximum_probability'])
def test_studio_transmits_explicit_label_policy_without_mutating_configuration(policy):
    original = {'url':'http://judge.invalid/v1', 'selection_policy':policy}
    args = _judge_cli_args({'decision_judge':original})
    assert args[args.index('--judge-selection-policy') + 1] == policy
    assert original['selection_policy'] == policy
    assert not any('api_key' in item for item in args)


@pytest.mark.parametrize('policy', ['invalid', None, [], False])
def test_studio_rejects_invalid_label_policy(policy):
    with pytest.raises(ValueError, match='标签选择'):
        _judge_cli_args({'decision_judge':{'url':'http://judge.invalid/v1', 'selection_policy':policy}})


def test_legacy_base_cli_roundtrip_and_api_configuration_stay_compatible():
    parser = argparse.ArgumentParser()
    add_judge_arguments(parser)
    flags = _judge_cli_args({'decision_judge':{'url':'http://judge.invalid/v1', 'recheck_unknown':False}})
    config = judge_config_from_args(parser.parse_args(flags))
    assert config['recheck_unknown'] is False
    assert config['threshold'] == .9
    assert 'selection_policy' not in config
    assert DecisionJudgeConfig(**config).threshold == .9


@pytest.mark.parametrize('policy', [None, 'threshold', 'maximum_probability'])
def test_local_runner_passes_runtime_label_policy_to_agent(tmp_path, monkeypatch, policy):
    from osworld_agent import persistent_chrome
    from osworld_agent import run_windows_local as runner
    from test_incognito_lifecycle import _stub_main
    monkeypatch.setattr(persistent_chrome, '_settings_path', lambda:tmp_path / 'browser-settings.json')
    _stub_main(monkeypatch, str(tmp_path / 'owned profile'))
    sys.argv += ['--judge-url','http://judge.invalid/v1', '--result-dir',str(tmp_path / 'logs')]
    if policy:
        sys.argv += ['--judge-selection-policy', policy]
    monkeypatch.setattr(runner.time, 'sleep', lambda _:None)
    original = runner.Agent
    captured = []
    def build_agent(**kwargs):
        captured.append(kwargs['decision_judge'])
        return original(**kwargs)
    monkeypatch.setattr(runner, 'Agent', build_agent)
    assert runner.main() == 0
    assert captured[0]['selection_policy'] == (policy or 'maximum_probability')
    assert captured[0]['url'] == 'http://judge.invalid/v1'
    assert captured[0]['threshold'] == .9
