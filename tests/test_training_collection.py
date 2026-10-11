"""Training collection invariants with deterministic local fakes only."""
import base64
import copy
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from osworld_agent.training_events import TrainingEventWriter, read_events
from osworld_agent.trajectory import (build_trajectory, iter_sft_records, iter_on_policy_records,
                                      dump_trajectory, adapt_sft_images, main as export_main)
from osworld_agent.rl.rollout import RolloutResult, run_rollout, run_rollouts
from osworld_agent.rl.credit import build_tree, assign_step_level_credit
from osworld_agent.rl.semantic import semantic_action


def image_url():
    handle = BytesIO()
    Image.new('RGB', (3, 3), 'blue').save(handle, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(handle.getvalue()).decode()


def call(role='decision', **kwargs):
    result = dict(actor_role=role, policy_trainable=True,
        messages=[{'role': 'user', 'content': 'Observed public task'}], response='{}',
        valid_action=True, valid_plan=True, policy_revision='weights-1',
        prompt_token_ids=[10, 11], token_ids=[12, 13], token_logprobs=[-0.3, -1.2],
        tokenizer_revision='tok-1', processor_revision='proc-1', chat_template_revision='chat-1',
        policy_revision_source='server', rl_ready=True)
    result.update(kwargs)
    return result


def trajectory():
    return {'task': {'id': 'task'}, 'run_manifest': {},
        'outcome': {'evaluation_available': True, 'reward_valid': True,
                    'score': 1, 'reward': 1, 'reward_source': 'completion_contract'},
        'planning_calls': [call('planner')],
        'steps': [{'step': 1, 'trainable': True,
            'action': {'action': 'click', 'args': {'target': 'Download'}},
            'policy_input': call()['messages'], 'policy_output': '{}',
            'decision_calls': [call()], 'execution_info': {'status': 'executed'},
            'assessment': {'action_status': 'success'}, 'verification': {'status': 'passed'},
            'terminated': True, 'truncated': False, 'reward': 0}]}


def test_events_survive_exception_and_assets_are_deduplicated(tmp_path):
    destination = tmp_path / 'events.jsonl'
    with pytest.raises(RuntimeError):
        with TrainingEventWriter(destination, run_id='episode') as writer:
            writer.emit('observation', observation_id='o1', image=image_url())
            writer.emit('model_call', parent_decision_id='d1', observation_id='o1',
                        actor_role='decision', source='sampled', call={'image': image_url()})
            raise RuntimeError('simulated process-level failure')
    events = read_events(destination)
    assert [item['event_type'] for item in events] == ['episode_start', 'observation', 'model_call']
    assert events[-1]['parent_decision_id'] == 'd1'
    assert len({event['event_id'] for event in events}) == 3
    assert len(list((tmp_path / 'assets').iterdir())) == 1
    assert 'data:image/' not in destination.read_text()
    with pytest.raises(FileExistsError):
        TrainingEventWriter(destination)


def test_events_ignore_only_trailing_torn_write(tmp_path):
    path = tmp_path / 'events.jsonl'
    with TrainingEventWriter(path):
        pass
    with path.open('a') as handle:
        handle.write('{"incomplete":')
    assert len(read_events(path)) == 1
    with path.open('a') as handle:
        handle.write('\n')
    with pytest.raises(json.JSONDecodeError):
        read_events(path)


@pytest.mark.parametrize('status', ['rejected', 'failed', 'failure', 'uncertain', 'error'])
def test_positive_sft_drops_known_bad_execution(status):
    data = trajectory()
    data['steps'][0]['execution_info']['status'] = status
    assert list(iter_sft_records(data)) == []
    assert data['steps'][0]['execution_info']['status'] == status  # raw data preserved


def test_sft_separates_planner_and_decision_and_no_grounder():
    data = trajectory()
    assert [record['actor_role'] for record in iter_sft_records(data)] == ['decision']
    assert [record['actor_role'] for record in iter_sft_records(data, actor_role='planner')] == ['planner']
    data['planning_calls'][0]['valid_plan'] = False
    assert list(iter_sft_records(data, actor_role='planner')) == []
    with pytest.raises(ValueError):
        list(iter_sft_records(data, actor_role='grounder'))


def test_early_finish_is_not_positive_even_when_episode_later_succeeds():
    data = trajectory()
    step = data['steps'][0]
    step['action'] = {'action': 'request_finish', 'args': {'answer': 'done'}}
    step['verification'] = {'status': 'failed'}
    assert list(iter_sft_records(data)) == []
    step['verification']['status'] = 'passed'
    assert len(list(iter_sft_records(data))) == 1


def test_private_evaluator_does_not_enter_exports():
    data = trajectory()
    data['private_evaluation'] = {'secret': 'PRIVATE_GOLD_123'}
    data['run_manifest']['private_spec'] = {'answer': 'PRIVATE_GOLD_123'}
    assert 'PRIVATE_GOLD_123' not in json.dumps(list(iter_sft_records(data)))
    assert 'PRIVATE_GOLD_123' not in json.dumps(list(iter_on_policy_records(data, 'weights-1')))


def test_uncaptured_code_agent_sampling_cannot_enter_on_policy_training():
    data = trajectory()
    data['steps'][0]['action'] = {'action': 'call_code_agent', 'args': {'task': 'edit file'}}
    with pytest.raises(ValueError, match='Code-agent sampling'):
        list(iter_on_policy_records(data, 'weights-1'))


def test_on_policy_includes_trainable_planner_and_explicitly_freezes_teacher():
    data = trajectory()
    records = list(iter_on_policy_records(data, 'weights-1'))
    assert [record['actor_role'] for record in records] == ['planner', 'decision']
    data['planning_calls'][0] = {'policy_trainable': False, 'response': 'teacher plan'}
    assert len(list(iter_on_policy_records(data, 'weights-1'))) == 1


@pytest.mark.parametrize('field', ['prompt_token_ids', 'token_ids', 'token_logprobs',
    'policy_revision', 'tokenizer_revision', 'processor_revision', 'chat_template_revision', 'policy_trainable'])
def test_on_policy_rejects_incomplete_real_metadata(field):
    data = trajectory()
    data['steps'][0]['decision_calls'][0].pop(field)
    with pytest.raises(ValueError):
        list(iter_on_policy_records(data, 'weights-1'))


def test_on_policy_retry_is_not_a_terminal_environment_transition():
    data = trajectory()
    data['steps'][0]['decision_calls'].insert(0, call(valid_action=False))
    records = list(iter_on_policy_records(data, 'weights-1'))
    assert records[1]['terminated'] is False and records[1]['step_reward'] is None
    assert records[2]['terminated'] is True


def test_unknown_rewards_are_null_and_cannot_export():
    result = SimpleNamespace(trajectory=[], score=0, evaluation_available=False, plan=None)
    data = build_trajectory('task', 'browser', 'task', result)
    assert data['outcome']['reward'] is None
    assert data['outcome']['reward_valid'] is False
    assert list(iter_sft_records(data)) == []
    with pytest.raises(ValueError, match='Independent'):
        list(iter_on_policy_records(data, 'weights-1'))


def test_multimodal_sft_adapter_preserves_order_and_validates_assets(tmp_path):
    data = trajectory()
    data['steps'][0]['policy_input'][0]['content'] = [
        {'type': 'text', 'text': 'image follows'},
        {'type': 'image_url', 'image_url': {'url': image_url()}}]
    path = tmp_path / 'trajectory.json'
    dump_trajectory(path, data)
    persisted = json.loads(path.read_text())
    record = adapt_sft_images(next(iter_sft_records(persisted)), path)
    assert record['prompt'][0]['content'][-1] == {'type': 'image'}
    assert Path(record['images'][0]).is_file()
    output = tmp_path / 'sft.jsonl'
    assert export_main([str(path), '--output', str(output), '--image-paths']) == 0
    assert json.loads(output.read_text())['images'] == record['images']


class IsolatedEnv:
    def __init__(self):
        self.episodes = []
        self.closed = False

    def begin_episode(self, task_id, isolated):
        self.episodes.append(task_id)
        return {'isolated': isolated, 'initial_state_id': 'blank-browser-v1', 'episode_id': task_id}

    def close(self):
        self.closed = True


class Agent:
    def __init__(self, valid=True):
        self.env = IsolatedEnv()
        self.valid = valid
        self.received = []

    def run(self, instruction, *, max_steps, completion_spec=None, run_id=None, event_path=None):
        self.received.append(max_steps)
        if event_path:
            with TrainingEventWriter(event_path, run_id=run_id) as writer:
                writer.emit('episode_end')
        return SimpleNamespace(trajectory=[], score=1 if self.valid else 0, success=self.valid,
            steps=1, final_answer='done', evaluation_available=self.valid, termination_reason='verified',
            reward_source='test_evaluator', run_manifest={})


def test_collector_passes_max_steps_and_isolates_every_episode(tmp_path):
    agent = Agent()
    results = run_rollouts(agent, 'task', 2, max_steps=7, output_dir=tmp_path)
    assert agent.received == [7, 7]
    assert len(set(agent.env.episodes)) == 2
    assert results[0].initial_state_id == results[1].initial_state_id
    assert len(list(tmp_path.glob('*/events.jsonl'))) == 2
    assert len(list(tmp_path.glob('*/trajectory.json'))) == 2


def test_collector_masks_unknown_and_restores_factory_environment():
    agent = Agent(valid=False)
    original = agent.env
    fresh = IsolatedEnv()
    result = run_rollout(agent, 'task', env_factory=lambda: fresh)
    assert result.score is None and not result.reward_valid
    assert agent.env is original and fresh.closed


def test_collector_rejects_reset_without_isolation():
    agent = Agent()
    agent.env = SimpleNamespace(reset=lambda: None)
    with pytest.raises(ValueError, match='begin_episode'):
        run_rollout(agent, 'task')


def credit_rollout(score=1, state='blank-browser-v1', revision='weights-1'):
    return RolloutResult(instruction='task', semantic_actions=['click(x)'], score=score,
        evaluation_available=score is not None, reward_valid=score is not None,
        initial_state_id=state, policy_revision=revision, task_spec_hash='spec-1')


def test_unknown_credit_cannot_count_as_failure():
    with pytest.raises(ValueError, match='unknown'):
        build_tree([credit_rollout(None)])


@pytest.mark.parametrize('other', [credit_rollout(state='logged-in'), credit_rollout(revision='weights-2')])
def test_credit_refuses_mixed_initial_states_or_policies(other):
    with pytest.raises(ValueError, match='Split'):
        build_tree([credit_rollout(), other])


def test_credit_uses_explicit_prior_and_stable_semantics():
    first = credit_rollout()
    second = credit_rollout(0)
    second.semantic_actions = ['click(y)']
    assert [row[0] for row in assign_step_level_credit([first, second], normalize=False)] == pytest.approx([1/6, -1/6])
    assert [row[0] for row in assign_step_level_credit([first, second], alpha=2, beta=2, normalize=False)] == pytest.approx([0.1, -0.1])
    a = {'action': 'click', 'args': {'target': 'Download', 'target_ref': 'snapshot-1:123'}}
    b = {'action': 'click', 'args': {'target': 'Download', 'target_ref': 'snapshot-2:999'}}
    assert semantic_action(a) == semantic_action(b)


@pytest.mark.parametrize('patch', [
    {'rl_ready': False}, {'policy_revision_source': 'configured'},
    {'rollout_validation_errors': ['checkpoint mismatch']},
    {'token_logprobs': [float('nan'), -1]}, {'policy_revision': 'weights-new'},
])
def test_rl_rejects_server_validation_failures(patch):
    data = trajectory()
    data['steps'][0]['decision_calls'][0].update(patch)
    with pytest.raises(ValueError):
        list(iter_on_policy_records(data, 'weights-1'))


def test_factory_environment_closed_when_isolation_validation_fails():
    agent = Agent()
    original = agent.env
    bad = IsolatedEnv()
    bad.begin_episode = lambda **kwargs: {'isolated': False}
    with pytest.raises(ValueError, match='isolation'):
        run_rollout(agent, 'task', env_factory=lambda: bad)
    assert bad.closed and agent.env is original


def test_private_contract_is_only_passed_as_evaluator_argument():
    agent = Agent()
    received = []
    def run(instruction, *, max_steps, private_completion_spec=None, **kwargs):
        received.append((instruction, private_completion_spec))
        return SimpleNamespace(trajectory=[], score=0, evaluation_available=False,
                               success=False, steps=0, final_answer='')
    agent.run = run
    result = run_rollout(agent, 'public task', private_completion_spec={'secret': 'gold'})
    assert received == [('public task', {'secret': 'gold'})]
    assert result.instruction == 'public task'


def test_collection_cli_writes_masked_episode_and_durable_events(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from osworld_agent.rl.rollout import main
    factory_module = ModuleType('offline_agent_factory')
    agent = Agent(valid=False)
    factory_module.create = lambda: agent
    monkeypatch.setitem(sys.modules, 'offline_agent_factory', factory_module)
    tasks = tmp_path / 'tasks.jsonl'
    tasks.write_text(json.dumps({'instruction': 'task'}) + '\n')
    output = tmp_path / 'runs'
    assert main(['--factory', 'offline_agent_factory:create', '--tasks', str(tasks),
                 '--output-dir', str(output), '--max-steps', '3']) == 0
    paths = list(output.glob('*/trajectory.json'))
    assert len(paths) == 1
    stored = json.loads(paths[0].read_text())
    assert stored['outcome']['reward'] is None
    assert agent.received == [3] and agent.env.closed
    assert read_events(paths[0].parent / 'events.jsonl')[-1]['event_type'] == 'episode_end'


def replanning_trajectory():
    data = trajectory()
    initial = call('planner', call_id='p1', plan_stage='initial')
    replan = call('planner', call_id='p2', plan_stage='replan')
    first = call(call_id='d1')
    second = call(call_id='d2')
    data['planning_calls'] = [initial, replan]
    step1 = data['steps'][0]
    step1.update(action_id='a_1', observation_id='o_1', decision_calls=[first], terminated=False)
    step2 = copy.deepcopy(step1)
    step2.update(step=2, action_id='a_2', observation_id='o_2', decision_calls=[second], terminated=True)
    data['steps'].append(step2)
    data['events'] = [
        {'event_type': 'model_call', 'event_id': 'e1', 'call': initial,
         'timestamp': 40, 'parent_decision_id': None},
        {'event_type': 'model_call', 'event_id': 'e2', 'call': first,
         'timestamp': 30, 'parent_decision_id': 'a_1', 'observation_id': 'o_1'},
        {'event_type': 'execution', 'event_id': 'e3', 'parent_decision_id': 'a_1'},
        {'event_type': 'model_call', 'event_id': 'e4', 'call': replan,
         'timestamp': 20, 'parent_decision_id': 'a_1'},
        {'event_type': 'model_call', 'event_id': 'e5', 'call': second,
         'timestamp': 10, 'parent_decision_id': 'a_2', 'observation_id': 'o_2'},
    ]
    return data


def test_rl_calls_follow_event_order_across_replanning_not_role_or_timestamp():
    data = replanning_trajectory()
    records = list(iter_on_policy_records(data, 'weights-1'))
    assert [record['call_id'] for record in records] == ['p1', 'd1', 'p2', 'd2']
    assert [record['actor_role'] for record in records] == ['planner', 'decision', 'planner', 'decision']
    assert [record['event_id'] for record in records] == ['e1', 'e2', 'e4', 'e5']
    assert [record['step'] for record in records] == [None, 1, 1, 2]
    assert records[-1]['terminated'] and not any(record['terminated'] for record in records[:-1])
    assert all(record['call_order_source'] == 'events' for record in records)


def test_sft_roles_keep_call_and_parent_transition_associations():
    data = replanning_trajectory()
    planners = list(iter_sft_records(data, actor_role='planner'))
    decisions = list(iter_sft_records(data))
    assert [(item['call_id'], item['step'], item['plan_stage']) for item in planners] == [
        ('p1', None, 'initial'), ('p2', 1, 'replan')]
    assert planners[1]['parent_decision_id'] == 'a_1'
    # Replanning follows an action, so its actual observation cannot be inferred
    # from that action's pre-execution observation ID.
    assert planners[1]['event_id'] == 'e4' and planners[1]['observation_id'] is None
    assert [(item['call_id'], item['step'], item['observation_id']) for item in decisions] == [
        ('d1', 1, 'o_1'), ('d2', 2, 'o_2')]


def test_rl_legacy_without_events_remains_explicitly_compatible():
    data = trajectory()
    records = list(iter_on_policy_records(data, 'weights-1'))
    assert [record['actor_role'] for record in records] == ['planner', 'decision']
    assert all(record['call_order_source'] == 'legacy_role_order' for record in records)
    assert all(record['event_id'] is None for record in records)
    assert len(list(iter_sft_records(data, actor_role='planner'))) == 1


def test_rl_partial_event_index_does_not_silently_reorder_missing_calls():
    data = replanning_trajectory()
    data['events'] = [event for event in data['events'] if event['event_id'] != 'e4']
    with pytest.raises(ValueError, match='event stream is incomplete'):
        list(iter_on_policy_records(data, 'weights-1'))


def test_duplicate_call_ids_are_rejected_as_ambiguous():
    data = replanning_trajectory()
    data['events'].append(copy.deepcopy(data['events'][0]))
    with pytest.raises(ValueError, match='Duplicate model call_id'):
        list(iter_on_policy_records(data, 'weights-1'))
