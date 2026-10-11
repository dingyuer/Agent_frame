"""Offline human demonstration logic; no browser, server, or model requests."""
import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from osworld_agent.actions import Action, action_from_decision, parse_decision, validate_action
from osworld_agent.config import ContextConfig
from osworld_agent.env import Observation
from osworld_agent.human_demonstration import HumanDemonstrationSession
from osworld_agent.trajectory import iter_on_policy_records, iter_sft_records
from osworld_agent.training_events import read_events


def observation(text='PRE_ONLY', color='blue', url='https://example.test/start', node_name='Download'):
    return Observation(screenshot=Image.new('RGB', (12, 8), color), text=text,
        info={'observation_id': text, 'screenshot_space': 'css_viewport', 'page_id': 'page'},
        context=[{'kind': 'browser_ax', 'snapshot_id': text, 'url': url,
            'nodes': [{'ref': 'ref1', 'id': 'button1', 'role': 'button', 'name': node_name,
                       'in_viewport': True}]}])


def session(tmp_path, **kwargs):
    return HumanDemonstrationSession('在页面上做一次人工示范', tmp_path, **kwargs)


def record(recorder, action=None, before=None, after=None, metadata=None, status='executed'):
    before = before or observation()
    after = after or observation('POST_ONLY', 'red')
    prepared = recorder.prepare_observation(before)
    metadata = {**(metadata or {}), 'preparation_id': prepared['preparation_id'],
                'frozen_policy_input': prepared['policy_input'],
                'frozen_prompt_metadata': prepared['prompt_metadata']}
    result = recorder.record_transition(action or Action('click', {'target_ref': 'ref1', 'x': 6, 'y': 4}),
                                       before, after, {'status': status}, metadata)
    return result, prepared


def exported(recorder, **kwargs):
    result = recorder.export_sft(**kwargs)
    lines = Path(result['path']).read_text().splitlines()
    return result, [json.loads(line) for line in lines]


def test_human_unknown_task_can_export_reviewed_but_not_verified(tmp_path):
    recorder = session(tmp_path)
    step, _ = record(recorder)
    recorder.review_step(step['step_id'], keep=True)
    outcome = recorder.stop(claimed_complete=True)
    assert outcome['human_claimed_complete'] and not outcome['success']
    assert outcome['reward'] is None and outcome['reward_valid'] is False
    result, samples = exported(recorder)
    assert result['count'] == 1
    sample = samples[0]
    assert sample['human_reviewed'] and not sample['independent_verified']
    assert sample['source'] == 'human' and sample['actor'] == 'human'
    assert sample['provenance']['on_policy'] is False
    assert sample['reward'] is None and not sample['reward_valid']
    recorder.close()


def test_exact_pre_image_ax_prompt_and_schema_without_synthetic_reasoning(tmp_path):
    recorder = session(tmp_path)
    step, prepared = record(recorder)
    recorder.review_step(step['step_id'], keep=True, intent='HUMAN_INTENT', expected_effect='HUMAN_EXPECTATION')
    recorder.stop()
    result, samples = exported(recorder)
    sample = samples[0]
    text = json.dumps(sample['prompt'], ensure_ascii=False)
    assert 'PRE_ONLY' in text and 'POST_ONLY' not in text
    assert 'HUMAN_INTENT' not in text and 'HUMAN_EXPECTATION' not in text
    assert len(sample['images']) == 1
    assert not Path(sample['images'][0]).is_absolute()
    with Image.open(Path(result['path']).parent / sample['images'][0]) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
    completion = parse_decision(sample['completion'][0]['content'])
    assert set(completion) == {'next_action', 'expected_effect'}
    assert completion['expected_effect'] == 'HUMAN_EXPECTATION'
    action = action_from_decision(completion)
    assert validate_action(action) is None
    assert action.args['target'] == 'Download'
    assert not {'x', 'y', 'coordinate_space'} & action.args.keys()
    assert 'screen_analysis' not in completion and 'state_update' not in completion
    recorder.close()


def test_coordinate_only_action_requires_human_semantic_target(tmp_path):
    recorder = session(tmp_path)
    step, _ = record(recorder, Action('click', {'x': 5, 'y': 4}))
    assert step['semantic_error'] == 'missing_semantic_target'
    recorder.review_step(step['step_id'], keep=True)
    recorder.stop()
    result, samples = exported(recorder)
    assert not samples and result['excluded'][0]['reason'] == 'missing_semantic_target'
    recorder.review_step(step['step_id'], keep=True, target='the Download button')
    _, samples = exported(recorder)
    assert parse_decision(samples[0]['completion'][0]['content'])['next_action']['target'] == 'the Download button'
    recorder.close()


def test_general_intent_does_not_make_coordinate_only_action_semantic(tmp_path):
    recorder = session(tmp_path)
    step, _ = record(recorder, Action('click', {'x': 5, 'y': 4}))
    recorder.review_step(step['step_id'], keep=True, intent='Find the paper')
    recorder.stop()
    assert exported(recorder)[0]['count'] == 0
    recorder.close()


@pytest.mark.parametrize('reason', ['nested_scroll_target_not_exposed', 'position_sensitive_target'])
def test_controller_semantic_exclusion_cannot_be_overridden_by_review(tmp_path, reason):
    recorder = session(tmp_path)
    step, _ = record(recorder, metadata={'sft_exclude_reason': reason})
    recorder.review_step(step['step_id'], keep=True, target='Clearly labelled slider',
                         intent='Adjust the slider', safe_for_training=True)
    recorder.stop()
    result, samples = exported(recorder)
    assert samples == []
    assert result['excluded'] == [{'step_id': step['step_id'], 'reason': reason}]
    archive = json.loads((tmp_path / 'trajectory.json').read_text())
    assert archive['steps'][0]['metadata']['sft_exclude_reason'] == reason
    assert archive['steps'][0]['raw_action']['action'] == 'click'
    assert archive['steps'][0]['review']['keep'] is True
    recorder.close()


def test_missing_dialog_screenshot_can_be_recorded_but_never_training_input(tmp_path):
    recorder = session(tmp_path)
    before = observation()
    before.screenshot = None
    before.info['dialog'] = {'type': 'alert', 'message': 'Accept to continue'}
    step, prepared = record(recorder, Action('handle_dialog', {'accept': True}), before=before)
    assert prepared['policy_input'] is None
    assert not prepared['prompt_metadata']['usable_for_training']
    recorder.review_step(step['step_id'], keep=True)
    recorder.stop()
    result, samples = exported(recorder)
    assert samples == [] and result['excluded'][0]['reason'] == 'missing_pre_action_input'
    recorder.close()


def test_summary_keeps_portable_review_images_and_omits_full_observations(tmp_path):
    recorder = session(tmp_path)
    record(recorder)
    item = recorder.summary()['steps'][0]
    assert item['observation_id'] == 'PRE_ONLY'
    assert item['next_observation_id'] == 'POST_ONLY'
    assert 'observation' not in item and 'next_observation' not in item
    assert 'context' not in item and 'policy_input' not in item
    assert item['screenshot'].startswith('assets/')
    assert item['next_screenshot'].startswith('assets/')
    assert item['screenshot'] != item['next_screenshot']
    assert (tmp_path / item['screenshot']).is_file()
    assert recorder.list_steps()[0]['observation']['context']
    recorder.close()


def test_hidden_dom_refs_are_not_policy_labels(tmp_path):
    recorder = session(tmp_path, context_config=ContextConfig(structure_tokens=0))
    step, prepared = record(recorder)
    assert prepared['prompt_metadata']['exposed_refs'] == []
    assert step['action'] == {'action': 'click', 'args': {'target': 'Download'}}
    recorder.review_step(step['step_id'], keep=True)
    recorder.stop()
    _, samples = exported(recorder)
    action = parse_decision(samples[0]['completion'][0]['content'])['next_action']
    assert 'target_ref' not in action and 'element_id' not in action
    recorder.close()


@pytest.mark.parametrize('kind,args,field,identifier', [
    ('switch_tab', {'tab_id': 'opaque-tab-target'}, 'exposed_tab_ids', 'opaque-tab-target'),
    ('close_tab', {'tab_id': 'opaque-tab-target'}, 'exposed_tab_ids', 'opaque-tab-target'),
    ('handle_dialog', {'accept': True, 'dialog_id': 'opaque-dialog-target'},
     'exposed_dialog_ids', 'opaque-dialog-target'),
])
def test_browser_action_ids_exist_in_frozen_policy_input(tmp_path, kind, args, field, identifier):
    recorder = session(tmp_path)
    before = observation()
    before.info.update(tabs=[{'tab_id': 'opaque-tab-target', 'title': 'Paper', 'url': 'https://example.test/'}],
                       dialog={'id': 'opaque-dialog-target', 'type': 'confirm', 'message': 'Continue?'})
    step, prepared = record(recorder, Action(kind, args), before=before)
    assert identifier in json.dumps(prepared['policy_input'])
    assert identifier in prepared['prompt_metadata'][field]
    recorder.review_step(step['step_id'], keep=True)
    recorder.stop()
    _, samples = exported(recorder)
    assert len(samples) == 1
    assert identifier in json.dumps(samples[0]['prompt'])
    recorder.close()


@pytest.mark.parametrize('kind,args,reason', [
    ('switch_tab', {'tab_id': 'opaque-tab-target'}, 'unexposed_tab_id'),
    ('handle_dialog', {'accept': True, 'dialog_id': 'opaque-dialog-target'}, 'unexposed_dialog_id'),
])
def test_budget_hidden_browser_ids_cannot_become_positive_labels(tmp_path, kind, args, reason):
    recorder = session(tmp_path, context_config=ContextConfig(structure_tokens=0))
    before = observation()
    before.info.update(tabs=[{'tab_id': 'opaque-tab-target', 'title': 'Paper'}],
                       dialog={'id': 'opaque-dialog-target', 'message': 'Continue?'})
    step, prepared = record(recorder, Action(kind, args), before=before)
    assert 'opaque-tab-target' not in json.dumps(prepared['policy_input'])
    assert 'opaque-dialog-target' not in json.dumps(prepared['policy_input'])
    recorder.review_step(step['step_id'], keep=True)
    recorder.stop()
    result, samples = exported(recorder)
    assert samples == [] and result['excluded'][0]['reason'] == reason
    recorder.close()


def test_prepare_required_and_frozen_inputs_cannot_be_forged(tmp_path):
    recorder = session(tmp_path)
    before, after = observation(), observation('POST')
    with pytest.raises(ValueError, match='pre-action preparation'):
        recorder.record_transition(Action('goto', {'url': 'https://example.test/'}), before, after)
    prepared = recorder.prepare_observation(before)
    corrupted = copy.deepcopy(prepared['policy_input'])
    corrupted[-1]['content'][0]['text'] += 'TARGET_FROM_FUTURE'
    with pytest.raises(ValueError, match='altered'):
        recorder.record_transition(Action('goto', {'url': 'https://example.test/'}), before, after,
            metadata={'preparation_id': prepared['preparation_id'], 'frozen_policy_input': corrupted})
    recorder.close()


def test_compiler_exposed_refs_can_be_added_without_changing_captured_state(tmp_path):
    recorder = session(tmp_path)
    before, after = observation(), observation('POST')
    prepared = recorder.prepare_observation(before)
    before.info['exposed_refs'] = prepared['prompt_metadata']['exposed_refs']
    result = recorder.record_transition(Action('press', {'key': 'enter'}), before, after,
        {'status': 'executed'}, {'preparation_id': prepared['preparation_id']})
    assert result['observation']['id'] == 'PRE_ONLY'
    with pytest.raises(ValueError, match='reused'):
        recorder.record_transition(Action('press', {'key': 'enter'}), before, after,
            {'status': 'executed'}, {'preparation_id': prepared['preparation_id']})
    recorder.close()


def test_before_snapshot_mutation_is_rejected(tmp_path):
    recorder = session(tmp_path)
    before, after = observation(), observation('POST')
    prepared = recorder.prepare_observation(before)
    before.context[0]['nodes'][0]['name'] = 'DIFFERENT_CURRENT_PAGE'
    with pytest.raises(ValueError, match='altered'):
        recorder.record_transition(Action('press', {'key': 'enter'}), before, after,
            metadata={'preparation_id': prepared['preparation_id']})
    recorder.close()


@pytest.mark.parametrize('status', ['rejected', 'failed', 'uncertain', 'unknown'])
def test_failed_or_uncertain_execution_not_positive_even_after_human_keep(tmp_path, status):
    recorder = session(tmp_path)
    step, _ = record(recorder, status=status)
    recorder.review_step(step['step_id'], keep=True)
    recorder.stop(claimed_complete=True)
    result, samples = exported(recorder)
    assert not samples and result['excluded'][0]['reason'] == 'execution_not_confirmed'
    recorder.close()


def test_keep_is_required_and_discard_can_never_leak_into_positive_samples(tmp_path):
    recorder = session(tmp_path)
    step, _ = record(recorder)
    recorder.stop()
    assert exported(recorder)[0]['count'] == 0
    recorder.review_step(step['step_id'], keep=False)
    assert exported(recorder)[0]['count'] == 0
    recorder.close()


@pytest.mark.parametrize('flag', ['sensitive', 'contains_secrets', 'is_password'])
def test_sensitive_input_is_redacted_and_cannot_be_overridden(tmp_path, flag):
    recorder = session(tmp_path)
    step, _ = record(recorder, Action('type', {'target': 'Password', 'text': 'NEVER_STORE_PASSWORD_123'}),
                     after=observation('NEVER_STORE_PASSWORD_123'), metadata={flag: True})
    assert step['sensitive'] and step['observation']['screenshot'] is None
    with pytest.raises(ValueError, match='Sensitive'):
        recorder.review_step(step['step_id'], keep=True)
    recorder.stop()
    assert 'NEVER_STORE_PASSWORD_123' not in (tmp_path / 'events.jsonl').read_text()
    assert 'NEVER_STORE_PASSWORD_123' not in (tmp_path / 'trajectory.json').read_text()
    assert exported(recorder)[0]['count'] == 0
    recorder.close()


def test_password_node_is_sensitive_without_metadata_hint(tmp_path):
    recorder = session(tmp_path)
    before = observation()
    before.context[0]['nodes'][0].update(role='textbox', name='Password', input_type='password')
    step, _ = record(recorder, Action('type', {'target_ref': 'ref1', 'text': 'VERY_SECRET'}), before=before)
    assert step['sensitive']
    assert 'VERY_SECRET' not in (tmp_path / 'events.jsonl').read_text()
    recorder.close()


def test_sensitive_url_cannot_escape_through_final_verification_evidence(tmp_path):
    recorder = session(tmp_path, completion_spec={'checks': [
        {'kind': 'url_equals', 'url': 'https://example.test/done'}]})
    after = observation('SECRET_FRAME_ID', url='https://example.test/SECRET_TOKEN_VALUE')
    record(recorder, Action('type', {'target': 'Password', 'text': 'SECRET_TOKEN_VALUE'}),
           after=after, metadata={'sensitive': True})
    assert recorder.stop(final_observation=after)['verification']['status'] == 'unknown'
    for path in ('events.jsonl', 'trajectory.json'):
        contents = (tmp_path / path).read_text()
        assert 'SECRET_TOKEN_VALUE' not in contents and 'SECRET_FRAME_ID' not in contents
    recorder.close()


def test_earlier_actions_in_next_input_but_review_never_rewrites_input(tmp_path):
    recorder = session(tmp_path)
    first, _ = record(recorder, Action('press', {'key': 'enter'}))
    second, prepared = record(recorder, Action('press', {'key': 'tab'}), before=observation('STEP2_BEFORE'))
    assert 'enter' in json.dumps(prepared['policy_input'])
    recorder.review_step(first['step_id'], keep=True, intent='FUTURE_REVIEW_ANNOTATION')
    recorder.review_step(second['step_id'], keep=True)
    recorder.stop()
    _, samples = exported(recorder)
    assert 'FUTURE_REVIEW_ANNOTATION' not in json.dumps(samples[-1]['prompt'])
    recorder.close()


def test_only_framework_verifier_can_mark_success_and_rl_rejects_human(tmp_path):
    spec = {'checks': [{'kind': 'url_equals', 'url': 'https://example.test/done'}]}
    recorder = session(tmp_path, completion_spec=spec)
    step, _ = record(recorder, Action('goto', {'url': 'https://example.test/done'}),
        after=observation('DONE', url='https://example.test/done'))
    recorder.review_step(step['step_id'], keep=True)
    assert recorder.stop(claimed_complete=False)['success'] is True
    result, samples = exported(recorder, require_verified=True)
    assert result['count'] == 1 and samples[0]['independent_verified']
    archive = json.loads((tmp_path / 'trajectory.json').read_text())
    with pytest.raises(ValueError, match='External demonstrations'):
        list(iter_on_policy_records(archive, 'weights-1'))
    assert list(iter_sft_records(archive)) == []  # Only the explicit reviewed-human exporter emits positives.
    recorder.close()


def test_claimed_completion_cannot_override_known_contract_failure(tmp_path):
    recorder = session(tmp_path, completion_spec={'checks': [
        {'kind': 'url_equals', 'url': 'https://example.test/done'}]})
    step, _ = record(recorder)
    recorder.review_step(step['step_id'], keep=True)
    outcome = recorder.stop(claimed_complete=True)
    assert not outcome['success'] and outcome['verification']['status'] == 'failed'
    assert exported(recorder)[0]['count'] == 0
    recorder.close()


def test_private_targets_do_not_enter_policy_input_or_ui_summary(tmp_path):
    recorder = session(tmp_path, private_completion_spec={'checks': [
        {'kind': 'text_present', 'text': 'PRIVATE_GOLD_ANSWER'}]})
    step, prepared = record(recorder)
    assert 'PRIVATE_GOLD_ANSWER' not in json.dumps(prepared['policy_input'])
    assert 'PRIVATE_GOLD_ANSWER' not in json.dumps(recorder.summary())
    assert recorder.summary()['steps'][0]['step'] == step['step_id']
    assert recorder.summary()['steps'][0]['keep'] is None
    recorder.review_step(step['step_id'], keep=True)
    assert recorder.summary()['steps'][0]['keep'] is True
    recorder.stop()
    archive = json.loads((tmp_path / 'trajectory.json').read_text())
    assert archive['evaluation_context']['completion_spec']['checks'][0]['text'] == 'PRIVATE_GOLD_ANSWER'
    recorder.close()


def test_multiple_preparations_are_supported_and_event_assets_survive_close(tmp_path):
    recorder = session(tmp_path)
    before, after = observation(), observation('POST')
    prepared = recorder.prepare_observation(before)
    recorder.prepare_observation(after)
    recorder.record_transition(Action('press', {'key': 'enter'}), before, after,
        {'status': 'executed'}, {'preparation_id': prepared['preparation_id']})
    recorder.close()
    events = read_events(tmp_path / 'events.jsonl')
    transition = events[-1]['transition']
    assert (tmp_path / transition['observation']['screenshot']).is_file()
    assert len(list((tmp_path / 'assets').iterdir())) >= 1
    assert not any(event['event_type'] == 'model_call' for event in events)
