"""Intent survives unsupported wording without upgrading guesses to evidence."""
import pytest

from osworld_agent.task_specs import infer_paper_spec, task_profile, task_analysis, audit_requirement_proposals
from osworld_agent.verifiers import CompletionVerifier
from osworld_agent.pipeline import Pipeline
from test_pipeline_completion import Env, Model
from test_pipeline_training import paper_plan
from test_paper_goal_binding import official_event
from test_paper_completion import pdf_file
from test_paper_resolver import TITLE, resolve


@pytest.mark.parametrize('task', [
    '搜索论文roboclaw并下载，然后改名为data',
    '请帮我搜索论文 roboclaw 并下载，然后改名为 data',
    '检索论文《roboclaw》并下载，保存为data',
    '查找roboclaw的论文，下载后改名为data',
    '搜索并下载论文roboclaw，然后重命名为data',
    '先搜索roboclaw论文，然后下载，再改名为data',
    '下载论文roboclaw，然后改名为data',
])
def test_word_orders_preserve_query_and_filename(task):
    checks = infer_paper_spec(task)['checks']
    assert checks == [{'kind': 'paper_download', 'query': 'roboclaw', 'count': 1,
                       'format': 'pdf', 'filename': 'data.pdf'}]


def test_plain_paper_request_infers_query_without_paper_id_or_known_url():
    task = '搜索论文Attention is all you need并下载'
    checks = infer_paper_spec(task)['checks']
    assert checks == [{'kind': 'paper_download', 'query': 'Attention is all you need',
                       'count': 1, 'format': 'pdf'}]
    profile = task_profile(task)
    assert profile['goals'][0]['query'] == 'Attention is all you need'
    assert 'arxiv_id' not in profile['goals'][0] and 'url' not in profile['goals'][0]


def test_default_project_folder_is_an_artifact_path_check(tmp_path):
    task = '搜索论文 Attention is all you need 并下载 PDF 到默认工程文件夹'
    checks = infer_paper_spec(task)['checks']
    assert checks == [{'kind': 'paper_download', 'query': 'Attention is all you need',
                       'count': 1, 'format': 'pdf', 'workspace': 'default'}]
    test_task = '搜索论文roboclaw并下载 PDF 到默认工程文件夹'
    verifier = CompletionVerifier(test_task)
    verifier.bind_paper('check_1', resolve().binding)
    selected = tmp_path / 'selected'
    selected.mkdir()
    good = pdf_file(selected / 'paper.pdf', TITLE, 'Example Author', 'Abstract')
    from osworld_agent.env import Observation
    verifier.observe(Observation(info={'initial_state': {'workspace_dir': str(selected)},
        'artifacts': [official_event(good)]}))
    assert verifier.evaluate()['status'] == 'passed'

    wrong = pdf_file(tmp_path / 'outside.pdf', TITLE, 'Example Author', 'Abstract')
    outside = CompletionVerifier(test_task)
    outside.bind_paper('check_1', resolve().binding)
    outside.observe(Observation(info={'initial_state': {'workspace_dir': str(selected)},
        'artifacts': [official_event(wrong)]}))
    report = outside.evaluate()
    assert report['status'] == 'failed'
    assert 'outside the default project folder' in report['checks'][0]['reason']


def test_planner_sees_preloaded_search_page_separately_from_plain_task(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    task = '搜索论文 Attention Is All You Need 并下载'
    env = Env()
    env.browser_start_url = 'http://127.0.0.1:8090/demo/paper-search'
    env.obs = __import__('osworld_agent.env', fromlist=['Observation']).Observation(context=[{
        'kind': 'browser_ax', 'url': env.browser_start_url,
        'title': 'Title search', 'nodes': [{'ref': '1', 'role': 'searchbox', 'name': 'Search paper title'},
                                          {'ref': '2', 'role': 'button', 'name': 'Search'}]}])
    model = Model([{'type': 'fail', 'reason': 'test ends'}], paper_plan())
    Pipeline(model=model, env=env, grounding=None, system_prompt='test', max_steps=1).run(task)
    planner_input = model.calls[0][0]['content']
    assert env.reset_count == 1
    assert task in planner_input and env.browser_start_url in planner_input
    assert 'Search paper title' in planner_input
    assert '1706.03762' not in planner_input


def test_paper_plan_cannot_promote_recalled_author_or_id(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    task = '搜索一篇叫Attention is all you need的论文并下载'
    plan = paper_plan()
    for step in plan['steps']:
        step['goal'] = '确认 Vaswani 和 arXiv ID 1706.03762'
        step['expected_evidence'] = '看到 1706.03762'
        step['verification'] = [{'kind': 'text_present', 'text': 'Vaswani'}]
    plan['keywords'] = ['Attention is all you need Vaswani']
    plan['final_check'] = '下载 arXiv 1706.03762'
    model = Model([{'type': 'fail', 'reason': 'test ends'}], plan)
    result = Pipeline(model=model, env=Env(), grounding=None, system_prompt='test', max_steps=1).run(task)
    assert result.termination_reason != 'planning_failed'
    assert 'Vaswani' in result.planning_calls[0]['response']  # raw proposal retained for diagnosis
    applied = __import__('json').dumps(result.plan, ensure_ascii=False)
    assert 'Vaswani' not in applied and '1706.03762' not in applied
    assert result.plan['keywords'] == ['Attention is all you need']
    assert 'Vaswani' not in model.calls[1][-1]['content']


def test_paper_plan_refines_disposable_invalid_predicates_before_validation(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    plan = paper_plan()
    for step in plan['steps']:
        step['verification'] = [{'kind': 'element_state', 'name': 'PDF link',
                                 'role': 'link', 'states': {}}]
    plan['steps'][1]['goal'] = '确认 Vaswani 和 1706.03762'
    model = Model([{'type': 'fail', 'reason': 'test ends'}], plan)
    result = Pipeline(model=model, env=Env(), grounding=None, system_prompt='test', max_steps=1).run(
        '搜索论文 Attention is all you need 并下载 PDF 到默认工程文件夹')
    assert result.termination_reason != 'planning_failed'
    assert len(result.planning_calls) == 1
    assert result.planning_calls[0]['valid_plan']
    applied = __import__('json').dumps(result.plan, ensure_ascii=False)
    assert 'Vaswani' not in applied and '1706.03762' not in applied
    assert result.plan['steps'][0]['verification'][0]['kind'] == 'search_results'
    assert next(s for s in result.plan['steps'] if s['phase'] == 'download')['verification'] == [
        {'kind': 'download_complete', 'check_id': 'check_1'}]


def test_matching_task_needs_same_identity_and_actual_final_filename(tmp_path):
    verifier = CompletionVerifier('搜索论文roboclaw并下载，然后改名为data')
    verifier.bind_paper('check_1', resolve().binding)
    wrong_name = pdf_file(tmp_path / 'wrong.pdf', TITLE, 'Example Author', 'Abstract')
    verifier.observe(None, {'artifacts': [official_event(wrong_name)]})
    assert verifier.evaluate()['status'] == 'failed'
    correct = pdf_file(tmp_path / 'data.pdf', TITLE, 'Example Author', 'Abstract')
    verifier.observe(None, {'artifacts': [official_event(correct, ident='another-download')]})
    assert verifier.evaluate()['status'] == 'passed'


@pytest.mark.parametrize('tail', ['然后发邮件给导师', '并保存到桌面', '并确认作者是Alice'])
def test_extra_requirement_is_never_dropped(tail):
    checks = infer_paper_spec('搜索论文roboclaw并下载，然后改名为data，' + tail)['checks']
    assert checks[0]['filename'] == 'data.pdf'
    assert checks[1]['kind'] == 'unresolved_requirement'
    assert tail in checks[1]['text']


@pytest.mark.parametrize('task', [
    '搜索论文roboclaw，不要下载', '如果找到论文roboclaw就下载',
    '下载作者是Alice的roboclaw论文', '搜索论文A和B并下载',
])
def test_ambiguous_or_conditional_requests_do_not_create_a_guessed_paper(task):
    spec = infer_paper_spec(task)
    assert not any(c['kind'] == 'paper_download' for c in (spec or {}).get('checks', []))


def test_workflow_is_available_even_with_empty_or_unresolved_contract():
    task = '把找到的论文roboclaw保存一份，文件名用data'
    for spec in ({'checks': []}, infer_paper_spec(task)):
        profile = task_profile(task, spec)
        assert profile['kind'] == 'paper_download'
        assert profile['intent_source'] == 'task_hint' and not profile['contract_available']
        assert {s['phase'] for s in profile['workflow']} == {'search', 'identify', 'locate_fulltext', 'download', 'verify'}
        assert task in task_analysis(task, spec)['unresolved_requirements']
    assert task_profile('搜索论文roboclaw，不要下载', {'checks': []})['kind'] == 'generic'


def test_proposals_cannot_invent_fields_or_become_completion_contracts(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    task = '把找到的论文roboclaw保存一份，文件名用data'
    plan = paper_plan()
    plan['requirements'] = [
        {'kind': 'paper_download', 'source_text': task, 'value': 'roboclaw', 'arxiv_id': 'invented'},
        {'kind': 'rename', 'source_text': '文件名用data', 'value': 'data.pdf'},
        {'kind': 'other', 'source_text': '不存在的要求'},
    ]
    model = Model([{'type': 'fail', 'reason': 'fixture'}], plan)
    result = Pipeline(model=model, env=Env(), grounding=None, system_prompt='test', max_steps=1).run(task)
    audit = result.plan['requirements']
    assert len(audit['accepted']) == 1 and len(audit['rejected']) == 2
    assert audit['accepted'][0]['trusted'] is False and 'arxiv_id' not in audit['accepted'][0]
    assert not audit['coverage_verified'] and audit['original_task'] == task
    assert result.run_manifest['public_completion']['checks'][0]['kind'] == 'unresolved_requirement'
    assert not result.success and not result.evaluation_available
    assert any(e['event_type'] == 'task_analysis' for e in result.events)


def test_action_aliases_are_bounded_and_cannot_restore_nonbasic_actions():
    assert Pipeline._plan_actions(['open_url', 'key_press', 'type_text', 'goto']) == ['goto', 'press', 'type']
    for names in (['download'], ['rename_artifact'], ['run_code'], [{}]):
        with pytest.raises(ValueError):
            Pipeline._plan_actions(names)


def test_raw_source_quote_alone_does_not_prove_semantic_correctness():
    task = '不要下载论文roboclaw'
    audit = audit_requirement_proposals(task, [{'kind': 'paper_download', 'source_text': task, 'value': 'roboclaw'}])
    assert audit['accepted'][0]['trusted'] is False
    assert not audit['coverage_verified']
