"""One source preference reaches planning and execution without changing goals."""
import json

import pytest

from osworld_agent.decision_context import action_scope, system_sections
from osworld_agent.decision_modes import decision_mode
from osworld_agent.env import Observation
from osworld_agent.paper_sources import paper_search_instruction
from osworld_agent.prompts import build_planner_prompt
from osworld_agent.task_specs import infer_paper_spec, task_profile
from test_mode_prompts import browser_observation
from test_planning_service import Model, accepting, service, state
from test_pipeline_training import paper_plan


def prompt_parts(task):
    profile = task_profile(task, {'checks': []})
    obs = browser_observation()
    mode = decision_mode(obs)
    sections = dict(system_sections(action_scope(None, obs, profile=mode), obs,
        profile=mode, task=task, step={'phase': 'search', 'goal': '检索目标论文'}, task_profile=profile))
    return (build_planner_prompt(task, {'checks': []}, profile), sections['paper_search'],
            next(step['instruction'] for step in profile['workflow'] if step['phase'] == 'search'))


def test_default_paper_source_is_arxiv_in_planner_current_step_and_workflow():
    task = '搜索论文WorldGUI并下载'
    original = infer_paper_spec(task)
    for prompt in prompt_parts(task):
        assert paper_search_instruction(task) in prompt
        assert '默认优先使用 arXiv 搜索入口 https://arxiv.org/search/' in prompt
        assert '不要先去 Google Scholar' in prompt
        assert '2502.08047' not in prompt and '1706.03762' not in prompt
        assert 'Vaswani' not in prompt
    assert infer_paper_spec(task) == original


@pytest.mark.parametrize('task,source', [
    ('请在Google Scholar搜索论文WorldGUI并下载', 'https://scholar.google.com/scholar?hl=en'),
    ('请用谷歌搜索论文WorldGUI并下载', 'https://www.google.com/'),
    ('请从Nature官网搜索论文WorldGUI并下载', 'https://www.nature.com/'),
    ('请从https://publisher.example/search 搜索论文WorldGUI并下载', 'https://publisher.example/search'),
])
def test_user_source_overrides_default_in_every_paper_prompt(task, source):
    for prompt in prompt_parts(task):
        assert paper_search_instruction(task) in prompt
        assert source in prompt
        assert '不得把用户指定的其他来源擅自换成 arXiv' in prompt
        assert '默认优先使用 arXiv 搜索入口' not in prompt


def test_only_source_constraint_does_not_receive_unconditional_cross_site_fallback():
    task = '仅在Google Scholar搜索论文WorldGUI并下载'
    for prompt in prompt_parts(task):
        assert '即使检索受阻也不得自行换站' in prompt
        assert 'https://arxiv.org/search/' not in prompt
        assert '来源受限' in prompt or '仅在来源策略允许' in prompt


@pytest.mark.parametrize('phase', ['search', 'identify', 'locate_fulltext', 'download', 'verify'])
def test_user_source_instruction_remains_in_later_paper_steps_without_duplicate_text(phase):
    task = '仅在Google Scholar搜索论文WorldGUI并下载'
    obs = browser_observation()
    mode = decision_mode(obs)
    sections = system_sections(action_scope(None, obs, profile=mode), obs,
        profile=mode, task=task, step={'phase': phase}, task_profile=task_profile(task, {'checks': []}))
    system = '\n'.join(text for _, text in sections)
    assert system.count(paper_search_instruction(task)) == 1
    assert '即使检索受阻也不得自行换站' in system
    assert 'https://arxiv.org/search/' not in system


@pytest.mark.parametrize('task,preloaded', [
    ('请在Google Scholar搜索论文WorldGUI并下载', 'https://arxiv.org/search/'),
    ('请在arXiv搜索论文WorldGUI并下载', 'https://scholar.google.com/scholar?hl=en'),
])
def test_preloaded_search_box_does_not_override_users_explicit_source(task, preloaded):
    model, events, calls, accepted = Model([paper_plan()]), [], [], []
    planner, record = service(model, events)
    current = state()
    current.task_profile = task_profile(task, {'checks': []})
    current.obs = Observation(context=[{'kind': 'browser_ax', 'url': preloaded, 'title': 'Search',
        'nodes': [{'role': 'textbox', 'name': 'Search query'}, {'role': 'button', 'name': 'Search'}]}])
    assert planner.plan(task, current, {'checks': []}, [], calls, {'start_url': preloaded},
                        accepting(events, record, accepted))
    prompt = model.calls[0][0][0]['content']
    assert json.dumps(preloaded) in prompt
    assert paper_search_instruction(task) in prompt
    assert '先遵从用户明确指定的来源' in prompt
    assert '不得因预载页面有搜索框就改走其他来源' in prompt
    assert '若此页已有可用搜索框，应优先在这里按标题搜索' not in prompt


def test_explicit_fixture_environment_can_be_reused_without_inventing_paper_identity():
    task = '搜索论文WorldGUI并下载'
    planner_text, live_text, workflow = prompt_parts(task)
    for prompt in (planner_text, live_text, workflow):
        assert '显式预载' in prompt or '明确预载' in prompt
        assert '不与用户指定来源冲突' in prompt or '不能覆盖用户明确指定的来源' in prompt or '不与用户来源冲突' in prompt
        assert '2502.08047' not in prompt
