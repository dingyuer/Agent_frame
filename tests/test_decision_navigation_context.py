"""Navigation operands must reach the operator, not only the planner."""
import copy

import pytest

from osworld_agent.decision_context import navigation_brief
from osworld_agent.execution_queue import ExecutionQueue
from osworld_agent.pipeline import StepState
from test_execution_queue_review import pipeline, step


PORTAL = dict(id='w3-portal', title='公司 W3 门户', aliases=['w3网站'],
              keywords=['3MS'], url='https://w3.huawei.com', enabled=True,
              content='完整业务背景不应逐步重复注入')


@pytest.mark.parametrize('kind,value', [('url_host', 'w3.huawei.com'),
    ('url_equals', 'https://w3.huawei.com/3ms?lang=zh')])
def test_real_decision_input_contains_selected_entry_and_current_destination(kind, value):
    p, model, env = pipeline([], [])
    p._knowledge_selection = {'entries': [PORTAL]}
    predicate = {'kind': kind, 'host' if kind == 'url_host' else 'url': value}
    state = StepState(task='去公司w3网站看一下在3MS可以做什么事情', obs=env.obs)
    state.queue = ExecutionQueue({'steps': [step('打开公司网站', [predicate]),
        step('后续任务不可暴露', [{'kind': 'url_host', 'host': 'later.example.test'}])]})
    original = copy.deepcopy(state.queue.plan)
    for revision in (0, 1):
        if revision:
            state.queue = ExecutionQueue(copy.deepcopy(original), revision=revision)
        text = p.build_input(state)[-1]['content']
        assert '公司 W3 门户' in text and 'https://w3.huawei.com' in text
        assert value in text
        assert '本步目标域名：' in text if kind == 'url_host' else '本步目标网址：' in text
        assert PORTAL['content'] not in text
        assert 'later.example.test' not in text and '后续任务不可暴露' not in text
        assert 'verification' not in text
    assert not model.calls  # No extra inference for carrying operands.


def test_full_episode_first_decision_gets_kb_even_when_goal_omits_url(monkeypatch):
    from osworld_agent import knowledge_base
    from test_knowledge_base import snapshot
    monkeypatch.setattr(knowledge_base, 'load_knowledge', lambda: snapshot(PORTAL))
    p, model, _ = pipeline([{'type': 'goto', 'url': PORTAL['url']}],
        [step('打开公司w3网站', [{'kind': 'url_host', 'host': 'w3.huawei.com'}])])
    p.run('去公司w3网站看一下在3MS可以做什么事情')
    prompt = model.calls[1][-1]['content']
    assert '已知入口' in prompt and PORTAL['url'] in prompt
    assert '本步目标域名：w3.huawei.com' in prompt


def test_compact_entries_keep_semantic_selection_and_skip_disabled_and_bodies():
    semantic = {**PORTAL, 'title': '内部平台', 'aliases': [], 'keywords': []}
    disabled = {**PORTAL, 'url': 'https://disabled.test', 'enabled': False}
    text = '\n'.join(navigation_brief('查看业务说明', None, {'entries': [semantic, disabled]}))
    assert PORTAL['url'] in text  # Do not require another alias match.
    assert 'disabled.test' not in text and PORTAL['content'] not in text
    assert navigation_brief('查看业务说明', None, {'entries': []}) == []


def test_many_entries_prioritize_current_host_and_never_cut_urls():
    entries = [{**PORTAL, 'title': '其他系统', 'aliases': [], 'keywords': [],
                'url': f'https://other{i}.test'} for i in range(50)] + [PORTAL]
    text = '\n'.join(navigation_brief('查看门户',
        step(criteria=[{'kind': 'url_host', 'host': 'w3.huawei.com'}]), {'entries': entries}))
    assert PORTAL['url'] in text and '其余知识入口已省略' in text
    assert len(text) < 5000


def test_explicit_user_url_is_preserved_with_priority_over_kb_and_plan():
    p, _, env = pipeline([], [])
    p._knowledge_selection = {'entries': [PORTAL]}
    task = '只访问 https://explicit.example.test/company 查看功能'
    state = StepState(task=task, obs=env.obs)
    state.queue = ExecutionQueue({'steps': [step('访问门户',
        [{'kind': 'url_host', 'host': 'w3.huawei.com'}])]})
    text = p.build_input(state)[-1]['content']
    assert task in text
    assert '原始任务明确指定的网址/目标优先于知识和计划' in text
