"""Planner/decision prompts preserve partial navigation and real download facts."""
import json
from types import SimpleNamespace

from osworld_agent.decision_context import CORE_PROMPT, current_brief
from osworld_agent.env import Observation
from osworld_agent.prompts import build_planner_prompt
from test_download_facts import completed
from test_planning_service import Model, VALID, accepting, service, state


def decision_state(facts):
    return SimpleNamespace(queue=SimpleNamespace(current={
        'goal': '核查已下载 PPT 是否满足要求', 'expected_evidence': '完整文件符合用户要求'}),
        download_facts=facts, execution_feedback={}, step_review={}, finish_feedback=None,
        recovery={}, step_progress={}, routing_hint={})


def test_generic_plan_distinguishes_navigation_location_and_download_without_guessing_contracts():
    task = '到指定门户的方案库随便下载一个 PPT'
    spec = {'checks': []}
    prompt = build_planner_prompt(task, spec, {'kind': 'generic'})
    assert '不要把“进入方案库”和“定位一个可下载 PPT”捆成一个子目标' in prompt
    assert 'URL 变化不等于附件可下载' in prompt
    assert 'href/targetHref 对应的明确目标 URL' in prompt
    assert '不需要虚构 check_id' in prompt
    assert '尚未观察到的未来地址不得猜测' in prompt
    assert '当前视口没有 PPT，不能断言全库没有 PPT' in prompt
    assert '类型筛选' in prompt and '卡片详情检查附件' in prompt
    assert '不要按单次鼠标动作拆步骤' in prompt
    assert 'artifact_download(format)' in prompt
    assert spec == {'checks': []}
    assert 'itsc' not in prompt.lower() and 'huawei' not in prompt.lower()


def test_planner_receives_observed_link_target_instead_of_guessing_future_navigation():
    model, events, calls, accepted = Model([VALID]), [], [], []
    planner, record = service(model, events)
    current = state()
    href = 'https://portal.example.test/#/solutions'
    current.obs = Observation(context=[{'kind': 'browser_ax',
        'url': 'https://portal.example.test/', 'title': 'Portal',
        'nodes': [{'role': 'link', 'name': 'Solution Library', 'href': href}]}])
    assert planner.plan('进入方案库查找 PPT', current, {'checks': []}, [], calls, {},
                        accepting(events, record, accepted))
    prompt = model.calls[0][0][0]['content']
    assert json.dumps({'role': 'link', 'name': 'Solution Library', 'href': href}) in prompt
    # Observed navigation facts do not become trusted completion checks.
    assert accepted[0]['steps'][0]['check_ids'] == []


def test_replan_and_next_decision_keep_verified_file_when_popup_close_hides_event(tmp_path):
    verifier, path, event = completed(tmp_path)
    verifier.observe({'info': {}, 'context': [{'kind': 'browser_ax',
        'url': 'https://example.test/library', 'active_modals': [], 'nodes': []}]})
    facts = verifier.download_facts()
    model, events, calls, accepted = Model([{**VALID, 'strategy_change':
        '下载产物已经落盘，保留现有文件，只核对剩余内容要求，不再次触发下载。'}]), [], [], []
    planner, record = service(model, events)
    current = state()
    current.download_facts = facts
    current.obs = Observation(info={})
    assert planner.plan('到方案库下载 PPT', current, {'checks': []}, [], calls, {},
        accepting(events, record, accepted), repair=True,
        verification_feedback={'status': 'unknown', 'checks': []})
    planner_prompt = model.calls[0][0][0]['content']
    brief = current_brief(decision_state(facts))
    for prompt in (planner_prompt, brief):
        assert json.dumps(facts, ensure_ascii=False) in prompt
        assert event['artifact_id'] in prompt and str(path) in prompt
    assert '只规划剩余的内容、格式、身份、命名或其他业务要求' in planner_prompt
    assert '不要重新下载同一文件' in brief
    assert verifier.evaluate()['status'] == 'unknown'


def test_failed_persistent_file_evidence_is_not_presented_as_completed_task(tmp_path):
    verifier, path, _ = completed(tmp_path)
    path.unlink()
    facts = verifier.download_facts()
    brief = current_brief(decision_state(facts))
    assert facts[0]['status'] == 'failed'
    assert json.dumps(facts, ensure_ascii=False) in brief
    assert '失败、缺失或不符要求的文件不能复用为成功证据' in brief
    assert '这不等于全部业务条件通过' in brief


def test_generic_execution_guides_change_of_search_path_without_overclaiming_library_absence():
    assert '进入目标区域、定位具体文件或详情附件、完成下载是不同结果' in CORE_PROMPT
    assert '不要因尚未找到文件而反复点击已选导航' in CORE_PROMPT
    assert '当前视口或列表页没有 PPT 不能证明全库没有' in CORE_PROMPT
    assert '类型筛选或卡片详情附件' in CORE_PROMPT
    assert '滚动连续无新内容时换策略' in CORE_PROMPT
