"""Pure plan compilation preserves trusted contracts and existing refinements."""
import copy

import pytest

from osworld_agent.execution_queue import MAX_PLAN_STEPS
from osworld_agent.plan_compiler import compile_plan, normalize_plan_actions


def step(goal="检查页面", **extra):
    return {"goal": goal, "expected_evidence": "页面显示实际结果", **extra}


def compile_generic(obj, task="检查页面", public_spec=None):
    return compile_plan(obj, task, public_spec or {"checks": []}, {"kind": "generic"}, [])


def test_trusted_contract_is_copied_and_planner_contract_cannot_replace_it():
    spec = {"checks": [{"id": "system_1", "kind": "url_equals", "url": "https://example.test/right"}]}
    proposal = {"steps": [step(check_ids=["system_1"])],
                "completion_spec": {"checks": []}, "final_check": "模型声称全部完成"}
    before = copy.deepcopy((spec, proposal))
    plan = compile_generic(proposal, public_spec=spec)
    assert (spec, proposal) == before
    assert plan["completion_spec"] == spec and plan["completion_spec"] is not spec
    assert plan["steps"][0]["check_ids"] == ["system_1"]
    assert plan["steps"][0]["verification"] == []
    assert plan["steps"][0]["status"] == "pending"
    plan["completion_spec"]["checks"].clear()
    assert spec["checks"]


@pytest.mark.parametrize("task,fmt", [("随便下载一个 PPT", "ppt"), ("下载 PDF", "pdf"), ("下载文件", "any")])
def test_generic_download_requires_artifact_even_with_hallucinated_checks(task, fmt):
    proposal = {"steps": [step("下载目标文件", check_ids=["invented"], verification=[
        {"kind": "download_complete", "check_id": "invented"}, {"kind": "visual"},
        {"kind": "url_host", "host": "example.test"}])]}
    before = copy.deepcopy(proposal)
    plan = compile_generic(proposal, task=task)
    assert proposal == before
    assert plan["steps"][0]["check_ids"] == []
    assert plan["steps"][0]["verification"] == [
        {"kind": "url_host", "host": "example.test"}, {"kind": "artifact_download", "format": fmt}]
    assert plan["completion_spec"] == {"checks": []}


def test_non_download_generic_step_does_not_invent_artifact_requirement():
    plan = compile_generic({"steps": [step("查看页面", verification=[
        {"kind": "download_valid", "check_id": "invented"}])]})
    assert plan["steps"][0]["verification"] == [{"kind": "visual"}]


def test_locating_download_link_does_not_consume_the_download_receipt():
    plan = compile_generic({'steps': [
        step('在页面中定位 PDF 全文下载链接', phase='locate_fulltext', verification=[{'kind':'visual'}]),
        step('点击 PDF 链接触发下载，保存至默认目录', phase='download', verification=[{'kind':'visual'}]),
        step('确认 PDF 文件已完整下载至默认目录', phase='verify', verification=[{'kind':'visual'}]),
    ]}, task='下载 PDF')
    assert plan['steps'][0]['verification'] == [{'kind':'visual'}]
    assert plan['steps'][1]['verification'] == [{'kind':'artifact_download','format':'pdf'}]
    assert plan['steps'][2]['verification'] == [{'kind':'artifact_download','format':'pdf','purpose':'recheck'}]
    acquisition = compile_generic({'steps':[step('找到下载链接并下载 PDF', verification=[{'kind':'visual'}])]}, task='下载 PDF')
    assert acquisition['steps'][0]['verification'][0]['kind'] == 'artifact_download'


def test_copied_public_unresolved_requirement_remains_semantic_not_passed():
    requirement={'id':'check_1','kind':'unresolved_requirement','text':'核对论文身份'}
    spec={'checks':[requirement]}
    plan=compile_generic({'steps':[step('核对论文身份',verification=[requirement])]}, public_spec=spec)
    assert plan['steps'][0]['verification']==[{'kind':'visual'}]
    assert plan['completion_spec']==spec
    with pytest.raises(ValueError):
        compile_generic({'steps':[step(verification=[{**requirement,'text':'伪造完成条件'}])]},public_spec=spec)


def test_opening_a_file_is_executed_before_automatic_reporting():
    plan=compile_generic({'steps':[step('打开文件并展示报告',phase='report',actions=['cli'],
                                      verification=[{'kind':'app_open','app':'TextEdit'}])]})
    assert plan['steps'][0]['phase']=='deliverable_operation'
    assert plan['steps'][0]['actions']==['cli']
    assert plan['steps'][0]['verification']==[{'kind':'app_open','app':'TextEdit'}]


def test_paper_refinement_precedes_validation_and_preserves_observed_identity_workflow():
    phases = ["search", "identify", "locate_fulltext", "download", "verify"]
    profile = {"kind": "paper_download", "workflow": [{"phase": p} for p in phases]}
    goals = [{"check_id": "paper_1", "query": "Attention Is All You Need"}]
    spec = {"checks": [{"id": "paper_1", "kind": "paper_download", "query": goals[0]["query"]}]}
    proposal = {"steps": [step("预设 Vaswani 和 1706.03762", phase=p,
                                verification=[{"kind": "element_state"}], check_ids=["paper_1"])
                           for p in phases], "keywords": ["Vaswani"], "final_check": "已点击下载即可"}
    before = copy.deepcopy((proposal, profile, goals, spec))
    plan = compile_plan(proposal, "搜索论文Attention Is All You Need并下载", spec, profile, goals)
    assert (proposal, profile, goals, spec) == before
    assert [s["phase"] for s in plan["steps"]] == [
        "search", "identify", "locate_fulltext", "download", "inspect_download", "verify"]
    assert "预设 Vaswani" not in plan["steps"][0]["goal"]
    assert "不编造作者或论文 ID" in plan["steps"][0]["goal"]
    assert plan["steps"][0]["verification"] == [{"kind": "search_results", "query": goals[0]["query"]}]
    assert plan["steps"][3]["verification"] == [{"kind": "download_complete", "check_id": "paper_1"}]
    assert plan["steps"][4]["verification"] == [{"kind": "download_valid", "check_id": "paper_1"}]
    assert plan["keywords"] == [goals[0]["query"]]
    assert plan["final_check"] == "框架确认目标论文 PDF 已保存到默认工程目录且内容身份匹配"


def test_requirement_annotations_remain_untrusted_and_invalid_provenance_is_rejected():
    task = "下载文件，然后查看页面"
    proposal = {"steps": [step()], "requirements": [
        {"kind": "other", "source_text": "查看页面"},
        {"kind": "paper_download", "source_text": "下载文件", "value": "模型编造论文"}]}
    plan = compile_generic(proposal, task=task)
    audit = plan["requirements"]
    assert audit["accepted"][0]["trusted"] is False
    assert audit["accepted"][0]["source_start"] == task.index("查看页面")
    assert len(audit["rejected"]) == 1
    assert audit["coverage_verified"] is False
    assert plan["completion_spec"]["checks"] == []


@pytest.mark.parametrize("proposal,error", [
    ({"steps": []}, f"steps 必须是非空子目标列表；资源上限为 {MAX_PLAN_STEPS} 项，请合并同一目标的零碎动作"),
    ({"steps": [step(goal=" ")]}, "每一步需要非空 goal"),
    ({"steps": [step(expected_evidence="")]}, "每一步必须给出非空 expected_evidence，说明实际可以观察到的结果"),
    ({"steps": [step(actions=["run_in_terminal"])]}, "actions 只能包含已注册的 GUI/DOM/CLI 动作"),
    ({"steps": [step(verification=[{"kind": "not_supported"}])]}, "第 1 个子目标 verification：未知的步骤 verification.kind"),
])
def test_original_validation_errors_are_preserved(proposal, error):
    with pytest.raises(ValueError) as caught:
        compile_generic(proposal)
    assert str(caught.value) == error


@pytest.mark.parametrize("extra,error", [
    ({"check_ids": ["invented"]}, "check_ids 只能引用给定的系统验收条件；没有时用 []"),
    ({"verification": [{"kind": "download_complete", "check_id": "invented"}]}, "下载谓词 check_id 必须引用已有系统条件"),
])
def test_planner_cannot_reference_unknown_trusted_check(extra, error):
    with pytest.raises(ValueError) as caught:
        compile_generic({"steps": [step(**extra)]}, public_spec={"checks": [{"id": "system_1"}]})
    assert str(caught.value) == error


def test_paper_repair_requires_verify_but_does_not_require_completed_phases_again():
    profile = {"kind": "paper_download", "workflow": [{"phase": p} for p in ("search", "verify")]}
    obj = {"steps": [step(phase="verify")], "keywords": ["query"]}
    plan = compile_plan(obj, "论文任务", {"checks": []}, profile, [], repair=True)
    assert len(plan["steps"]) == 1
    with pytest.raises(ValueError, match="论文下载规划缺少流程阶段：search"):
        compile_plan(obj, "论文任务", {"checks": []}, profile, [])


@pytest.mark.parametrize("phases,keywords,error", [
    (["unknown"], ["query"], "论文计划包含未知流程阶段"),
    (["verify", "search"], ["query"], "流程阶段顺序必须为 search → verify"),
    (["search", "verify"], [], "论文搜索必须提供非空 keywords"),
])
def test_paper_phase_and_keyword_errors_are_preserved(phases, keywords, error):
    profile = {"kind": "paper_download", "workflow": [{"phase": p} for p in ("search", "verify")]}
    with pytest.raises(ValueError) as caught:
        compile_plan({"steps": [step(phase=p) for p in phases], "keywords": keywords},
                     "论文任务", {"checks": []}, profile, [])
    assert str(caught.value) == error


def test_action_aliases_and_deduplication_match_existing_planner():
    assert normalize_plan_actions(["open_url", "navigate", "input_text", "type_text", "key_press", "press_key"]) == [
        "goto", "type", "press"]


def test_unknown_server_filename_uses_download_receipt_not_guessed_path():
    proposal={'steps':[step(goal='核查下载的论文 PDF', phase='verify', actions=['request_finish'],
        verification=[{'kind':'file_exists','path':'2603.11558.pdf'}])]}
    plan=compile_generic(proposal, task='打开 https://arxiv.org/abs/2603.11558 下载论文 PDF')
    assert plan['steps'][0]['verification']==[{'kind':'artifact_download','format':'pdf','purpose':'recheck'}]
    explicit=compile_generic(proposal,task='下载论文 PDF，然后改名为2603.11558.pdf')
    assert {'kind':'file_exists','path':'2603.11558.pdf'} in explicit['steps'][0]['verification']


def test_unresolved_task_still_gets_a_browser_download_milestone():
    proposal={'steps':[step(goal='下载文件 PDF', phase='download', actions=['click'], verification=[{'kind':'visual'}])]}
    plan=compile_plan(proposal, '下载 PDF', {'checks':[{'id':'unknown','kind':'unresolved_requirement','text':'下载 PDF'}]},
                      {'kind':'generic'}, [])
    assert plan['steps'][0]['verification']==[{'kind':'artifact_download','format':'pdf'}]


def test_saved_pdf_inspection_reuses_receipt_without_new_acquisition():
    proposal = {'steps': [step(goal='确认 PDF 文件已保存至指定目录', phase='verify',
        verification=[{'kind': 'artifact_download', 'format': 'pdf'}])]}
    plan = compile_generic(proposal, task='下载指定论文 PDF 到默认目录')
    assert plan['steps'][0]['verification'] == [
        {'kind': 'artifact_download', 'format': 'pdf', 'purpose': 'recheck'}]
