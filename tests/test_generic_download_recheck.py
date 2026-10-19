"""Receiving another file and rechecking the received file are distinct facts."""
import hashlib
import zipfile

import pytest

from osworld_agent.env import Observation
from osworld_agent.execution_queue import ExecutionQueue, normalize_verification
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import Execution, Pipeline, StepState
from osworld_agent.plan_compiler import compile_plan
from osworld_agent.step_verifier import code_review
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import Env, Model


def plan(*goals):
    return compile_plan({"steps": [
        {"goal": goal, "expected_evidence": "实际下载文件已完整保存", "verification": [{"kind": "visual"}]}
        for goal in goals]}, "去方案库下载 PPT", {"checks": []}, {"kind": "generic"}, [])


def artifact(path, ident="download-1"):
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("[Content_Types].xml", "<Types/>")
        package.writestr("ppt/presentation.xml", "<presentation/>")
    stat = path.stat()
    return {"artifact_id": ident, "download_id": ident, "transport": "browser", "completed": True,
            "path": str(path), "bytes": stat.st_size, "file_identity": [stat.st_dev, stat.st_ino],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "url": "https://example.test/" + path.name}


def test_receive_then_recheck_advances_without_a_second_download(tmp_path, monkeypatch):
    monkeypatch.setattr("osworld_agent.pipeline._viz_log_status", None)
    proposal = plan("下载一个 PPT", "检查 PPT 下载成功")
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt="test")
    pipe.completion = CompletionVerifier("去方案库下载 PPT")
    state = StepState(task="去方案库下载 PPT", memory=StructuredMemory(),
                      obs=Observation(info={"artifacts": [artifact(tmp_path / "slides.pptx")]}))
    state.memory.plan = proposal
    state.queue = ExecutionQueue(proposal)
    execution = Execution(kind="framework_observation", obs=state.obs, info={"status": "observed"})
    pipe._update_plan_progress(state, execution)
    assert state.queue.cursor == 1
    assert state.consumed_downloads == {"download-1"}
    pipe._update_plan_progress(state, execution)
    assert state.queue.current is None
    assert state.step_review["status"] == "passed"
    # A transport milestone does not claim that all semantic requirements passed.
    assert pipe.completion.evaluate()["status"] == "unknown"


def test_ready_download_milestones_reconcile_without_another_policy_turn(tmp_path, monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status',None)
    proposal=plan('下载一个 PPT','检查 PPT 下载成功')
    pipe=Pipeline(model=Model([]),env=Env(),grounding=None,system_prompt='test')
    pipe.completion=CompletionVerifier('去方案库下载 PPT')
    state=StepState(task='去方案库下载 PPT',memory=StructuredMemory(),step_id=4,
                   obs=Observation(info={'artifacts':[artifact(tmp_path/'slides.pptx')]}))
    state.memory.plan=proposal;state.queue=ExecutionQueue(proposal)
    pipe._advance_ready_download_steps(state)
    assert state.queue.current is None
    assert state.step_id==4 and not pipe.model.calls
    assert state.consumed_downloads=={'download-1'}


@pytest.mark.parametrize("goal", ["下载另一个 PPT", "下载第二个 PPT", "重新下载 PPT",
                                  "检查后下载另一个 PPT", "download another PPT"])
def test_another_download_cannot_reuse_consumed_artifact(tmp_path, goal):
    steps = plan("下载一个 PPT", goal)["steps"]
    event = artifact(tmp_path / "slides.pptx")
    obs = Observation(info={"artifacts": [event]})
    assert "purpose" not in steps[1]["verification"][0]
    assert code_review(steps[1], obs, {}, {"consumed_artifact_ids": ["download-1"]})["status"] == "unknown"
    event2 = artifact(tmp_path / "another.pptx", "download-2")
    obs.info["artifacts"].append(event2)
    assert code_review(steps[1], obs, {}, {"consumed_artifact_ids": ["download-1"]})["status"] == "passed"


@pytest.mark.parametrize("goal", ["检查 PPT 下载成功", "确认已下载 PPT 文件完整", "Verify downloaded PPT"])
def test_recheck_requires_previously_accepted_artifact(tmp_path, goal):
    step = plan(goal)["steps"][0]
    obs = Observation(info={"artifacts": [artifact(tmp_path / "slides.pptx")]})
    assert step["verification"] == [{"kind": "artifact_download", "format": "ppt", "purpose": "recheck"}]
    assert code_review(step, obs, {}, {})["status"] == "unknown"
    assert code_review(step, obs, {}, {"consumed_artifact_ids": ["invented"]})["status"] == "unknown"
    assert code_review(step, Observation(), {}, {"consumed_artifact_ids": ["download-1"]})["status"] == "unknown"
    assert code_review(step, obs, {}, {"consumed_artifact_ids": ["download-1"]})["status"] == "passed"


@pytest.mark.parametrize("change", ["bytes", "deleted", "replaced"])
def test_recheck_rereads_integrity_instead_of_latching_success(tmp_path, change):
    path = tmp_path / "slides.pptx"
    event = artifact(path)
    step = plan("检查 PPT 下载成功")["steps"][0]
    contract = {"consumed_artifact_ids": ["download-1"]}
    obs = Observation(info={"artifacts": [event]})
    assert code_review(step, obs, {}, contract)["status"] == "passed"
    if change == "bytes":
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif change == "deleted":
        path.unlink()
    else:
        path.rename(tmp_path / "old.pptx")
        path.write_bytes((tmp_path / "old.pptx").read_bytes())
    assert code_review(step, obs, {}, contract)["status"] == "unknown"


def test_planner_cannot_mark_an_acquisition_as_recheck_or_bind_own_file_id():
    proposal = {"steps": [{"goal": "下载第二个 PPT", "expected_evidence": "第二份真实文件",
        "verification": [{"kind": "artifact_download", "format": "ppt", "purpose": "recheck",
                          "artifact_id": "invented"}]}]}
    compiled = compile_plan(proposal, "下载两个 PPT", {"checks": []}, {"kind": "generic"}, [])
    assert compiled["steps"][0]["verification"] == [{"kind": "artifact_download", "format": "ppt"}]
    assert normalize_verification([{"kind": "artifact_download", "format": "ppt"}]) == [
        {"kind": "artifact_download", "format": "ppt"}]
    with pytest.raises(ValueError, match="purpose"):
        normalize_verification([{"kind": "artifact_download", "purpose": "arbitrary"}])
