"""主循环流水线：观察 → 决策（证据先行）→ 定位 → 执行 → 控制状态更新。

闭环设计（核心）：
- 操作员输出 next_action 和一句话 summary；
- 执行后由框架代码或独立核查员检查结果，维护执行队列；
- 完整计划、验收证据与控制状态由框架保存，操作员只接收当前子目标；
- request_finish 只是请求，框架验证后才真正结束。

阶段（run() 里按序调用）：
  ① preprocess   预处理：确保观察 + 记录截图 + 构建记忆文本
  ② build_input  构造输入：当前子目标 + 近期摘要 + 按预算选择的结构和截图
  ③ decide       决策：next_action + summary（兼容旧字段）
  ④ ground       grounding：描述→坐标，记录坐标供下一步标记
  ⑤ execute      执行：skill / shell / subagent / terminal(request_finish|fail) / env
  ⑥ postprocess  后处理：记录 expected/observed + 控制状态 + 卡死检测 + Finish 验证
"""
from __future__ import annotations

import copy
import json
import time
import hashlib
import math
import re
import unicodedata
import uuid
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .actions import (
    Action, validate_action, action_signature, get_action_spec, ACTION_REGISTRY,
    parse_decision, action_from_decision,
)
from .env import Environment, Observation
from .model import ChatModel, Grounding, resolve_coords, mark_coordinate
from .model.decision_model import encode_image, ImageEncodingError
from .scene import scene_hint
from .memory import StructuredMemory
from .skills import SkillContext
from .prompts import build_format_feedback, build_planner_prompt
from .context import PromptCompiler, ContextBudgetError, ObservationIncompleteError
from .verifiers import CompletionVerifier
from .task_specs import task_profile, task_analysis, audit_requirement_proposals, TASK_ANALYSIS_VERSION
from .step_progress import ProgressConfig, StepProgressMonitor
from .model.decision_judge import DecisionJudge, DecisionJudgeConfig
from .paper_workflow import PaperWorkflow, WORKFLOW_VERSION, binding_group_hash
from .paper_resolver import ADAPTER_VERSION
from .execution_queue import ExecutionQueue, normalize_verification, MAX_PLAN_STEPS
from .decision_context import action_scope, scoped_system, selected_memory, current_brief
from .step_verifier import StepReviewer, code_review, make_review_model
from .actions import BASIC_ACTIONS
from .browser_feedback import active_paper, paper_action_error, operational_facts, exposed_nodes, paper_review_error
from .visual_readiness import is_pdf_observation, content_thumbnail, stable_content

# 实时状态推送（可选）：execute 阶段进度推给 viz 前端，失败不影响主流程
try:
    from .viz.logging import log_status as _viz_log_status
except Exception:  # noqa: BLE001
    _viz_log_status = None

# 确定性卡死检测参数
_NO_CHANGE_DIFF_THRESHOLD = 2.0       # 64x64 灰度 mean-abs-diff 阈值（< 此值视为无变化）
_LOOP_HISTORY_LEN = 6                 # 周期循环检测的历史长度
_FAILURE_REPEAT_LIMIT = 2              # 执行失败/解析失败只计一层，不再重复累积 STALL


# --------------------------------------------------------------------------- #
# 数据容器
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WaitConfig:
    """Framework waits are bounded wall time, never extra policy decisions."""
    enabled: bool = True
    poll_interval_s: float = 0.25
    download_start_timeout_s: float = 2.0
    render_timeout_s: float = 20.0
    operation_timeout_s: float = 300.0
    task_timeout_s: float = 1800.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("wait_config.enabled must be boolean")
        for name in ("poll_interval_s", "download_start_timeout_s", "operation_timeout_s", "task_timeout_s", "render_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"wait_config.{name} must be a finite nonnegative number")
            if name != "download_start_timeout_s" and value == 0:
                raise ValueError(f"wait_config.{name} must be positive")


@dataclass
class AgentResult:
    """单任务运行结果。"""
    task: str
    success: bool = False
    score: float = 0.0
    steps: int = 0
    final_answer: str = ""
    trajectory: List[dict] = field(default_factory=list)
    screenshots: Dict[int, Any] = field(default_factory=dict)   # step -> 决策前截图（PIL 或 bytes）
    plan: Optional[Dict[str, Any]] = None   # 阶段0 产出的任务计划（steps/keywords/final_check）
    evaluation_available: bool = False
    reward_source: str = "unavailable"
    termination_reason: str = "unknown"
    verification: Dict[str, Any] = field(default_factory=dict)
    planning_calls: List[dict] = field(default_factory=list)
    run_id: str = ""
    run_manifest: Dict[str, Any] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)
    private_evaluation: Dict[str, Any] = field(default_factory=dict)
    review_calls: List[dict] = field(default_factory=list)


@dataclass
class StepState:
    """跨步骤流动的循环状态。"""
    task: str = ""
    obs: Optional[Observation] = None
    memory: Optional[StructuredMemory] = None
    skill_ctx: Optional[SkillContext] = None
    skill_results: Dict[str, str] = field(default_factory=dict)
    terminal_result: str = ""
    code_agent_result: str = ""
    page_source: str = ""
    browser_result: str = ""
    memory_text: str = ""
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)
    prompt_text: str = ""                     # 本步喂给决策模型的输入文本（任务+记忆+结构观察+工具结果）
    route: Optional[Dict[str, Any]] = None
    resolved_action: Optional[Any] = None
    # 视觉记忆
    screenshot_history: List[Any] = field(default_factory=list)
    last_grounding_coord: Optional[Tuple[int, int]] = None
    last_grounding_target: str = ""
    # 程序维护的信息
    step_id: int = 0
    last_action: str = ""
    last_target: str = ""
    execution_status: str = ""
    failure_count: int = 0
    # 卡死检测
    no_change_count: int = 0
    action_keys: List[str] = field(default_factory=list)
    shell_outputs: List[str] = field(default_factory=list)   # 最近 shell 命令输出（用于 shell 卡死检测）
    verification: Dict[str, Any] = field(default_factory=dict)
    task_profile: Dict[str, Any] = field(default_factory=dict)
    recovery: Dict[str, Any] = field(default_factory=dict)
    phase_events: List[dict] = field(default_factory=list)
    execution_feedback: Dict[str, Any] = field(default_factory=dict)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    failure_class: str = ""
    failure_key: str = ""
    semantic_targets: Dict[str, Any] = field(default_factory=dict)
    waiting_download_ids: set = field(default_factory=set)
    last_browser_page: Dict[str, Any] = field(default_factory=dict)
    queue: Optional[ExecutionQueue] = None
    step_review: Dict[str, Any] = field(default_factory=dict)
    progress_monitor: Optional[StepProgressMonitor] = None
    step_progress: Dict[str, Any] = field(default_factory=dict)
    routing_hint: Dict[str, Any] = field(default_factory=dict)
    finish_feedback: str = ""
    render_waits: Dict[tuple, str] = field(default_factory=dict)
    operation_facts: List[str] = field(default_factory=list)


@dataclass
class Decision:
    """③ 决策阶段的产出（先分析当前屏幕 → 证据先行 → 预期效果）。"""
    raw: str = ""
    action: Optional[Action] = None
    summary: str = ""                    # 一句话观察与行动计划；非完成证据
    screen_analysis: str = ""            # 当前屏幕状态描述（含打开的应用）
    observed_effect: str = ""            # 上一步动作的实际观察（先客观描述）
    action_status: str = ""              # success / failure / uncertain
    state_update: Dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""            # 下一步动作的预期效果
    calls: List[dict] = field(default_factory=list)   # 每次决策调用的完整记录（含重试）


@dataclass
class Execution:
    """⑤ 执行阶段的产出。"""
    kind: str = ""                        # skill / shell / subagent / terminal / finish_request / env
    outcome: str = ""
    executed: str = ""
    obs: Optional[Observation] = None
    done: bool = False
    success: bool = False
    answer: str = ""
    info: Dict[str, Any] = field(default_factory=dict)   # 执行器返回的详细信息（browser_dom 状态等）
    reward: float = 0.0                                  # 环境奖励（browser_dom 透传 step_result.reward）


# --------------------------------------------------------------------------- #
# 主循环流水线
# --------------------------------------------------------------------------- #
class Pipeline:
    """主循环流水线。run() 只列 6 个阶段。"""

    def __init__(self, *, model: ChatModel, grounding: Grounding, env: Environment,
                 system_prompt: str, code_agent=None, max_steps: int = 20,
                 max_retries: int = 3, enable_reflection: bool = True,
                 early_stop_repeat: int = 3, step_callback=None,
                 completion_spec=None, context_config=None, private_completion_spec=None,
                 planner_model=None, event_path=None, run_id=None, run_manifest=None, max_replans=1,
                 wait_config=None, verifier_model=None, progress_config=None, decision_judge=None):
        self.model = model
        self.grounding = grounding
        self.env = env
        self.code_agent = code_agent
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.enable_reflection = enable_reflection
        self.early_stop_repeat = early_stop_repeat
        self.step_callback = step_callback  # 每步结束后回调(step_no, decision, execution, state)
        self.compiler = PromptCompiler(context_config)
        self.completion_spec = completion_spec
        self.private_completion_spec = private_completion_spec
        self.planner_model = planner_model or model
        self.event_path = event_path
        self.run_id = run_id or uuid.uuid4().hex
        self.run_manifest = copy.deepcopy(run_manifest or {})
        self.event_writer = None
        self.events = []
        self._current_step = 0
        self.max_replans = max(0, int(max_replans))
        self._replans = 0
        self.completion = None
        self.planning_calls = []
        self._verification = None         # 最近一次 finish 验证的调用记录
        self.wait_config = wait_config if isinstance(wait_config, WaitConfig) else WaitConfig(**(wait_config or {}))
        self.progress_config = progress_config if isinstance(progress_config, ProgressConfig) else ProgressConfig(**(progress_config or {}))
        judge_config = decision_judge if isinstance(decision_judge, DecisionJudgeConfig) else DecisionJudgeConfig(**(decision_judge or {}))
        self.decision_judge = DecisionJudge(judge_config) if judge_config.url else None
        self._task_deadline = None
        self.verifier_model = verifier_model if verifier_model is not None else make_review_model(model)
        self.reviewer = StepReviewer(self.verifier_model, self.compiler)
        self._active_actions = None
        self._model_final_report = None

    # ================= 主循环 ================= #
    def run(self, task: str) -> AgentResult:
        self._task_deadline = time.monotonic() + self.wait_config.task_timeout_s
        self._replans = 0
        self._current_step = 0
        self.reviewer.calls.clear()
        self._model_final_report = None
        self.completion = CompletionVerifier(task, self.completion_spec, self.private_completion_spec)
        self._paper_workflow = PaperWorkflow()
        self.events = []
        manifest = self._make_manifest(task)
        result = AgentResult(task=task, run_id=self.run_id, run_manifest=manifest,
            private_evaluation={"spec": copy.deepcopy(self.private_completion_spec), "verifier_version": "3.1"})
        if self.event_path:
            from .training_events import TrainingEventWriter
            path = str(self.event_path).replace("{run_id}", self.run_id)
            if Path(path).suffix != ".jsonl":
                path = str(Path(path) / self.run_id / "events.jsonl")
            self.event_writer = TrainingEventWriter(path, run_id=self.run_id, manifest=manifest)
            self.events = self.event_writer.events
        else:
            self._emit("episode_start", manifest=manifest)
        result.events = self.events
        result.review_calls = self.reviewer.calls
        try:
            result = self._run(task, result)
            result.run_manifest["goal_bindings"] = self.completion.bindings
            result.run_manifest["goal_binding_hash"] = binding_group_hash(self.completion.bindings)
            self._emit("episode_end", outcome={
                "success": result.success, "score": result.score,
                "reward": result.score if result.evaluation_available else None,
                "reward_valid": result.evaluation_available, "reward_source": result.reward_source,
                "termination_reason": result.termination_reason, "verification": result.verification},
                goal_bindings=result.run_manifest["goal_bindings"],
                goal_binding_hash=result.run_manifest["goal_binding_hash"])
            return result
        except BaseException as exc:
            self._emit("episode_error", error_type=type(exc).__name__,
                       step=self._current_step, reward=None, reward_valid=False)
            raise
        finally:
            if self.event_writer:
                self.event_writer.close()

    def _emit(self, event_type, **payload):
        if self._current_step:
            payload.setdefault("parent_decision_id", f"a_{self._current_step}")
        if self.event_writer:
            return self.event_writer.emit(event_type, **payload)
        event = {"event_id": f"{self.run_id}:e{len(self.events) + 1}", "run_id": self.run_id,
                 "event_type": event_type, "timestamp": time.time(), "actor_role": "framework",
                 "source": "framework", **copy.deepcopy(payload)}
        self.events.append(event)
        return event

    def _make_manifest(self, task):
        def model_record(model):
            cfg = getattr(getattr(model, "inner", model), "cfg", None)
            return {key: getattr(cfg, key, None) for key in (
                "name", "policy_revision", "tokenizer_revision", "processor_revision",
                "chat_template_revision", "temperature", "max_tokens", "capture_rollout")}
        root = Path(__file__).resolve().parent
        source_hash = hashlib.sha256()
        for name in ("pipeline.py", "prompts.py", "context.py", "browser_session.py", "verifiers.py",
                     "execution_queue.py", "step_verifier.py", "step_progress.py", "decision_context.py", "browser_feedback.py", "visual_readiness.py", "memory/structured_memory.py", "desktop_state.py",
                     "adapters/windows_env.py", "adapters/vmware_env.py", "task_specs.py", "paper_workflow.py", "paper_resolver.py", "actions.py", "config/action.yaml", "model/decision_model.py", "model/decision_judge.py"):
            source_hash.update(name.encode())
            source_hash.update((root / name).read_bytes())
        manifest = {"run_id": self.run_id, "task": task, "framework_source_hash": source_hash.hexdigest(),
                    "compiler_version": self.compiler.version, "verifier_version": "3.1",
                    "paper_resolver_version": ADAPTER_VERSION, "paper_workflow_version": WORKFLOW_VERSION,
                    "decision_model": model_record(self.model), "planner_model": model_record(self.planner_model),
                    "planner_trainable": self.planner_model is self.model,
                    "grounder_trainable": False, "verifier_trainable": False,
                    "verifier_model": model_record(self.verifier_model), "max_steps": self.max_steps,
                    "public_completion": self.completion.public_spec,
                    "task_analysis": task_analysis(task, self.completion.public_spec),
                    "completion_hash": hashlib.sha256(json.dumps(self.completion.spec, sort_keys=True).encode()).hexdigest(),
                    **self.run_manifest}
        manifest["run_id"] = self.run_id
        runtime = {"context": asdict(self.compiler.config), "max_steps": self.max_steps,
                   "max_retries": self.max_retries, "max_replans": self.max_replans,
                   "early_stop_repeat": self.early_stop_repeat,
                   "failure_repeat_limit": _FAILURE_REPEAT_LIMIT, "wait_config": asdict(self.wait_config),
                   "progress_config": asdict(self.progress_config), "task_analysis_version": TASK_ANALYSIS_VERSION,
                   "decision_judge": self.decision_judge.config.public() if self.decision_judge else None,
                   "paper_resolver_version": ADAPTER_VERSION, "paper_workflow_version": WORKFLOW_VERSION,
                   "enable_reflection": self.enable_reflection, "verifier_model": model_record(self.verifier_model),
                   "decision_context_version": "current_step_brief_v2", "max_plan_steps": MAX_PLAN_STEPS, "step_verifier_version": "1.0",
                   "decision_model": model_record(self.model), "planner_model": model_record(self.planner_model)}
        manifest["runtime_config"] = runtime
        manifest["runtime_config_hash"] = hashlib.sha256(
            json.dumps(runtime, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return manifest

    def _record_call(self, call, actor_role="decision"):
        call.setdefault("call_id", f"{self.run_id}:call:{len(self.events) + 1}")
        call["actor_role"] = actor_role
        call["policy_trainable"] = actor_role == "decision" or (actor_role == "planner" and self.planner_model is self.model)
        self._emit("model_call", actor_role=actor_role, source="model", call=call)

    def _run(self, task: str, result: AgentResult) -> AgentResult:
        state = self._init_state(task)
        state.task_profile = task_profile(task, self.completion.public_spec)
        if self.decision_judge:
            state.routing_hint['task'] = self.decision_judge.classify_task(task, remaining_s=self._task_time_remaining())
            self._record_typed_judge()
            state.task_profile['execution_hint'] = state.routing_hint['task']
            result.run_manifest['task_routing_hint'] = copy.deepcopy(state.routing_hint['task'])
            self._publish_diagnostic('decision_judgment', judgment=state.routing_hint['task'])
        self._publish_diagnostic('task_analysis', analysis=task_analysis(task, self.completion.public_spec))
        unresolved = [c["text"] for c in self.completion.public_spec.get("checks", [])
                      if c.get("kind") == "unresolved_requirement"]
        if unresolved:
            # 这些要求无法确定性验收，交由 request_finish 时的「模型验收器」判断，不提前终止。
            self._emit("contract_unresolved", requirements=unresolved, policy_trainable=False)
        # 规划是执行前置条件。小模型规划失败时不盲目开始操作。
        planned = self._plan(task, state)
        result.plan = state.memory.plan
        result.planning_calls = self.planning_calls
        if not planned:
            result.termination_reason = "task_timeout" if self._task_time_remaining() <= 0 else "planning_failed"
            result.final_answer = "任务拆解失败，未执行任何操作。请检查规划模型输出。"
            result.verification = {"status": "unknown", "reason": result.termination_reason, "checks": []}
            return result

        for step_no in range(1, self.max_steps + 1):
            if self._task_time_remaining() <= 0:
                result.termination_reason = "task_timeout"
                break
            started = time.perf_counter()
            state.step_id = step_no
            self._current_step = step_no
            result.steps = step_no
            state = self.preprocess(state)
            if step_no == 1:
                initial = copy.deepcopy(getattr(self.env, "episode_manifest", {}) or
                                        getattr(self.env, "initial_state", {}) or {})
                if initial:
                    result.run_manifest["initial_state"] = initial
                    result.run_manifest["initial_state_id"] = initial.get("initial_state_id", "")
                    self._emit("environment_initialized", initial_state=initial)
            if state.obs is not None:
                result.screenshots[step_no] = state.obs.screenshot
            try:
                messages = self.build_input(state)
            except (ObservationIncompleteError, ImageEncodingError):
                state.obs = self._observe()
                state.screenshot_history[-1] = state.obs.screenshot
                try:
                    messages = self.build_input(state)
                except (ObservationIncompleteError, ImageEncodingError):
                    result.termination_reason = "observation_unavailable"
                    result.final_answer = "当前截图不可用，已停止操作并保存采集记录。"
                    self.completion.observe(None)
                    break
            except ContextBudgetError:
                result.termination_reason = "context_budget_exceeded"
                result.final_answer = "任务计划与当前截图超出输入预算，未继续执行。"
                break
            before = self._observation_record(state.obs, f"o_{step_no}")
            decision = self.decide(messages, state)
            for call in decision.calls:
                call.setdefault("actor_role", "decision")
                call.setdefault("provenance", "sampled")
            if decision.action is None:
                # Retain every sampled output, but bound format/transport recovery.
                state.obs = self._observe()
                self.completion.observe(state.obs)
                state.verification = self.completion.evaluate()
                failure = "model_transport" if decision.calls and decision.calls[-1].get("error_type") else "parse_failure"
                state.execution_status = "rejected"
                validation_error = decision.calls[-1].get('validation_error', '') if decision.calls else ''
                state.execution_feedback = {"status": "rejected", "dispatched": False,
                                            "error_code": failure, "reason": "未获得合法动作，未执行环境操作"
                                            + ("：" + validation_error[:200] if validation_error else "")}
                self._count_failure(state, failure, None, state.obs)
                self._track_step_progress(state, None, state.execution_feedback, state.obs,
                                          review={"plan_step_id": state.queue.current['id'] if state.queue and state.queue.current else 'final',
                                                  "status": "unknown"})
                result.trajectory.append({
                    "step": step_no, "error": "动作解析失败",
                    "raw": (decision.raw or "")[-500:],
                    "observation": before,
                    "policy_input": copy.deepcopy(decision.calls[-1]["messages"] if decision.calls else messages),
                    "policy_output": decision.raw,
                    "decision_calls": decision.calls,
                    "model_calls": decision.calls,
                    "prompt_metadata": state.prompt_metadata,
                    "valid_action": False,
                    "terminated": False, "truncated": False,
                    "next_observation": self._observation_record(state.obs, f"o_{step_no+1}"),
                    "execution_info": copy.deepcopy(state.execution_feedback),
                    "execution_status": "rejected", "verification": copy.deepcopy(state.verification),
                    "step_progress": copy.deepcopy(state.step_progress),
                })
                self._emit("transition", transition=result.trajectory[-1])
                state.memory.add_event("⚠️ 上一步动作解析失败，请重新输出合法的 next_action JSON")
                if state.verification.get("status") == "passed":
                    result.success = True
                    result.termination_reason = "verified"
                    break
                if self._task_time_remaining() <= 0:
                    result.termination_reason = "task_timeout"
                    break
                if state.memory.stall_count >= _FAILURE_REPEAT_LIMIT or state.step_progress.get('status') == 'stalled':
                    if self._recover_plan(state, result, step_no):
                        break
                continue
            ground_started = time.perf_counter()
            action = (self.ground(decision.action, state) if self._task_time_remaining() > 0
                      else copy.deepcopy(decision.action))
            self._emit("route", action=self._compact_action(decision.action),
                       routing=state.route, resolved_action=self._compact_action(action))
            ground_ms = (time.perf_counter() - ground_started) * 1000
            execute_started = time.perf_counter()
            # execute 阶段推送进度（下载等长耗时动作让前端看到"执行中"而非干等）
            if _viz_log_status is not None:
                try:
                    _viz_log_status({"event": "executing", "step": step_no,
                                     "action": decision.action.action if decision.action else ""})
                except Exception:  # noqa: BLE001
                    pass
            self._set_download_deadline()
            execution = (self.execute(action, state) if self._task_time_remaining() > 0 else
                         Execution(kind="deadline", info={"status": "rejected", "dispatched": False,
                                   "error_code": "task_timeout", "reason": "任务总时限已到，未执行新动作"}))
            execution = self._settle_execution(action, execution, state)
            execution = self._settle_rendering(execution, state)
            execution = self._advance_paper_workflow(action, execution, state)
            self._emit("execution", action=self._compact_action(state.resolved_action or action),
                       execution_info=execution.info, outcome=execution.outcome)
            if _viz_log_status is not None:
                try:
                    _viz_log_status({"event": "executed", "step": step_no,
                                     "status": (execution.info or {}).get("status", ""),
                                     "action": (execution.info or {}).get("action", "")})
                except Exception:  # noqa: BLE001
                    pass
            execute_ms = (time.perf_counter() - execute_started) * 1000
            state, should_break = self.postprocess(step_no, decision, execution, state, result)
            self._emit("verification", report=state.verification)
            result.steps = step_no
            entry = next((t for t in reversed(result.trajectory)
                          if t.get("action_id") == f"a_{step_no}"), None)
            if entry is not None:
                entry.update({
                    "observation": before,
                    "next_observation": self._observation_record(state.obs, f"o_{step_no+1}"),
                    "policy_input": copy.deepcopy(decision.calls[-1]["messages"] if decision.calls else messages),
                    "policy_output": decision.raw,
                    "decision_calls": decision.calls,
                    "model_calls": decision.calls,
                    "prompt_metadata": state.prompt_metadata,
                    "valid_action": True,
                    "routing": copy.deepcopy(state.route),
                    "execution_info": execution.info,
                    "framework_operations": copy.deepcopy(execution.info.get("framework_operations", [])),
                    "verification": copy.deepcopy(state.verification),
                    "terminated": bool(should_break and (result.success or execution.done)),
                    "truncated": bool(should_break and not (result.success or execution.done)),
                    "timing_ms": {"ground": round(ground_ms, 2), "execute_observe": round(execute_ms, 2),
                                  "total": round((time.perf_counter() - started) * 1000, 2)},
                })
                entry["step_reviews"] = [copy.deepcopy(e["report"]) for e in self.events
                    if e.get("event_type") == "step_verification" and e.get("report", {}).get("decision_step") == step_no
                    and e.get("report", {}).get("status") != "running"]
                entry["execution_status"] = execution.info.get("status", "unknown")
                entry["effect_verified"] = bool(execution.info.get("effect_verified", False))
                entry["schema_valid"] = True
                self._emit("transition", transition=entry)
            if self.step_callback is not None:
                try:
                    self.step_callback(step_no, decision, execution, state)
                except Exception:  # noqa: BLE001  回调失败不影响主流程
                    pass
            if should_break:
                break

        result.score = self._evaluate(result)
        if result.termination_reason == "unknown":
            result.termination_reason = "max_steps"
        # 补齐最后一步的 assessment（任务结束时没有下一步决策来评估 a_N）
        self._fill_final_assessment(result)
        transitions = [t for t in result.trajectory if "policy_input" in t]
        if transitions:
            last = transitions[-1]
            if result.success or result.termination_reason in {"evaluation_failed", "agent_failed", "environment_ended"}:
                last["terminated"] = True
            elif result.termination_reason in {"verification_changed", "evaluation_error"}:
                last["terminated"] = False
            last["verification"] = copy.deepcopy(result.verification)
            last["step_reviews"] = [copy.deepcopy(e["report"]) for e in self.events
                if e.get("event_type") == "step_verification" and e.get("report", {}).get("decision_step") == last["step"]
                and e.get("report", {}).get("status") != "running"]
            last["truncated"] = not last.get("terminated", False)
            last["done"] = True
            self._emit("transition_finalized", transition=last)
        return result

    @staticmethod
    def _observation_record(obs, observation_id):
        """把 Observation 转成可落盘的完整记录（截图 base64，dump 时外置去重）。"""
        if obs is None:
            return {"id": observation_id, "context": [], "screenshot": None}
        return {"id": observation_id, "context": copy.deepcopy(obs.context),
                "text": getattr(obs, "text", ""), "info": copy.deepcopy(obs.info),
                "screenshot": encode_image(obs.screenshot)}

    # ================= 六个阶段 ================= #
    def preprocess(self, state: StepState) -> StepState:
        if state.obs is None:
            state.obs = self.env.reset(state.task)
            self._set_download_deadline()
        state.obs = self._settle_rendering(Execution(kind="env", obs=state.obs), state).obs
        self.completion.observe(state.obs)
        state.verification = self.completion.evaluate()
        state.screenshot_history.append(state.obs.screenshot)
        if len(state.screenshot_history) > 3:
            state.screenshot_history = state.screenshot_history[-3:]
        state.memory_text = self._build_memory_text(state)
        self._emit("observation", observation_id=f"o_{state.step_id}",
                   observation=self._observation_record(state.obs, f"o_{state.step_id}"))
        return state

    def build_input(self, state: StepState) -> List[dict]:
        page_source = state.page_source
        state.page_source = ""  # 一次性返回后清空，避免每步重复塞大段源码
        browser_result = state.browser_result
        state.browser_result = ""  # 一次性返回后清空
        # 结构观察块：browser AX snapshot（context）+ 桌面 AX（text）
        blocks = list(state.obs.context) if state.obs else []
        observation_info = state.obs.info if state.obs else {}
        browser_events = {k: observation_info[k] for k in (
            "dialog", "dialogs", "tabs", "active_modals", "browser_events", "artifacts",
            "pending_downloads", "download_events", "download_candidates", "document_kind", "content_type", "screenshot_stale")
            if observation_info.get(k)}
        if browser_events:
            blocks.insert(0, {"kind": "browser_events", "text": json.dumps(browser_events, ensure_ascii=False)})
        if state.obs and state.obs.text:
            blocks.append({"kind": "desktop_a11y", "text": state.obs.text})
        # 显式场景状态：让模型明确知道「有结构走 DOM / 无结构走视觉」
        hint = scene_hint(state.obs)
        blocks.insert(0, {"kind": "scene", "text": "当前场景: " + json.dumps(hint, ensure_ascii=False)})
        tool_results = {k: v for k, v in {"page_source": page_source, "browser_result": browser_result}.items() if v}
        auxiliary = json.dumps(tool_results, ensure_ascii=False) if tool_results else ""
        self._active_actions = action_scope(state.queue.current if state.queue else None, state.obs,
                                            bool(getattr(self.env, "browser_only", False)))
        system = scoped_system(self._active_actions, state.obs, getattr(self.env, "os_name", "linux"))
        current = state.queue.current if state.queue else None
        facts = operational_facts(state.obs, self.completion,
                                  self.completion.public_report(state.verification) if self.completion else {}, current,
                                  os_name=getattr(self.env, "os_name", ""))
        if facts != state.operation_facts:
            state.operation_facts = facts
            self._publish_diagnostic('operation_feedback', facts=facts)
        required = current_brief(state, self.completion.public_spec.get('checks', []) if self.completion else [], facts)
        images = self._build_images(state) if getattr(self.model, "vision", True) else []
        if observation_info.get("dialog"):
            # Native JS dialogs may block a new screenshot. Their fresh event
            # permits only dialog recovery; any retained image is explicitly stale.
            images = [(label, shot) for label, shot in images if shot is not None]
        user_text, images, metadata = self.compiler.compile(
            system=system, task=state.task, memory=state.memory_text,
            blocks=blocks, auxiliary=auxiliary, focus=(current or {}).get("goal", ""),
            required=required, required_title="当前子任务", memory_title="近期行动摘要（当时的观察与计划，非完成证据）", recent_memory=True,
            images=images,
            require_image=bool(getattr(self.model, "vision", True)) and not bool(observation_info.get("dialog")))
        metadata.update(selected_actions=self._active_actions, plan_window=state.queue.window() if state.queue else {},
                        system_estimate=self.compiler.count(system), context_mode="current_step_brief_v2")
        state.prompt_metadata = metadata
        state.prompt_text = user_text
        if state.obs:
            state.obs.info["exposed_refs"] = metadata["exposed_refs"]
        return [
            {"role": "system", "content": system},
            self.model.build_vision_message(user_text, images),
        ]

    def decide(self, messages: List[dict], state: Optional[StepState] = None) -> Decision:
        raw, action = "", None
        calls: List[dict] = []
        base_messages = copy.deepcopy(messages)
        messages = copy.deepcopy(base_messages)
        dom_repair_used = False
        attempts = max(1, self.max_retries)
        for attempt in range(attempts):
            if self._task_time_remaining() <= 0:
                break
            try:
                raw = self.model.chat(messages, json_mode=True)
            except Exception as exc:  # noqa: BLE001  传输失败：记录本次调用，不进入重试
                call = copy.deepcopy(getattr(self.model, "last_call", {}))
                call.update({"messages": copy.deepcopy(messages),
                             "error_type": type(exc).__name__, "valid_action": False})
                calls.append(call)
                self._record_call(call)
                return Decision(raw="", action=None, calls=calls)
            call = copy.deepcopy(getattr(self.model, "last_call", {}))
            call.update({"messages": copy.deepcopy(messages), "response": raw, "valid_action": False,
                         "format_attempt": attempt + 1})
            calls.append(call)
            obj = parse_decision(raw)
            action = action_from_decision(obj or {})
            if action is not None:
                err = self._validate_action(action)
                error_type = 'invalid_action'
                if err is None and state is not None and not dom_repair_used and attempt + 1 < attempts:
                    err = self._dom_reference_feedback(action, state)
                    if err:
                        dom_repair_used = True
                        error_type = 'missing_dom_reference'
                patch = (obj or {}).get("state_update") or {}
                # Legacy auxiliary memory is optional. An invalid patch must
                # not prevent an otherwise valid primitive from executing.
                if not isinstance(patch, dict) or not isinstance(patch.get('known_fact_add', {}), dict):
                    call['ignored_fields'] = ['state_update']
                    patch = {}
                if err is None:
                    call["valid_action"] = True
                    def field(name):
                        value = obj.get(name)
                        return value if isinstance(value, str) else ''
                    summary = field('summary') or field('screen_analysis')
                    expected_effect = field('expected_effect') or summary
                    self._record_call(call)
                    return Decision(raw=raw, action=action, summary=summary,
                                    screen_analysis=field('screen_analysis') or summary,
                                    observed_effect=field('observed_effect'), action_status=field('action_status'),
                                    state_update=patch, expected_effect=expected_effect,
                                    calls=calls)
            else:
                if obj is None:
                    err, error_type = '输出不是合法 JSON 对象', 'invalid_json'
                elif 'next_action' not in obj:
                    err, error_type = '缺少 next_action 对象，需要 type 和本轮列出的必填参数', 'missing_action'
                else:
                    err, error_type = 'next_action 必须为包含 type 的对象', 'invalid_action'
            call.update(validation_error=err, validation_error_type=error_type)
            self._record_call(call)
            retry = attempt + 1 < attempts
            self._publish_diagnostic('decision_format_error', status='retry' if retry else 'exhausted',
                                     reason=err, error_type=error_type, attempt=attempt + 1, dispatched=False)
            if retry:
                # One bounded repair pair; do not append every malformed answer
                # to the next attempt or carry it into the next decision step.
                feedback = build_format_feedback(self.compiler.clip(err, 300))
                budget = max(0, 1280 - self.compiler.count(feedback) - 128)
                messages = [*copy.deepcopy(base_messages),
                            {'role': 'assistant', 'content': self.compiler.clip(raw, budget)},
                            {'role': 'user', 'content': feedback}]
        return Decision(raw=raw, action=None, calls=calls)

    def _dom_reference_feedback(self, action, state):
        """One pre-dispatch repair; unavailable AX and explicit desktop remain visual."""
        if action.action not in {'click', 'type', 'select', 'double_click', 'right_click'}:
            return None
        if action.args.get('target_ref') or action.args.get('element_id'):
            return None
        # A PDF document may expose only its viewer/container or a sparse
        # partial AX tree. Never demand a made-up DOM reference on that surface.
        # Explicit refs and semantic DOM routing still work in ground().
        if is_pdf_observation(state.obs):
            return None
        from .browser_session import INTERACTIVE
        nodes = [n for n in exposed_nodes(state.obs)
                 if n.get('direct') and n.get('role') in INTERACTIVE
                 and not n.get('states', {}).get('disabled')]
        if action.action == 'type':
            nodes = [n for n in nodes if n.get('role', '').lower() in {'textbox', 'searchbox', 'combobox'}]
        if not nodes:
            return None
        try:
            route = self.env.route_action(copy.deepcopy(action), state.obs)
        except Exception:
            return None
        if route and route.get('channel') == 'browser_dom':
            return None
        return ('当前 AX 有可操作元素，但此动作没有可确定的 DOM 引用。请从本轮 AX 中选择正确的 '
                'target_ref/element_id，结合论文标题和链接核对目标，不能猜编号；若目标确实是浏览器工具栏或系统窗口，'
                '先 observe_desktop。网页画布等无对应 AX 元素时可保留视觉 target。此候选动作尚未执行。')

    def ground(self, action: Action, state: StepState) -> Action:
        action = copy.deepcopy(action)
        if state.obs and state.obs.info.get("dialog") and action.action not in {
                "handle_dialog", "observe_desktop", "observe_browser", "fail", "list_tabs"}:
            state.route = {"channel": "reobserve", "error_code": "blocked",
                           "reason": "原生对话框仍打开，请先使用 handle_dialog 明确接受或取消"}
            state.resolved_action = action
            return action
        state.last_grounding_target = str(action.args.get("target", "")
                                          or action.args.get("from_target", ""))
        # 先路由：带 target_ref 且引用有效 → browser_dom；否则走 GUI grounding
        try:
            state.route = self.env.route_action(action, state.obs)
        except Exception:  # noqa: BLE001
            state.route = None
        error = paper_action_error(action, state.obs,
                                   active_paper(self.completion, state.queue.current if state.queue else None), state.route)
        if error:
            state.route = {"channel": "reobserve", "error_code": "paper_identity_mismatch", "reason": error}
        if state.route is not None:
            state.last_grounding_coord = None
            state.resolved_action = action
            return action
        resolved = self._resolve(action, state.obs)
        self._attach_observation_space(resolved, state.obs)
        state.last_grounding_coord = self._coordinate_of(resolved)
        state.resolved_action = resolved
        if get_action_spec(action.action) and get_action_spec(action.action).resolve != "none":
            self._emit("grounding", actor_role="grounder", source="frozen_model",
                       observation_id=f"o_{state.step_id}", target=state.last_grounding_target,
                       resolved_action=self._compact_action(resolved), policy_trainable=False)
        return resolved

    @staticmethod
    def _attach_observation_space(action, obs):
        if obs is not None and any(k in action.args for k in ("x", "x1")):
            info = obs.info or {}
            action.args["coordinate_space"] = info.get("screenshot_space", "screen_pixels")
            action.args["observation_metadata"] = {
                k: copy.deepcopy(info[k]) for k in (
                    "screenshot_space", "screenshot_size", "page_id", "document_id", "snapshot_id",
                    "captured_at", "url", "scroll_offset")
                if k in info}

    def _observe(self):
        """Refresh without resetting the task; failure must not reuse stale evidence."""
        try:
            obs = self.env.observe()
            if obs is not None:
                return obs
        except Exception as exc:
            return Observation(info={"observation_error": type(exc).__name__})
        return Observation(info={"observation_error": "unavailable"})

    def _recover_browser(self, action, step_result, state):
        """One bounded visual recovery, only before input was dispatched.

        Even rejected DOM actions may have scrolled the page. Ground exclusively
        against a new observation. Stale/blocked/ambiguous actions are replanned.
        """
        info = step_result.info or {}
        self._emit("execution_attempt", action=self._compact_action(action), execution_info=info,
                   observation=self._observation_record(step_result.observation, "after_dom"), policy_trainable=False)
        if (info.get("status") != "rejected" or info.get("dispatched") is not False
                or info.get("error_code") not in {"not_found", "unsupported"}
                or action.action not in {"click", "double_click", "right_click", "type", "select", "drag_and_drop"}):
            return step_result
        fresh = self._observe()
        self._emit("observation", observation_id="recovery", observation=self._observation_record(fresh, "recovery"),
                   policy_trainable=False)
        old_info = state.obs.info if state.obs is not None else {}
        if (fresh.screenshot is None or fresh.info.get("screenshot_space") != "css_viewport"
                or not old_info.get("page_id")
                or fresh.info.get("page_id") != old_info.get("page_id")):
            step_result.observation = fresh
            return step_result
        visual = self._resolve(action, fresh)
        self._attach_observation_space(visual, fresh)
        if visual.args.get("grounding_failed"):
            step_result.observation = fresh
            return step_result
        self._emit("grounding", actor_role="grounder", source="frozen_model", observation_id="recovery",
                   resolved_action=self._compact_action(visual), policy_trainable=False)
        recovered = self.env.step(visual)
        self._emit("execution_attempt", action=self._compact_action(visual), execution_info=recovered.info,
                   policy_trainable=False)
        recovered.info = dict(recovered.info or {})
        recovered.info["recovery"] = {"from": copy.deepcopy(info),
                                      "action": self._compact_action(visual),
                                      "observation": self._observation_record(fresh, "recovery")}
        state.resolved_action = visual
        state.last_grounding_coord = self._coordinate_of(visual)
        return recovered

    def execute(self, action: Action, state: StepState) -> Execution:
        name = action.action
        if name not in BASIC_ACTIONS:
            return Execution(kind="rejected", executed="rejected", obs=state.obs,
                info={"status": "rejected", "dispatched": False, "error_code": "non_basic_action"})
        # browser_dom 通道：路由成功后走 DOM 定位执行（跳过 grounding）
        if state.route and state.route.get("channel") == "browser_dom":
            try:
                step_result = self.env.execute_routed(action, state.route)
                step_result = self._recover_browser(action, step_result, state)
                info = step_result.info if isinstance(step_result.info, dict) else {}
                return Execution(kind="browser_dom", executed=self._compact_action(action),
                                 outcome=json.dumps(info, ensure_ascii=False),
                                 obs=step_result.observation, info=info,
                                 reward=step_result.reward, done=step_result.done)
            except Exception as exc:  # noqa: BLE001
                return Execution(kind="browser_dom", executed=self._compact_action(action),
                                 outcome=f"结构化执行异常，需重新观察: {type(exc).__name__}",
                                 obs=self._observe(),
                                 info={"status": "uncertain", "error_type": type(exc).__name__})
        if state.route and state.route.get("channel") == "reobserve":
            return Execution(kind="browser_dom", outcome=state.route.get("reason", "需重新观察"),
                             executed="rejected", obs=self._observe(),
                             info={"status": "rejected", "dispatched": False,
                                   "reason": state.route.get("reason", "需重新观察"),
                                   "error": state.route.get("reason", "需重新观察"),
                                   "error_code": state.route.get("error_code", "stale")})

        spec = get_action_spec(name)
        kind = spec.kind if spec else "env"
        if kind not in {"env", "browser", "terminal"}:
            return Execution(kind="rejected", info={"status": "rejected", "dispatched": False,
                             "error_code": "non_basic_executor"}, obs=state.obs)

        if kind == "browser":
            if name == "get_page_source":
                try:
                    result = self.env.browser_action(name, action.args)
                    state.page_source = result[:12000] + ("\n...(源码过长已截断)" if len(result) > 12000 else "")
                    return Execution(kind="browser", executed="浏览器: get_page_source", obs=self._observe(),
                                     info={"status": "executed", "action": name}, outcome="页面源码将在下一步返回")
                except Exception as exc:
                    return Execution(kind="browser", obs=self._observe(), info={"status": "failed", "error_code": "source_unavailable"},
                                     outcome=f"页面源码读取失败：{type(exc).__name__}")
            step_result = self.env.step(action)
            result = json.dumps(step_result.info or {}, ensure_ascii=False)
            state.browser_result = result
            return Execution(kind="browser", executed=f"浏览器: {name}",
                             outcome=result, info=step_result.info or {}, obs=step_result.observation,
                             done=step_result.done, reward=step_result.reward)

        if kind == "terminal":
            if name == "fail":
                return Execution(kind="terminal", executed=name, outcome="任务失败",
                                 done=True, success=False, answer="")
            # request_finish：不立即结束，交给 postprocess 验证
            return Execution(kind="finish_request", executed=name, outcome="请求结束，待验证",
                             obs=self._observe(), done=False, success=False,
                             answer=str(action.args.get("answer", "")))

        # Basic GUI action, after coordinate resolution.
        try:
            step_result = self.env.step(action)
        except Exception as exc:  # noqa: BLE001
            return Execution(kind=kind, executed=self._compact_action(action),
                             outcome=f"执行异常: {exc}", obs=self._observe(),
                             info={"status": "uncertain", "error_type": type(exc).__name__})
        outcome = self._describe_outcome(step_result)
        # type 落点校验结果透传：把「疑似打错输入框」的提示拼进 outcome，模型下一轮能看到并纠正
        if isinstance(step_result.info, dict):
            tc = step_result.info.get("type_check")
            if step_result.info.get("type_wrong_target") and tc:
                outcome += f"；⚠️ 输入落点校验：{tc}"

        return Execution(kind=kind, executed=self._compact_action(action),
                         outcome=outcome,
                         obs=step_result.observation, done=step_result.done,
                         info=step_result.info or {}, reward=step_result.reward)

    def _task_time_remaining(self) -> float:
        deadline = getattr(self, "_task_deadline", None)
        return max(0.0, deadline - time.monotonic()) if deadline is not None else float("inf")

    def _advance_paper_workflow(self, action, execution, state):
        if execution.kind in {"terminal", "finish_request", "deadline"} or self._task_time_remaining() <= 0:
            return execution
        workflow = getattr(self, "_paper_workflow", None)
        if workflow is None or not self.completion.paper_goals():
            return execution

        def workflow_event(event, **payload):
            self._emit(event, **payload)
            if event == "framework_operation_started":
                self._wait_status("waiting", "框架正在下载已确认的论文", 0)
            elif event == "framework_operation_finished":
                status = payload.get("result", {}).get("status")
                self._wait_status("completed" if status in {"executed", "already_downloaded"} else "failed",
                                  "论文传输结束，正在验收" if status in {"executed", "already_downloaded"} else "论文传输需要检查",
                                  payload.get("elapsed_s", 0))
        # Framework transport/renaming never turns a rejected sampled click into
        # a positive policy action. Uncertain dispatch must be resolved first.
        for _ in range(2):
            self.completion.observe(execution.obs or state.obs, execution.info)
            if self.completion.evaluate().get("status") == "passed":
                break
            info = execution.info
            self._set_download_deadline()
            # Read-only identity binding; all downloading and renaming stays in GUI/DOM.
            update = workflow.advance(execution.obs or state.obs, self.completion, self.env, workflow_event,
                remaining_s=self._task_time_remaining(), allow_download=False)
            for message in update.messages:
                state.memory.add_event(message)
            state.task_profile["goals"] = self.completion.paper_goals()
            for goal in state.task_profile["goals"]:
                binding = goal.get("binding") or {}
                if binding:
                    state.memory.known_facts["paper:" + goal["check_id"]] = {
                        "title": binding.get("full_title"), "id": binding.get("canonical_id"),
                        "pdf_url": binding.get("pdf_url"), "filename": goal.get("filename")}

            execution.obs = update.observation
            if not update.operations:
                break
            execution.info.setdefault("framework_operations", []).extend(update.operations)
            metadata = self._browser_event_metadata() or {}
            for key in ("artifacts", "artifact_events", "download_events", "pending_downloads"):
                if key in metadata:
                    execution.info[key] = copy.deepcopy(metadata[key])
            raced = any(op.get("result", {}).get("status") == "in_progress" for op in update.operations)
            if not raced:
                break
            execution = self._settle_execution(action, execution, state)
            execution = self._settle_rendering(execution, state)
        return execution

    def _set_download_deadline(self) -> None:
        # Do not create a browser session for desktop-only tasks.
        session = getattr(self.env, "_browser_session", None)
        options = getattr(session, "options", None)
        if options is not None and hasattr(options, "download_timeout_s"):
            options.download_timeout_s = max(0.001, min(self.wait_config.operation_timeout_s,
                                                       self._task_time_remaining()))

    def _browser_event_metadata(self):
        reader = getattr(self.env, "browser_event_metadata", None)
        if reader is None:
            reader = getattr(getattr(self.env, "_browser_session", None), "event_metadata", None)
        if reader is None:
            return None
        try:
            value = reader()
            return value if isinstance(value, dict) else None
        except Exception as exc:
            self._emit("framework_wait", status="failed", reason="browser_events_unavailable",
                       error_type=type(exc).__name__, policy_trainable=False)
            return None

    @staticmethod
    def _download_id(item):
        return str(item.get("download_id") or item.get("artifact_id") or item.get("id") or "")

    @staticmethod
    def _observation_page(obs):
        if obs is None:
            return {}
        block = next((b for b in obs.context if b.get("kind") == "browser_ax"), {})
        return {key: obs.info.get(key, block.get(key)) for key in ("page_id", "document_id", "url")}

    def _settle_rendering(self, execution: Execution, state: StepState) -> Execution:
        """Wait for nonblank stable PDF content without spending policy steps.

        This is a bounded readiness heuristic, never completion evidence. A timed
        out document is offered to the policy once instead of waiting on every turn.
        """
        obs = execution.obs
        cfg = self.wait_config
        if not cfg.enabled or not is_pdf_observation(obs) or obs.info.get('dialog'):
            return execution
        key = tuple(self._observation_page(obs).get(k) for k in ('page_id', 'document_id', 'url'))
        previous = content_thumbnail(obs.screenshot)
        cached = state.render_waits.get(key)
        if (cached == 'timeout' and previous is None) or (cached == 'completed' and previous is not None):
            obs = copy.deepcopy(obs)
            obs.info['render_wait'] = {'status': cached, 'reason': 'pdf_render'}
            execution.obs = obs
            return execution
        started = time.monotonic()
        deadline = started + min(cfg.render_timeout_s, self._task_time_remaining())
        self._publish_diagnostic('framework_wait', status='waiting', reason='pdf_render', elapsed_s=0)
        status = 'timeout'
        last_update = started
        while time.monotonic() < deadline:
            time.sleep(min(cfg.poll_interval_s, max(0, deadline - time.monotonic())))
            fresh = self._observe()
            if fresh.info.get('observation_error'):
                previous = None
                continue
            obs = fresh
            fresh_key = tuple(self._observation_page(obs).get(k) for k in ('page_id', 'document_id', 'url'))
            if obs.info.get('dialog') or not is_pdf_observation(obs) or fresh_key != key:
                status = 'interrupted'
                break
            current = content_thumbnail(obs.screenshot)
            if stable_content(previous, current):
                status = 'completed'
                break
            previous = current
            if time.monotonic() - last_update >= 1:
                last_update = time.monotonic()
                self._publish_diagnostic('framework_wait', status='waiting', reason='pdf_render',
                                         elapsed_s=round(last_update - started, 3))
        if status != 'interrupted':
            state.render_waits[key] = status
            # Keep a bounded cache even for long browsing sessions.
            if len(state.render_waits) > 32:
                state.render_waits.pop(next(iter(state.render_waits)))
        obs = copy.deepcopy(obs)
        result = {'status': status, 'reason': 'pdf_render', 'elapsed_s': round(time.monotonic() - started, 3)}
        obs.info['render_wait'] = result
        execution.obs = obs
        execution.info['render_wait'] = result
        self._publish_diagnostic('framework_wait', **result)
        return execution

    def _settle_execution(self, action: Action, execution: Execution, state: StepState,
                          *, _deadline=None, _refresh_depth=0) -> Execution:
        """Observe relevant asynchronous downloads without resampling or replaying an action.

        Only event metadata is polled. A single fresh observation is collected after
        an actual wait, including on the last allowed decision. Missing signals get
        a bounded startup grace; a native dialog immediately yields to the policy.
        """
        cfg = self.wait_config
        if not cfg.enabled or execution.kind not in {"browser_dom", "env", "browser", "finish_request"}:
            return execution
        metadata = self._browser_event_metadata()
        if metadata is None:
            return execution
        info = execution.info
        original_metadata = copy.deepcopy(state.obs.info if state.obs else {})
        page = self._observation_page(state.obs)
        if page.get("page_id") or page.get("url"):
            state.last_browser_page = dict(page)
        else:
            page = dict(state.last_browser_page)
        target_url = str(action.args.get("url") or "").split("#", 1)[0]
        from urllib.parse import urljoin
        nodes = [n for b in (state.obs.context if state.obs else []) if b.get("kind") == "browser_ax"
                 for n in b.get("nodes", [])]
        target_node = next((n for n in nodes if action.args.get("target_ref")
                            and n.get("ref") == action.args["target_ref"]), {})
        linked_url = urljoin(page.get("url") or "", str(target_node.get("href") or target_node.get("url") or "")) if target_node else ""
        related_urls = {u.split("#", 1)[0] for u in (target_url, linked_url) if u}
        candidates = list(original_metadata.get("download_candidates", []))
        for block in state.obs.context if state.obs else []:
            candidates.extend(block.get("download_candidates", []))
        candidate_urls = {str(c.get("url", "")).split("#", 1)[0] for c in candidates if isinstance(c, dict)}
        prior_ids = {self._download_id(item) for key in ("artifacts", "download_events")
                     for item in original_metadata.get(key, []) if isinstance(item, dict)}
        tracked = set(state.waiting_download_ids)
        # Adapter artifact lists are cumulative across tabs, not causal evidence.
        # Only the executor's top-level operation ID identifies its own transfer.
        explicit_ids = {self._download_id(info)} - {""}
        started = time.monotonic()
        grace_end = started + cfg.download_start_timeout_s
        operation_end = min(started + cfg.operation_timeout_s,
                            _deadline if _deadline is not None else float("inf"))
        last_ui = -float("inf")
        announced = False
        waited = False
        outcome = "no_signal"
        failure = None

        def merge(events):
            for key in ("artifacts", "download_events", "pending_downloads", "artifact_events", "dialogs", "dialog"):
                if key in events:
                    info[key] = copy.deepcopy(events[key])
                    if execution.obs is not None:
                        # Keep the decision's before-observation immutable even if an adapter reuses it.
                        execution.obs.info[key] = copy.deepcopy(events[key])
            if events.get("dialogs") and "dialog" not in events:
                info["dialog"] = copy.deepcopy(events["dialogs"][0])
                if execution.obs is not None:
                    execution.obs.info["dialog"] = copy.deepcopy(events["dialogs"][0])
            elif "dialogs" in events and not events["dialogs"] and not events.get("dialog"):
                info.pop("dialog", None)
                if execution.obs is not None:
                    execution.obs.info.pop("dialog", None)

        def related(item):
            ident = self._download_id(item)
            if ident in tracked or ident in explicit_ids:
                return True
            urls = {str(item.get(key) or "").split("#", 1)[0] for key in ("url", "final_url", "source_url")}
            if related_urls.intersection(urls):
                return True
            if item.get("page_id") and page.get("page_id"):
                return item["page_id"] == page["page_id"]
            return bool(item.get("page_url") and page.get("url") and item["page_url"] == page["url"])

        def pending_items(events):
            source = events.get("pending_downloads")
            if source is None:
                source = [d for d in events.get("download_events", []) if d.get("status") == "in_progress"]
            selected = []
            for item in source:
                if not isinstance(item, dict):
                    continue
                ident = self._download_id(item)
                if ident and related(item):
                    selected.append(item)
                    tracked.add(ident)
            return selected

        # Copy container metadata only; PIL images need not be duplicated.
        if execution.obs is not None:
            execution.obs = copy.copy(execution.obs)
            execution.obs.info = copy.deepcopy(execution.obs.info)
        # Executor evidence covers the small interval before the first event poll.
        seed = {**info, **metadata}
        pending_items(seed)
        candidate = (action.action in {"click", "double_click", "goto", "download", "press", "hotkey"}
                     and info.get("dispatched") is not False
                     and info.get("status") not in {"rejected", "failed", "error"}
                     and (execution.kind == "browser_dom" or page.get("page_id")))
        # An event reader returning {} has no active browser; it is not a wait signal.
        candidate = candidate and bool(metadata or tracked)
        download_intent = (action.action == "download" or bool(target_node.get("download"))
                           or bool(related_urls.intersection(candidate_urls))
                           or bool(re.search(r"pdf|download|export|下载|导出", str(action.args), re.I)))
        candidate = candidate and download_intent
        while True:
            merge(metadata)
            pending = pending_items(metadata)
            completed = {self._download_id(item) for item in metadata.get("artifacts", [])
                         if isinstance(item, dict) and item.get("completed") is True and related(item)}
            failed = [item for item in metadata.get("download_events", []) if isinstance(item, dict)
                      and item.get("status") in {"failed", "canceled", "cancelled", "timeout"}
                      and related(item) and (self._download_id(item) in tracked or self._download_id(item) not in prior_ids)]
            if metadata.get("dialog") or metadata.get("dialogs"):
                outcome = "dialog"
                state.recovery = {"status": "resolve_dialog", "instruction": "先处理浏览器原生对话框，再检查下载状态；不要重复触发下载。"}
                break
            if failed and not pending:
                outcome, failure = "failed", "download_failed"
                break
            if (tracked and tracked.issubset(completed)) or (completed - prior_ids and not pending):
                outcome = "completed"
                break
            if not pending and not tracked and (not candidate or time.monotonic() >= grace_end):
                break
            if tracked and not pending:
                # Disappearance without a terminal event is not proof of completion.
                outcome, failure = "uncertain", "download_state_unavailable"
                break
            now = time.monotonic()
            deadline = min(operation_end, now + self._task_time_remaining())
            for item in pending:
                if isinstance(item.get("deadline_at"), (float, int)):
                    deadline = min(deadline, now + max(0.0, item["deadline_at"] - time.time()))
            if now >= deadline:
                outcome = "task_timeout" if self._task_time_remaining() <= 0 else "timeout"
                failure = "download_timeout" if pending else None
                cancel = getattr(self.env, "cancel_download", None) or getattr(
                    getattr(self.env, "_browser_session", None), "cancel_download", None)
                for item in pending:
                    if cancel is not None:
                        try:
                            canceled = cancel(self._download_id(item))
                            self._emit("framework_wait_cancel", download_id=self._download_id(item),
                                       result=canceled, policy_trainable=False)
                        except Exception as exc:
                            self._emit("framework_wait_cancel", download_id=self._download_id(item),
                                       error_type=type(exc).__name__, policy_trainable=False)
                final_metadata = self._browser_event_metadata()
                if final_metadata is not None:
                    metadata = final_metadata
                    merge(metadata)
                break
            if not announced:
                self._emit("framework_wait", status="waiting", reason="pending_download" if pending else "download_start_grace",
                           download_ids=sorted(tracked), elapsed_s=0, policy_trainable=False)
                announced = True
            if now - last_ui >= 1:
                self._wait_status("waiting", "等待下载完成" if pending else "等待浏览器下载事件", now - started)
                last_ui = now
            sleep_for = min(cfg.poll_interval_s, max(0.0, deadline - now))
            if not pending:
                sleep_for = min(sleep_for, max(0.0, grace_end - now))
            time.sleep(sleep_for)
            waited = True
            metadata = self._browser_event_metadata()
            if metadata is None:
                outcome, failure = "uncertain", "browser_events_unavailable"
                break
        if waited or outcome == "dialog":
            execution.obs = self._observe()
            if execution.obs is not None:
                execution.obs = copy.copy(execution.obs)
                execution.obs.info = copy.deepcopy(execution.obs.info)
            # Observation may dispatch late download/dialog events. Its newer
            # arrays must win over metadata read before taking the screenshot.
            fresh = execution.obs.info if execution.obs is not None else {}
            metadata = {**(metadata or {}), **{key: fresh[key] for key in
                ("artifacts", "download_events", "pending_downloads", "artifact_events", "dialogs", "dialog")
                if key in fresh}}
            if "dialogs" in fresh and not fresh.get("dialog"):
                metadata.pop("dialog", None)
            merge(metadata)
        terminal_ids = {self._download_id(item) for key in ("artifacts", "download_events")
                        for item in (metadata or {}).get(key, []) if isinstance(item, dict) and
                        (item.get("completed") is True or item.get("status") in {"failed", "canceled", "cancelled", "timeout"})}
        late_pending = pending_items(metadata or {})
        state.waiting_download_ids = tracked - terminal_ids
        if (metadata or {}).get("dialog") or (metadata or {}).get("dialogs"):
            outcome = "dialog"
        elif late_pending and not failure and _refresh_depth < 4 and time.monotonic() < operation_end:
            # Preserve the original deadline across late-event refreshes.
            return self._settle_execution(action, execution, state, _deadline=operation_end,
                                          _refresh_depth=_refresh_depth + 1)
        if failure:
            info["wait_error_code"] = failure
            info["wait_reason"] = f"{failure}；先检查实际下载状态，不要自动重发原动作"
            # Preserve dispatch status; waiting does not make a failed policy action succeed.
            if info.get("status") not in {"rejected", "failed", "error"}:
                info["status"] = "uncertain"
                info["reason"] = info["wait_reason"]
        if announced or failure or outcome == "dialog":
            elapsed = time.monotonic() - started
            info["framework_wait"] = {"status": outcome, "elapsed_s": round(elapsed, 3), "download_ids": sorted(tracked)}
            self._emit("framework_wait", **info["framework_wait"], policy_trainable=False)
            self._wait_status("failed" if failure else "completed",
                              "浏览器对话框需要处理" if outcome == "dialog" else "下载等待失败" if failure else "浏览器等待结束", elapsed)
        return execution

    @staticmethod
    def _wait_status(status, reason, elapsed):
        if _viz_log_status is not None:
            try:
                _viz_log_status({"event": "framework_wait", "status": status,
                                 "reason": reason, "elapsed_s": round(elapsed, 1)})
            except Exception:
                pass

    def postprocess(self, step_no: int, decision: Decision, execution: Execution,
                    state: StepState, result: AgentResult) -> Tuple[StepState, bool]:
        """⑥ 后处理：记录 expected/observed + 控制状态 + 卡死检测 + Finish 验证。"""
        # 1) 记录模型对上一步的判断（observed / action_status）
        state.memory.observed_effect = decision.observed_effect
        state.memory.action_status = decision.action_status
        # 2) 根据 action_status 记录事件/失败（针对上一步动作）
        if state.last_action:
            desc = self._action_desc(state.last_action, state.last_target)
            if decision.action_status == "failure":
                state.memory.add_failure(f"{desc} 失败：{decision.observed_effect or '无可见变化'}")
                state.memory.record_failed_strategy(desc)
                state.memory.add_event(f"{desc} → 失败")
            else:
                state.memory.add_event(f"{desc} → {decision.observed_effect or decision.action_status or '已执行'}")
        # 3) 应用 state_update 补丁
        state.memory.apply_state_update(decision.state_update)
        # 4) 回填上一步动作 a_{t-1} 的 assessment（decision 的 observed_effect/action_status
        #    评价的是上一步动作，而不是本步 next_action）
        self._fill_prev_assessment(result, decision)
        # 4.1) 记录这一步的 transition：a_t 的 assessment 留空，下一步回填
        result.trajectory.append({
            "step": step_no,
            "action_id": f"a_{step_no}",
            "observation_id": f"o_{step_no}",
            "screen_analysis": decision.screen_analysis,
            "summary": decision.summary,
            "state_update": decision.state_update,
            "page_info": state.obs.text if state.obs else "",  # 决策时的页面信息（可交互元素+selector 或空）
            "action": self._compact_action(decision.action),
            "grounding": {
                "target": state.last_grounding_target,
                "coord": list(state.last_grounding_coord) if state.last_grounding_coord else None,
            },
            "routing": state.route,
            "resolved_action": self._compact_action(state.resolved_action) if state.resolved_action is not None else None,
            "prompt_metadata": state.prompt_metadata,
            "prompt_text": state.prompt_text,
            "observation": {
                "context_kinds": [b.get("kind") for b in (state.obs.context if state.obs else [])],
                "exposed_refs_count": len((state.obs.info or {}).get("exposed_refs") or []) if state.obs else 0,
            },
            "expected_effect": decision.expected_effect,
            "executed": execution.executed,
            "outcome": execution.outcome,
            "reward": execution.reward,
            "done": False,
            "assessment": None,
        })
        # 5) 更新程序信息（当前动作 → 供下一轮）
        self._update_program_info(decision.action, execution, state)
        # 5.5) 记录动作历史（程序侧，让模型能回顾从头到尾做过了什么、结果如何）
        # 用精简 outcome（不存完整 JSON），否则动作历史快速膨胀、挤掉「最近事件/已知事实」等关键记忆
        if decision.action:
            a = decision.action
            adesc = self._action_desc(a.action, self._target_of(a))
            aoutcome = self._brief_outcome(execution)
            state.memory.add_action(step_no, adesc, aoutcome)
            state.memory.add_decision_note(step_no, decision.summary or decision.screen_analysis, adesc,
                                           str(execution.info.get('status') or ''))
        # 6) env 动作更新观察
        before_observation = state.obs
        before_screenshot = state.screenshot_history[-1] if state.screenshot_history else None
        if execution.obs is not None:
            state.obs = execution.obs
        self.completion.observe(state.obs, execution.info)
        state.verification = self.completion.evaluate()
        if execution.info.get("error_code") == "blocked":
            state.recovery = {"status": "resolve_obstruction", "original_action": self._compact_action(decision.action),
                              "blocker": execution.info.get("blocker"),
                              "blocker_ref": execution.info.get("blocker_ref"),
                              "active_modals": execution.info.get("active_modals", []),
                              "instruction": "先根据当前截图和AX处理障碍，再重新定位原目标；不要直接重复原点击。"}
        elif state.recovery and execution.info.get("status") == "executed":
            state.recovery["status"] = "reobserve_original_target"
        self._update_plan_progress(state, execution, before_observation=before_observation,
                                   expected_effect=decision.expected_effect)
        result.trajectory[-1]["step_verification"] = copy.deepcopy(state.step_review)
        result.trajectory[-1]["execution_queue"] = state.queue.snapshot() if state.queue else {}
        # 6.5) 单步动作有效性校验：env 动作执行后屏幕无变化 → 当场提示换策略
        # （不等到连续 2 步 STALL，每一步点空/点错都立即提示，避免反复点同一无效目标）
        if execution.kind == "env" and execution.obs is not None and before_screenshot is not None:
            if self._screen_diff(before_screenshot, execution.obs.screenshot) < _NO_CHANGE_DIFF_THRESHOLD:
                state.memory.add_event(
                    "⚠️ 这一步动作后屏幕没有可见变化：可能点空 / 点错 / 焦点不对。"
                    "下一步请换一种方式（键盘快捷键 / 不同菜单路径 / 重新定位目标），不要重复同样的点击。")
        # 7) 控制状态 + 确定性卡死检测
        self._compute_control_state(state, decision, execution, before_screenshot, before_observation)
        result.trajectory[-1]["step_progress"] = copy.deepcopy(state.step_progress)
        if state.step_review.get("status") == "passed":
            self._clear_stall(state)
            state.memory.control_state = "ADVANCE"
        should_break = False
        # 8) 设置下一步的 expected_effect（当前决策的 expected_effect → 下一步对比用）
        state.memory.expected_effect = decision.expected_effect
        # All accepted task conditions must be backed by framework evidence.
        if state.verification.get("status") == "passed" and execution.kind != "terminal":
            result.success = True
            result.final_answer = execution.answer or "框架已逐项验证所有任务完成条件。"
            result.termination_reason = "verified"
            should_break = True
        if not should_break and (execution.kind == "finish_request" or
                                 (state.queue is not None and state.queue.current is None and execution.kind != "terminal")):
            confirmed, reason = self._verify_finish(state, execution.answer)
            if confirmed:
                result.success = True
                result.termination_reason = "verified"
                result.final_answer = execution.answer or "独立核查确认原始任务的全部要求已满足。"
                should_break = True
            elif getattr(self.env, "evaluation_available", False):
                # OSWorld evaluation may run postconfig; invoke it once at episode end.
                result.termination_reason = "evaluation_requested"
                result.final_answer = execution.answer
                should_break = True
            elif not self.completion.spec.get("checks") and self.verifier_model is None:
                result.termination_reason = "completion_unverifiable"
                result.final_answer = "执行结束，但系统尚未验证任务完成。模型答复：" + execution.answer
                should_break = True
            else:
                state.finish_feedback = reason
                state.memory.add_event(f"request_finish 未通过系统验证：{reason}")
        if execution.done:
            if execution.kind == "terminal":
                result.success = False
                result.termination_reason = "agent_failed"
            elif not result.success:
                result.termination_reason = "environment_ended"
            result.final_answer = execution.answer or result.final_answer
            should_break = True
        # Verification has priority over recovery. Reserve the final decision's
        # aftermath for settlement/verification; a plan with no action slot is useless.
        if not should_break and self._task_time_remaining() <= 0:
            result.termination_reason = "task_timeout"
            should_break = True
        if not should_break and state.memory.control_state == "STALL":
            should_break = self._recover_plan(state, result, step_no)
        # （reflection 已关闭：旧实现调了模型但结果被丢弃，属无效调用，浪费一次推理）

        return state, should_break

    # ================= 内部工具 ================= #
    def _recover_plan(self, state: StepState, result: AgentResult, step_no: int) -> bool:
        if step_no >= self.max_steps:
            return False
        self._publish_diagnostic("framework_recovery", status="requested", failure_class=state.failure_class,
                   failure_count=state.memory.stall_count, failure_key=state.failure_key,
                   step_progress=state.step_progress,
                   remaining_decisions=self.max_steps - step_no)
        repaired = False
        if self._replans < self.max_replans and self._task_time_remaining() > 0:
            self._replans += 1
            repaired = self._plan(state.task, state, repair=True)
        if repaired:
            result.plan = state.memory.plan
            self._clear_stall(state)
            state.memory.control_state = "RECOVER"
            state.memory.add_event("已根据实际失败证据重新规划；继续使用当前环境，不重置任务。")
        else:
            result.trajectory.append({"step": step_no, "early_stop": "持续失败且恢复预算耗尽"})
            result.termination_reason = "task_timeout" if self._task_time_remaining() <= 0 else "stalled"
        self._publish_diagnostic("framework_recovery", status="replanned" if repaired else "exhausted",
                   remaining_replans=max(0, self.max_replans - self._replans))
        return not repaired

    def _fill_prev_assessment(self, result: AgentResult, decision: Decision) -> None:
        """把 decision 的 observed_effect/action_status 回填到上一步动作的 transition。

        时序对齐：decision 里的 observed_effect/action_status 评价的是【上一步动作 a_{t-1}】，
        而不是本步的 next_action。所以把它们回填到最后一条「action 非空、assessment 未填」
        的 transition（即 a_{t-1} 那条），而不是记在本步 transition 里。
        """
        for t in reversed(result.trajectory):
            if t.get("action") and t.get("assessment") is None:
                t["assessment"] = {
                    "observed_effect": decision.observed_effect,
                    "action_status": decision.action_status,
                    "source": "model_annotation",
                    "verified": False,
                }
                break

    def _fill_final_assessment(self, result: AgentResult) -> None:
        """任务结束时补齐最后一步的 assessment。

        最后一步动作 a_N 执行后没有下一步决策来评估它（任务已结束），导致它的
        assessment 缺失——但最后一个关键动作（如 save）恰恰需要后验观察。用任务
        最终得分补上这条 assessment。
        """
        for t in reversed(result.trajectory):
            if t.get("action") and t.get("assessment") is None:
                # 使用框架最终验收；没有可用证据时保留 uncertain，避免生成虚假的失败标签。
                ok = bool(result.success) or result.score >= 1.0
                t["assessment"] = {
                    "observed_effect": ("任务结束，成功" if ok else
                        f"任务结束，最终得分 {result.score}" if result.evaluation_available else
                        "任务结束，缺少可用的最终验收证据"),
                    "action_status": "success" if ok else "failure" if result.evaluation_available else "uncertain",
                    "source": "episode_outcome", "scope": "episode",
                    "verified": result.evaluation_available,
                }
                t["done"] = True
                break

    def _init_state(self, task: str) -> StepState:
        return StepState(
            task=task,
            memory=StructuredMemory(),
            skill_ctx=SkillContext(env=self.env),
        )

    @staticmethod
    def _plan_actions(names):
        aliases = {"open_url": "goto", "navigate": "goto", "key_press": "press", "press_key": "press",
                   "input_text": "type", "type_text": "type"}
        if isinstance(names, list):
            names = [aliases.get(n, n) if isinstance(n, str) else n for n in names]
        if not isinstance(names, list) or any(not isinstance(n, str) or n not in BASIC_ACTIONS for n in names):
            raise ValueError("actions 只能包含基础 GUI/DOM 动作")
        return list(dict.fromkeys(names))

    def _plan(self, task: str, state: StepState, repair=False) -> bool:
        """Mandatory task decomposition, with one bounded repair attempt.

        A plan proposes actions and evidence; it cannot create or weaken the
        trusted completion contract. Only existing check IDs may be attached.
        """
        prompt = build_planner_prompt(task, self.completion.public_spec, state.task_profile)
        if repair:
            prompt += "\n当前计划受阻，请保留原目标，按以下已发生事实修订剩余步骤：\n" + json.dumps({
                "previous_plan": state.memory.plan, "recovery": state.recovery,
                "step_review": state.step_review, "step_progress": state.step_progress,
                "phase_events": state.phase_events, "feedback": self.completion.public_report(state.verification),
                "recent_actions": state.memory.action_history[-5:]}, ensure_ascii=False)
        messages = [{"role": "user", "content": prompt}]
        if not repair:
            self.planning_calls = []
        check_ids = {c["id"] for c in self.completion.public_spec.get("checks", [])}
        for attempt in range(2):
            if self._task_time_remaining() <= 0:
                return False
            raw = ""
            call = None
            try:
                raw = self.planner_model.chat(messages, retries=1, json_mode=True)
                call = copy.deepcopy(getattr(self.planner_model, "last_call", {}))
                call.update(messages=copy.deepcopy(messages), response=raw, valid_plan=False,
                            plan_stage="replan" if repair else "initial")
                self.planning_calls.append(call)
                obj = parse_decision(raw)
                steps = obj.get("steps") if isinstance(obj, dict) else None
                if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_PLAN_STEPS:
                    raise ValueError(f"steps 必须是非空子目标列表；资源上限为 {MAX_PLAN_STEPS} 项，请合并同一目标的零碎动作")
                normalized = []
                for i, step in enumerate(steps, 1):
                    if not isinstance(step, dict) or not isinstance(step.get("goal"), str) or not step["goal"].strip():
                        raise ValueError("每一步需要非空 goal")
                    if not isinstance(step.get("expected_evidence"), str) or not step["expected_evidence"].strip():
                        raise ValueError("每一步必须给出非空 expected_evidence，说明实际可以观察到的结果")
                    ids = step.get("check_ids", [])
                    if not isinstance(ids, list) or any(not isinstance(x, str) or x not in check_ids for x in ids):
                        raise ValueError("check_ids 只能引用给定的系统验收条件；没有时用 []")
                    try:
                        verification = [] if ids and "verification" not in step else normalize_verification(step.get("verification"))
                    except ValueError as exc:
                        raise ValueError(f"第 {i} 个子目标 verification：{exc}") from exc
                    normalized.append({"idx": i, "goal": step["goal"].strip(),
                                       "expected_evidence": str(step.get("expected_evidence") or ""),
                                       "check_ids": ids, "phase": str(step.get("phase") or ""), "status": "pending",
                                       "verification": verification,
                                       "actions": self._plan_actions(step.get("actions", []))})
                if state.task_profile.get("kind") == "paper_download":
                    phases = {step["phase"] for step in normalized}
                    allowed = {phase["phase"] for phase in state.task_profile["workflow"]}
                    required = {"verify"} if repair else allowed
                    if not phases.issubset(allowed):
                        raise ValueError("论文计划包含未知流程阶段")
                    if not required.issubset(phases):
                        raise ValueError("论文下载规划缺少流程阶段：" + ",".join(sorted(required - phases)))
                    ordered = [item["phase"] for item in state.task_profile["workflow"]]
                    indexes = [next(i for i, step in enumerate(normalized) if step["phase"] == phase)
                               for phase in ordered if phase in phases]
                    if indexes != sorted(indexes):
                        raise ValueError("流程阶段顺序必须为 " + " → ".join(ordered))
                    if not isinstance(obj.get("keywords"), list) or not any(isinstance(k, str) and k.strip() for k in obj["keywords"]):
                        raise ValueError("论文搜索必须提供非空 keywords")
                state.memory.plan = {"steps": normalized,
                    "requirements": audit_requirement_proposals(task, obj.get("requirements")),
                    "keywords": obj.get("keywords") if isinstance(obj.get("keywords"), list) else [],
                    "final_check": str(obj.get("final_check") or ""),
                    "completion_spec": copy.deepcopy(self.completion.public_spec)}
                state.memory.task_state["goal"] = task
                state.memory.task_state["current_subgoal"] = normalized[0]["goal"]
                state.queue = ExecutionQueue(state.memory.plan, revision=self._replans)
                state.progress_monitor = StepProgressMonitor(self.progress_config)
                state.step_progress = {}
                state.queue.sync_memory(state.memory)
                self._publish_diagnostic('task_analysis', analysis={**task_analysis(task, self.completion.public_spec),
                    'requirement_proposals': state.memory.plan['requirements']})
                self._publish_queue(state)
                state.memory.add_event(f"已制定任务计划：{len(normalized)} 步。只有系统证据可核实步骤完成。")
                call["valid_plan"] = True
                self._record_call(call, "planner")
                return True
            except Exception as exc:
                if call is None:
                    call = copy.deepcopy(getattr(self.planner_model, "last_call", {}))
                    call.update(messages=copy.deepcopy(messages), error_type=type(exc).__name__, valid_plan=False)
                    self.planning_calls.append(call)
                call["validation_error"] = str(exc)[:300]
                self._record_call(call, "planner")
                if attempt == 0:
                    if raw:
                        messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": f"规划不可用：{str(exc)[:300]}。请修正后输出完整 JSON。"})
        return False

    def _publish_review(self, report):
        self._emit("step_verification", report=report, policy_trainable=False)
        if _viz_log_status:
            try:
                _viz_log_status({"event": "step_verification", "step": self._current_step, **report})
            except Exception:
                pass

    def _publish_diagnostic(self, event, **payload):
        self._emit(event, **payload, policy_trainable=False)
        if _viz_log_status:
            try:
                _viz_log_status({'event': event, 'step': self._current_step, **payload})
            except Exception:
                pass

    def _track_step_progress(self, state, action, info, before, review=None):
        if state.queue is None:
            return
        if state.progress_monitor is None:
            state.progress_monitor = StepProgressMonitor(self.progress_config)
        report = state.step_review if review is None else review
        step = next((s for s in state.queue.plan['steps'] if s['id'] == report.get('plan_step_id')), state.queue.current)
        state.step_progress = state.progress_monitor.observe(step, report, before, state.obs, action, info)
        self._publish_diagnostic('step_progress', report=state.step_progress)
        if state.step_progress['status'] == 'stalled':
            state.failure_class = 'step_no_progress'
            state.failure_key = state.step_progress['plan_step_id']
            state.memory.stall_count = state.step_progress['idle_decisions']
            state.memory.add_event(state.step_progress['reason'])

    def _publish_queue(self, state):
        if not state.queue:
            return
        snapshot = state.queue.snapshot()
        self._emit("execution_queue", queue=snapshot, policy_trainable=False)
        if _viz_log_status:
            try:
                _viz_log_status({"event": "execution_queue", "step": self._current_step, "queue": snapshot})
            except Exception:
                pass

    def _record_typed_judge(self):
        if self.decision_judge and self.decision_judge.last_call:
            call = copy.deepcopy(self.decision_judge.last_call)
            self.reviewer.calls.append(call)
            self._record_call(call, 'verifier')

    def _review_visual(self, state, step, execution_info, code, identity, before=None):
        # Typed evaluator handles local effects/step checks. Whole-task semantic
        # review keeps the independent reviewer and all hard completion gates.
        typed = None
        if self.decision_judge and step.get('id') not in {'whole_task', 'final'}:
            self._publish_review({**identity, 'method': 'decision_model', 'status': 'running', 'reason': 'Decision 评估器正在核查动作前后证据'})
            typed = self.decision_judge.review(state.task, step, before, state.obs, execution_info, code,
                                               remaining_s=self._task_time_remaining())
            self._record_typed_judge()
            self._publish_review({**identity, **typed})
            state.routing_hint['surface'] = typed.get('judgment', {}).get('answers', {}).get('surface', {})
            if typed['status'] in {'passed', 'failed'}:
                return {**identity, **typed}
        self._publish_review({**identity, "method": "model", "status": "running", "reason": "独立核查截图与补充证据"})
        previous = len(self.reviewer.calls)
        report = self.reviewer.visual_review(state.task, step, state.obs, execution_info, code,
                                             remaining_s=self._task_time_remaining())
        for call in self.reviewer.calls[previous:]:
            self._record_call(call, "verifier")
        report = {**identity, **report}
        if typed:
            report['decision_judgment'] = typed
        self._publish_review(report)
        return report

    def _update_plan_progress(self, state: StepState, execution: Execution, before_observation=None, expected_effect='') -> None:
        """Review after each action, then advance exactly one queue entry on evidence."""
        state.finish_feedback = ""
        step = state.queue.current if state.queue else None
        if step is None:
            step = {"id": "final", "goal": state.task, "expected_evidence": "原始任务的全部要求得到满足",
                    "verification": [{"kind": "visual"}], "check_ids": []}
        identity = {"review_id": f"{self.run_id}:a{state.step_id}:{step['id']}",
                    "plan_step_id": step["id"], "goal": step["goal"], "decision_step": state.step_id}
        self._publish_review({**identity, "method": "code", "status": "running", "reason": "检查当前步骤的系统证据"})
        public_report = self.completion.public_report(state.verification)
        report = code_review(step, state.obs, execution.info, public_report)
        identity_error = paper_review_error(state.obs, self.completion, public_report, step)
        if identity_error:
            report = {"method": "code", "status": "failed", "reason": identity_error,
                      "evidence": [identity_error], "checks": [*report.get("checks", []),
                      {"kind": "paper_identity", "status": "failed", "reason": identity_error, "evidence": [identity_error]}]}
        # A complete trusted task contract is stronger than intermediate plan milestones.
        if state.verification.get("status") == "passed" and execution.kind != "terminal":
            report = {"method": "code", "status": "passed", "reason": "整项任务的全部确定性验收条件已满足",
                      "evidence": ["completion_contract"], "checks": report.get("checks", [])}
        report = {**identity, **report}
        self._publish_review(report)
        code_checks = copy.deepcopy(report.get('checks', []))
        if (report["status"] == "unknown" and execution.kind not in {"terminal", "deadline"}
                and not (state.obs.info if state.obs else {}).get("pending_downloads")
                and execution.info.get("status") not in {"pending", "in_progress"}):
            review_execution = {k: execution.info[k] for k in ('status', 'error_code', 'action') if k in execution.info}
            review_execution['expected_effect'] = expected_effect
            paper = active_paper(self.completion, step)
            if paper:
                report['bound_paper'] = {k: paper['binding'][k]
                                                    for k in ('full_title', 'canonical_id', 'pdf_url')}
            report = self._review_visual(state, step, review_execution, report, identity, before=before_observation)
        report['code_checks'] = code_checks
        state.step_review = report
        if state.queue:
            advanced = state.queue.accept_review(report)
            state.queue.sync_memory(state.memory)
            if advanced:
                self._clear_stall(state)
            self._publish_queue(state)
        state.memory.add_event(f"系统步骤核查[{report['method']}]: {report['status']}，{report['reason']}")

    def _programmatic_done(self, state: StepState, execution: Execution) -> str:
        return "contract" if state.verification.get("status") == "passed" else ""

    def _build_memory_text(self, state: StepState) -> str:
        return selected_memory(state.memory, state.step_review)

    def _program_info_text(self, state: StepState) -> str:
        coord = state.last_grounding_coord
        return "\n".join([
            "## 程序信息(program)",
            f"- step_id: {state.step_id}",
            f"- last_action: {state.last_action or '（无，这是第一步）'}",
            f"- last_target: {state.last_target or '（无）'}",
            f"- grounding_coordinate: {list(coord) if coord else '（无）'}",
            f"- execution_status: {state.execution_status or '（无）'}",
            f"- execution_feedback: {json.dumps(state.execution_feedback, ensure_ascii=False)}",
            f"- failure_count: {state.failure_count}",
        ])

    def _build_images(self, state: StepState) -> List[Tuple[str, Any]]:
        history = state.screenshot_history
        n = len(history)
        images: List[Tuple[str, Any]] = []
        if n >= 3:
            images.append(("【上上步截图 S(t-2)，原图】", history[-3]))
        if n >= 2:
            s_t1 = history[-2]
            if state.last_grounding_coord is not None:
                x, y = state.last_grounding_coord
                s_t1 = mark_coordinate(s_t1, x, y)
                images.append(("【上一步截图 S(t-1)，红圈=上一步实际点击位置】", s_t1))
            else:
                images.append(("【上一步截图 S(t-1)，原图】", s_t1))
        images.append(("【当前截图 S(t)，原图】", history[-1]))
        return images

    def _compute_control_state(self, state: StepState, decision: Decision,
                               execution: Execution, before_screenshot, before_observation=None) -> str:
        stall_reason = self._detect_stall(state, decision, execution, before_screenshot, before_observation)
        self._track_step_progress(state, decision.action, execution.info, before_observation or state.obs)
        if state.step_progress.get('status') == 'stalled':
            stall_reason = state.step_progress['reason']
        if stall_reason:
            state.memory.control_state = "STALL"
            sig = self._stable_action_key(decision.action)
            state.memory.add_forbidden_repeat(sig)
            state.memory.add_event(f"⚠️ STALL：{stall_reason}。请依据系统反馈改变策略；任务是否完成由系统验收判断。")
            return "STALL"
        if state.memory.stall_count:
            state.memory.control_state = "RECOVER"
        elif execution.info.get("effect_verified"):
            state.memory.control_state = "ADVANCE"
        else:
            state.memory.control_state = "CONTINUE"
        return state.memory.control_state

    @staticmethod
    def _brief_outcome(execution) -> str:
        """把执行结果精简成一句话，供动作历史使用（避免冗长 JSON 撑爆记忆预算）。"""
        info = execution.info if isinstance(execution.info, dict) else {}
        status = info.get("status")
        reason = str(info.get("reason") or info.get("error") or info.get("error_code") or execution.outcome or "缺少执行反馈")
        if status in {"rejected", "failed", "error"}:
            dispatched = "未下发" if info.get("dispatched") is False else "已下发" if info.get("dispatched") is True else "下发状态未知"
            return f"{status}（{dispatched}）：{reason[:160]}"
        if status == "uncertain":
            return f"结果不确定：{reason[:120]}；需要重新观察并验证，不能直接重复执行"
        if status in {"in_progress", "pending"}:
            return "操作仍在进行，等待框架事件验证，不能重复提交"
        if info.get("action") == "download":
            if info.get("status") == "already_downloaded":
                return "已下载过（跳过）"
            if info.get("verified") or any(a.get("completed") is True for a in info.get("artifacts", []) if isinstance(a, dict)):
                return f"下载已产生文件 {info.get('bytes', '?')}B，任务匹配由系统验收判断"
            return "已请求下载，尚无下载完成证据"
        if info.get("action") == "goto" and status == "executed":
            return f"已打开 {str(info.get('url', ''))[:60]}"
        if status == "executed":
            return "已执行"
        return (execution.outcome or "")[:80]

    @staticmethod
    def _stable_action_key(action) -> str:
        """稳定的动作签名：忽略每次变化的 ref/id/坐标/href，用于「重复动作」卡死检测。

        action_signature 会把 element_id/target_ref 也算进去（每次快照 ref 都变），
        导致「连续点同一链接」被判成不同动作、卡死检测失效。这里只保留语义 + 稳定参数。
        """
        if not action:
            return ""
        a = action.action
        target = str(action.args.get("target", "") or action.args.get("from_target", ""))
        target = re.sub(r"\(href[^)]*\)", "", target)          # 去掉 (href ...) 括号形式
        target = re.sub(r"href\s+[^\s,)]*", "href", target)    # 归一化 "href URL" → "href"
        target = re.sub(r"\[\d+\]", "", target)                # 去掉方括号数字 ref
        target = Pipeline._normalize_target(target)
        stable = {"action": a, "target": target[:200]}
        for k in ("text", "option", "key", "url", "amount", "direction", "overwrite"):
            if k in action.args:
                stable[k] = action.args[k]
        return json.dumps(stable, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _normalize_target(text):
        text = unicodedata.normalize("NFKC", str(text)).casefold()
        return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))

    def _semantic_action_key(self, action, obs, state=None):
        signature = json.loads(self._stable_action_key(action)) if action else {"action": "parse"}
        alias = json.dumps([self._observation_page(obs), signature], sort_keys=True, ensure_ascii=False)
        if action and obs is not None:
            nodes = [n for b in obs.context if b.get("kind") == "browser_ax" for n in b.get("nodes", [])]
            ref = action.args.get("target_ref") or action.args.get("element_id")
            node = next((n for n in nodes if ref and str(n.get("ref")) == str(ref)), None)
            if node is None:
                target = self._normalize_target(action.args.get("target") or action.args.get("from_target") or "")
                candidates = [n for n in nodes if self._normalize_target(n.get("name", ""))
                              and self._normalize_target(n.get("name", "")) in target]
                if len(candidates) == 1:
                    node = candidates[0]
            if node:
                signature["target"] = {"role": node.get("role"), "name": self._normalize_target(node.get("name", "")),
                                       "href": node.get("href") or node.get("url") or ""}
                if state is not None:
                    state.semantic_targets[alias] = signature["target"]
            elif state is not None and alias in state.semantic_targets:
                signature["target"] = state.semantic_targets[alias]
        return signature

    def _count_failure(self, state, failure_class, action, obs):
        key = json.dumps({"failure": failure_class, "page": self._observation_page(obs),
                          "target": self._semantic_action_key(action, obs, state)}, ensure_ascii=False, sort_keys=True)
        state.failure_counts[key] = state.failure_counts.get(key, 0) + 1
        state.failure_count += 1
        state.failure_class = failure_class
        state.failure_key = key
        state.memory.stall_count = state.failure_counts[key]
        return state.memory.stall_count

    @staticmethod
    def _clear_stall(state):
        state.failure_counts.clear()
        state.failure_class = state.failure_key = ""
        state.memory.stall_count = 0
        state.no_change_count = 0
        state.action_keys.clear()
        state.shell_outputs.clear()
        state.semantic_targets.clear()
        state.memory.forbidden_repeats.clear()

    @staticmethod
    def _semantic_observation(obs):
        if obs is None:
            return None
        return [{"kind": block.get("kind"), "url": block.get("url"),
                 "nodes": [{k: n[k] for k in ("role", "name", "value", "checked", "selected", "expanded", "focused", "disabled", "states", "href", "url") if k in n}
                           for n in block.get("nodes", [])]}
                for block in obs.context if block.get("kind") == "browser_ax"]

    def _detect_stall(self, state: StepState, decision: Decision,
                      execution: Execution, before_screenshot, before_observation=None) -> Optional[str]:
        """Count one actual failure per decision, scoped to semantic target and navigation.

        Snapshot refs never participate; model self-assessment never supplies proof
        of progress. A changed page/value/file or verified effect clears old failures.
        """
        before_observation = before_observation or state.obs
        info = execution.info or {}
        failure = info.get("wait_error_code")
        if not failure and info.get("status") in {"rejected", "failed", "failure", "error", "uncertain"}:
            failure = str(info.get("error_code") or info.get("status"))
        if not failure and (info.get("pending_downloads") or info.get("status") in {"in_progress", "pending"}):
            return None
        if not failure:
            before_ids = {self._download_id(a) for a in (before_observation.info if before_observation else {}).get("artifacts", [])}
            after_ids = {self._download_id(a) for a in info.get("artifacts", []) if a.get("completed")}
            changed = (bool(info.get("effect_verified")) or bool(after_ids - before_ids)
                       or (execution.obs is not None and (
                           self._observation_page(before_observation) != self._observation_page(execution.obs)
                           or self._semantic_observation(before_observation) != self._semantic_observation(execution.obs))))
            visible_change = execution.obs is not None and self._screen_diff(before_screenshot, execution.obs.screenshot) >= _NO_CHANGE_DIFF_THRESHOLD
            if changed or (visible_change and before_screenshot is not None):
                self._clear_stall(state)
                return None
            if execution.kind == "shell":
                output = state.terminal_result or ""
                if state.shell_outputs and output != state.shell_outputs[-1]:
                    self._clear_stall(state)
                state.shell_outputs.append(output)
                failure = "unchanged_shell_output"
            elif execution.kind in {"env", "browser_dom"} and decision.action and decision.action.action not in {
                    "wait", "press", "hotkey", "handle_dialog", "switch_tab"}:
                failure = "no_progress"
            else:
                return None
        count = self._count_failure(state, failure, decision.action, before_observation)
        limit = max(1, self.early_stop_repeat) if failure in {"no_progress", "unchanged_shell_output"} else _FAILURE_REPEAT_LIMIT
        state.action_keys.append(state.failure_key)
        state.action_keys[:] = state.action_keys[-_LOOP_HISTORY_LEN:]
        if count >= limit:
            return f"同一页面与语义目标的 {failure} 已发生 {count} 次"
        return None

    def _model_review_allowed(self, report):
        # Models may review only genuinely unspecified/semantic requirements.
        # A missing PDF, failed URL, or private check can never be waved through.
        unresolved = {c["id"] for c in self.completion.public_spec.get("checks", [])
                      if c.get("kind") == "unresolved_requirement"}
        checks = report.get("checks", [])
        return all(c.get("status") == "passed" or c.get("id") in unresolved for c in checks)

    def _verify_finish(self, state: StepState, answer: str) -> Tuple[bool, str]:
        report = self.completion.evaluate()
        state.verification = report
        self._verification = copy.deepcopy(report)
        if report.get("status") == "passed":
            return True, "全部确定性验收条件通过"
        if not self._model_review_allowed(report):
            public = self.completion.public_report(report)
            return False, public.get("reason") or "确定性验收条件尚未满足，模型不能覆盖"
        step = {"id": "whole_task", "goal": state.task,
                "expected_evidence": "核对原始任务的所有要求（包含数量、对象身份、保存或提交结果）；不能只核对当前步骤或规划摘要。",
                "verification": [{"kind": "visual"}]}
        identity = {"review_id": f"{self.run_id}:a{state.step_id}:whole_task",
                    "plan_step_id": "whole_task", "goal": state.task, "decision_step": state.step_id}
        code = {**identity, "method": "code", "status": "unknown", "reason": "明确条件已检查，剩余语义要求需独立核查",
                "checks": self.completion.public_report(report).get("checks", [])}
        self._publish_review(code)
        reviewed = self._review_visual(state, step, {}, code, identity)
        self._model_final_report = reviewed if reviewed["status"] == "passed" else None
        self._reviewed_state = state
        self._reviewed_observation = self._review_fingerprint(state.obs)
        return reviewed["status"] == "passed", reviewed["reason"]

    @classmethod
    def _review_fingerprint(cls, obs):
        if obs is None:
            return ""
        return hashlib.sha256((encode_image(obs.screenshot) + json.dumps(
            {"context": cls._semantic_observation(obs), "text": obs.text,
             "runtime": (obs.info or {}).get("runtime_state"),
             "pending_downloads": (obs.info or {}).get("pending_downloads"),
             "artifacts": (obs.info or {}).get("artifacts")}, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()

    @staticmethod
    def _screen_diff(a, b) -> float:
        """比较两张截图差异（64x64 灰度 mean-abs-diff，0~255 尺度，越小越相似）。

        纯 PIL 实现，不依赖 numpy。
        """
        if a is None or b is None:
            return 255.0
        try:
            from PIL import ImageChops
            ia = a.convert("L").resize((64, 64)) if hasattr(a, "convert") else None
            ib = b.convert("L").resize((64, 64)) if hasattr(b, "convert") else None
            if ia is None or ib is None:
                return 255.0
            hist = ImageChops.difference(ia, ib).histogram()
            return sum(i * count for i, count in enumerate(hist)) / (64 * 64)
        except Exception:  # noqa: BLE001
            return 255.0

    @staticmethod
    def _format_code_agent_result(result: Dict[str, Any]) -> str:
        lines = [
            f"- 任务: {result.get('task_instruction', '')}",
            f"- 执行步数: {result.get('steps_executed', 0)}/{result.get('budget', 0)}",
            f"- 完成状态: {result.get('completion_reason', '')}",
            f"- 摘要: {result.get('summary') or '（无）'}",
        ]
        notes = result.get("notes") or {}
        if notes:
            lines.append("- 子Agent 存下的笔记:")
            for k, v in notes.items():
                lines.append(f"  - {k}: {v}")
        hist = result.get("execution_history") or []
        if hist:
            shown = hist[:3] + (hist[-2:] if len(hist) > 3 else [])
            lines.append("- 关键执行历史（前3+后2步）:")
            for h in shown:
                lines.append(f"  - [{h.get('type')}] {h.get('code', '')[:80]} → {h.get('result', '')[:80]}")
        return "\n".join(lines)

    def _update_program_info(self, action: Action, execution: Execution, state: StepState) -> None:
        state.last_action = action.action
        state.last_target = self._target_of(action)
        if execution.kind == "finish_request":
            state.execution_status = "request_finish"
        elif execution.kind == "terminal":
            state.execution_status = "fail"
        else:
            state.execution_status = str(execution.info.get("status") or "unknown")
        state.execution_feedback = {key: copy.deepcopy(execution.info[key]) for key in (
            "status", "dispatched", "reason", "error", "error_code", "wait_error_code", "wait_reason", "effect_verified", "framework_wait")
            if key in execution.info}
        state.execution_feedback.setdefault("status", state.execution_status)
        state.execution_feedback.setdefault("dispatched", None)
        state.execution_feedback.setdefault("reason", execution.info.get("error") or execution.info.get("error_code")
                                            or execution.outcome[:200])

    def _validate_action(self, action: Action) -> Optional[str]:
        if action.action not in BASIC_ACTIONS:
            return "只允许基础 GUI/DOM 交互动作，不能调用代码、直接下载或直接修改文件"
        if self._active_actions is not None and action.action not in self._active_actions:
            return "本轮可用动作：" + ", ".join(self._active_actions)
        return validate_action(action)

    def _resolve(self, action: Action, obs: Observation) -> Action:
        spec = get_action_spec(action.action)
        if spec is None or spec.resolve == "none":
            return action
        # grounding 坐标空间 = 截图尺寸：浏览器场景截图是 viewport（CDP 截），桌面是全屏，
        # 这里按截图实际尺寸动态适配 grounding 的 width/height，坐标自然落在正确空间。
        if obs.screenshot is not None:
            try:
                w, h = obs.screenshot.size
                if hasattr(self.grounding, "width"):
                    self.grounding.width = int(w)
                    self.grounding.height = int(h)
            except Exception:  # noqa: BLE001
                pass
        if spec.resolve == "point":
            gr = self.grounding.locate(obs.screenshot, str(action.args.get("target", "")))
            coord = resolve_coords(gr)
            # 保留 target 之外的参数（如 type 的 text），让「定位+聚焦」类动作能带上其余参数
            extra = {k: v for k, v in action.args.items() if k != "target"}
            if coord is None:
                # UI-TARS 失败：尝试 OCR 文字定位兜底（对齐 AgentS3 的第二条 grounding 腿）
                coord = self._ocr_fallback(obs.screenshot, str(action.args.get("target", "")))
            if coord is None:
                # grounding 失败：不带可执行坐标，由执行器拒绝（空转），不做坐标钳制
                return Action(action.action, {"grounding_failed": True, **extra})
            x, y = coord
            return Action(action.action, {"x": x, "y": y, **extra})
        if spec.resolve == "select":
            gr = self.grounding.locate(obs.screenshot, str(action.args.get("target", "")))
            coord = resolve_coords(gr)
            if coord is None:
                coord = self._ocr_fallback(obs.screenshot, str(action.args.get("target", "")))
            if coord is None:
                return Action(action.action, {"grounding_failed": True,
                                              "option": action.args.get("option", "")})
            x, y = coord
            return Action(action.action, {"x": x, "y": y, "option": action.args.get("option", "")})
        if spec.resolve == "drag":
            g1 = self.grounding.locate(obs.screenshot, str(action.args.get("from_target", "")))
            g2 = self.grounding.locate(obs.screenshot, str(action.args.get("to_target", "")))
            c1, c2 = resolve_coords(g1), resolve_coords(g2)
            if c1 is None or c2 is None:
                return Action(action.action, {"grounding_failed": True})
            return Action(action.action, {"x1": c1[0], "y1": c1[1], "x2": c2[0], "y2": c2[1]})
        return action

    def _ocr_fallback(self, screenshot, target: str):
        """确定性的 OCR 兜底：仅定位归一化后唯一精确匹配的词或同行连续词组。

        不调用决策模型，也不生成定位训练样本。缺少依赖、OCR 失败、描述无法
        精确匹配或文本重复时返回 None，由正常决策循环重新观察和选择动作。
        """
        try:
            import unicodedata
            import pytesseract
            from PIL import Image
            from io import BytesIO
        except ImportError:
            return None

        def normalize(text):
            return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())

        needle = normalize(target or "")
        if not needle:
            return None

        try:
            img = screenshot
            if isinstance(screenshot, (bytes, bytearray)):
                img = Image.open(BytesIO(screenshot))
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            words = []
            for i, text in enumerate(data.get("text") or []):
                text = normalize(text or "")
                if not text:
                    continue
                left, top, width, height = (int(data[key][i]) for key in ("left", "top", "width", "height"))
                if left < 0 or top < 0 or width <= 0 or height <= 0:
                    return None
                words.append({
                    "text": text, "left": left, "top": top,
                    "right": left + width, "bottom": top + height,
                    "line": tuple(data[key][i] if key in data else None
                                  for key in ("page_num", "block_num", "par_num", "line_num")),
                    "word_num": int(data["word_num"][i]) if "word_num" in data else None,
                })
        except Exception:  # noqa: BLE001
            return None

        matches = []
        for start, first in enumerate(words):
            phrase = ""
            for end in range(start, len(words)):
                word = words[end]
                if word["line"] != first["line"]:
                    break
                if end > start and word["word_num"] is not None and (
                        word["word_num"] != words[end - 1]["word_num"] + 1):
                    break
                phrase = f"{phrase} {word['text']}".strip()
                if phrase == needle:
                    match = words[start:end + 1]
                    left, right = min(w["left"] for w in match), max(w["right"] for w in match)
                    top, bottom = min(w["top"] for w in match), max(w["bottom"] for w in match)
                    matches.append(((left + right) // 2, (top + bottom) // 2))
                    if len(matches) > 1:
                        return None
                    break
                if not needle.startswith(phrase + " "):
                    break
        return matches[0] if matches else None

    def _evaluate(self, result: AgentResult) -> float:
        def refresh_final_pages():
            checks = self.completion.spec.get("checks", [])
            if not any(check.get("scope") == "final" and check.get("kind") in {
                    "url_equals", "text_present", "element_state"} for check in checks):
                return
            # Evaluators, callbacks and redirects can change the page after the
            # last action observation. Final page predicates need fresh evidence;
            # milestones stay latched and artifact-only contracts avoid this call.
            observation = self._observe()
            self.completion.observe(observation)
            self._emit("observation", observation_id="o_final", stage="final_evaluation",
                       observation=self._observation_record(observation, "o_final"))

        if not getattr(self.env, "evaluation_available", False):
            refresh_final_pages()
            report = self.completion.evaluate()
            was_success = result.success
            model_report = self._model_final_report
            if model_report and self._model_review_allowed(report) and report.get("status") != "passed":
                # Observe again after callbacks; revoke/review if visible evidence changed.
                state = self._reviewed_state
                fresh = self._observe()
                self.completion.observe(fresh)
                report = self.completion.evaluate()
                if not self._model_review_allowed(report):
                    model_report = None
                elif self._review_fingerprint(fresh) != self._reviewed_observation:
                    state.obs = fresh
                    confirmed, _ = self._verify_finish(state, "")
                    model_report = self._model_final_report if confirmed else None
                if model_report:
                    result.success = True
                    result.evaluation_available = False
                    result.reward_source = "independent_model_review"
                    result.termination_reason = "verified_by_model"
                    result.verification = {**copy.deepcopy(model_report), "source": "independent_model_review",
                                           "reward_valid": False, "completion": report}
                    self._emit("verification", stage="final_evaluation", report=result.verification, policy_trainable=False)
                    return 0.0
            result.success = report.get("status") == "passed"
            result.verification = copy.deepcopy(report)
            if result.success and not was_success:
                result.termination_reason = "verified"
                result.final_answer = "框架已核实全部任务条件；任务状态满足要求。"
            if was_success and not result.success:
                result.termination_reason = "verification_changed"
                result.final_answer = "最终验收未通过，任务完成证据已失效。"
            result.evaluation_available = bool(self.completion.spec.get("checks")) and report.get("status") in {"passed", "failed"}
            result.reward_source = "completion_contract" if result.evaluation_available else "unavailable"
            self._emit("verification", stage="final_evaluation", report=result.verification)
            return 1.0 if result.success and report.get("status") == "passed" else 0.0
        try:
            score = float(self.env.evaluate())
            import math
            if not math.isfinite(score):
                raise ValueError("non-finite evaluator score")
            refresh_final_pages()
            contract = self.completion.evaluate()
            has_contract = bool(self.completion.spec.get("checks"))
            environment_passed = score >= 1.0
            status = ("passed" if environment_passed else "failed")
            if has_contract:
                if contract.get("status") == "failed":
                    status, score = "failed", 0.0
                elif environment_passed and contract.get("status") != "passed":
                    status, score = "unknown", 0.0
            result.evaluation_available = status != "unknown"
            result.reward_source = ("combined_evaluation" if has_contract else "environment_evaluator") if result.evaluation_available else "unavailable"
            result.success = status == "passed"
            result.verification = {"status": status, "source": result.reward_source, "score": score,
                "environment_passed": environment_passed,
                **({"completion": contract, "checks": contract.get("checks", [])} if has_contract else {})}
            if result.success:
                result.termination_reason = "verified_by_environment"
            elif status == "unknown":
                result.termination_reason = "completion_unverifiable"
                result.final_answer = "环境评测通过，但附加任务条件尚无足够证据，无法确认全部完成。"
            elif result.termination_reason in {"evaluation_requested", "verified"}:
                result.termination_reason = "evaluation_failed"
                result.final_answer = "独立评测未通过，任务尚未完成。"
            self._emit("verification", stage="final_evaluation", report=result.verification)
            return score
        except NotImplementedError:
            result.success = False
            result.evaluation_available = False
            result.reward_source = "unavailable"
            result.termination_reason = "evaluation_error"
            result.verification = {"status": "unknown", "source": "environment_evaluator", "reason": "not implemented"}
            result.final_answer = "独立评测不可用，无法确认任务完成。"
            self._emit("verification", stage="final_evaluation", report=result.verification)
            return 0.0
        except Exception as exc:  # noqa: BLE001
            # 评测器自身失败（如 postconfig 要下载的评测脚本被网络/代理拦掉）不应毁掉整次运行：
            # agent 的实际操作已经完成，把评测异常记入轨迹、按 0 分返回，保证 result/轨迹正常落盘。
            result.trajectory.append({
                "step": result.steps,
                "eval_error": f"评测阶段异常（非 agent 操作失败）：{str(exc)[:500]}",
            })
            result.success = False
            result.evaluation_available = False
            result.reward_source = "unavailable"
            result.termination_reason = "evaluation_error"
            result.verification = {"status": "unknown", "source": "environment_evaluator", "reason": type(exc).__name__}
            result.final_answer = "独立评测失败，无法确认任务完成。"
            self._emit("verification", stage="final_evaluation", report=result.verification)
            return 0.0

    # ---- 纯静态小工具 ----
    @staticmethod
    def _coordinate_of(action: Action) -> Optional[Tuple[int, int]]:
        a = action.args
        if "x" in a and "y" in a:
            return int(a["x"]), int(a["y"])
        if "x1" in a and "y1" in a:
            return int(a["x1"]), int(a["y1"])
        return None

    @staticmethod
    def _target_of(action: Action) -> str:
        a = action.args
        if action.action in ("click", "double_click", "right_click", "select"):
            return str(a.get("target", ""))
        if action.action == "drag_and_drop":
            return f"{a.get('from_target', '')} -> {a.get('to_target', '')}"
        if action.action == "type":
            t = str(a.get("target", ""))
            txt = str(a.get("text", ""))
            if t:
                return f"{txt!r} → {t}"
            return txt
        if action.action == "scroll":
            return f"{a.get('direction', '')} {a.get('amount', '')}"
        if action.action == "run_in_terminal":
            return str(a.get("command", ""))
        if action.action == "goto":
            return str(a.get("url", ""))
        if action.action == "hotkey":
            return "+".join(a.get("keys", []))
        if action.action == "press":
            return str(a.get("key", ""))
        return ""

    @staticmethod
    def _action_desc(action: str, target: str) -> str:
        if target:
            return f"{action} {target}"
        return action

    @staticmethod
    def _compact_action(action) -> str:
        if isinstance(action, Action):
            return json.dumps({"action": action.action, "args": action.args}, ensure_ascii=False)
        return str(action)

    @staticmethod
    def _describe_outcome(step_result) -> str:
        if step_result.done:
            return "任务完成"
        if step_result.reward and step_result.reward > 0:
            return f"有进展(奖励 {step_result.reward})"
        return "已执行（屏幕状态待下一步判断）"
