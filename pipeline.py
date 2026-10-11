"""主循环流水线：观察 → 决策（证据先行）→ 定位 → 执行 → 控制状态更新。

闭环设计（核心）：
- 执行前，模型在 next_action 的同时说明 expected_effect（预期观察到什么）；
- 执行后，模型先客观描述 observed_effect，再给 action_status（成功/失败）；
- 程序对比 expected vs observed + 确定性卡死检测，得出 control_state；
- Planner 只能 request_finish，程序验证后才真正结束。

阶段（run() 里按序调用）：
  ① preprocess   预处理：确保观察 + 记录截图 + 构建记忆文本
  ② build_input  构造输入：结构化记忆（含控制状态）+ 3 张截图
  ③ decide       决策：observed_effect + action_status + state_update + next_action + expected_effect
  ④ ground       grounding：描述→坐标，记录坐标供下一步标记
  ⑤ execute      执行：skill / shell / subagent / terminal(request_finish|fail) / env
  ⑥ postprocess  后处理：记录 expected/observed + 控制状态 + 卡死检测 + Finish 验证
"""
from __future__ import annotations

import copy
import json
import time
import hashlib
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
from .skills import is_skill, validate_skill, run_skill, enabled_skills, SkillContext
from .prompts import build_user_prompt, build_format_feedback, build_planner_prompt
from .context import PromptCompiler, ContextBudgetError, ObservationIncompleteError
from .verifiers import CompletionVerifier
from .task_specs import task_profile

# 实时状态推送（可选）：execute 阶段进度推给 viz 前端，失败不影响主流程
try:
    from .viz.logging import log_status as _viz_log_status
except Exception:  # noqa: BLE001
    _viz_log_status = None

# 确定性卡死检测参数
_NO_CHANGE_DIFF_THRESHOLD = 2.0       # 64x64 灰度 mean-abs-diff 阈值（< 此值视为无变化）
_NO_CHANGE_CONSECUTIVE_THRESHOLD = 2  # 连续 N 步无变化判定卡死
_REPEAT_ACTION_THRESHOLD = 3          # 连续 N 次相同 action+target 判定卡死
_LOOP_HISTORY_LEN = 6                 # 周期循环检测的历史长度
_LOOP_PERIODS = (2,)                  # 周期循环（A,B,A,B），周期1由 _REPEAT_ACTION_THRESHOLD 覆盖


# --------------------------------------------------------------------------- #
# 数据容器
# --------------------------------------------------------------------------- #
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


@dataclass
class Decision:
    """③ 决策阶段的产出（先分析当前屏幕 → 证据先行 → 预期效果）。"""
    raw: str = ""
    action: Optional[Action] = None
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
                 planner_model=None, event_path=None, run_id=None, run_manifest=None, max_replans=1):
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

    # ================= 主循环 ================= #
    def run(self, task: str) -> AgentResult:
        self.completion = CompletionVerifier(task, self.completion_spec, self.private_completion_spec)
        self.events = []
        manifest = self._make_manifest(task)
        result = AgentResult(task=task, run_id=self.run_id, run_manifest=manifest,
            private_evaluation={"spec": copy.deepcopy(self.private_completion_spec), "verifier_version": "3.0"})
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
        try:
            result = self._run(task, result)
            self._emit("episode_end", outcome={
                "success": result.success, "score": result.score,
                "reward": result.score if result.evaluation_available else None,
                "reward_valid": result.evaluation_available, "reward_source": result.reward_source,
                "termination_reason": result.termination_reason, "verification": result.verification})
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
            cfg = getattr(model, "cfg", None)
            return {key: getattr(cfg, key, None) for key in (
                "name", "policy_revision", "tokenizer_revision", "processor_revision",
                "chat_template_revision", "temperature", "max_tokens", "capture_rollout")}
        root = Path(__file__).resolve().parent
        source_hash = hashlib.sha256()
        for name in ("pipeline.py", "prompts.py", "context.py", "browser_session.py", "verifiers.py",
                     "task_specs.py", "actions.py", "config/action.yaml", "model/decision_model.py"):
            source_hash.update(name.encode())
            source_hash.update((root / name).read_bytes())
        manifest = {"run_id": self.run_id, "task": task, "framework_source_hash": source_hash.hexdigest(),
                    "compiler_version": self.compiler.version, "verifier_version": "3.0",
                    "decision_model": model_record(self.model), "planner_model": model_record(self.planner_model),
                    "planner_trainable": self.planner_model is self.model,
                    "grounder_trainable": False, "max_steps": self.max_steps,
                    "public_completion": self.completion.public_spec,
                    "completion_hash": hashlib.sha256(json.dumps(self.completion.spec, sort_keys=True).encode()).hexdigest(),
                    **self.run_manifest}
        manifest["run_id"] = self.run_id
        runtime = {"context": asdict(self.compiler.config), "max_steps": self.max_steps,
                   "max_retries": self.max_retries, "max_replans": self.max_replans,
                   "early_stop_repeat": self.early_stop_repeat,
                   "enable_reflection": self.enable_reflection,
                   "decision_model": model_record(self.model), "planner_model": model_record(self.planner_model)}
        manifest["runtime_config"] = runtime
        manifest["runtime_config_hash"] = hashlib.sha256(
            json.dumps(runtime, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return manifest

    def _record_call(self, call, actor_role="decision"):
        call.setdefault("call_id", f"{self.run_id}:call:{len(self.events) + 1}")
        call["actor_role"] = actor_role
        call["policy_trainable"] = actor_role == "decision" or self.planner_model is self.model
        self._emit("model_call", actor_role=actor_role, source="model", call=call)

    def _run(self, task: str, result: AgentResult) -> AgentResult:
        state = self._init_state(task)
        state.task_profile = task_profile(task, self.completion.public_spec)
        # 规划是执行前置条件。小模型规划失败时不盲目开始操作。
        planned = self._plan(task, state)
        result.plan = state.memory.plan
        result.planning_calls = self.planning_calls
        if not planned:
            result.termination_reason = "planning_failed"
            result.final_answer = "任务拆解失败，未执行任何操作。请检查规划模型输出。"
            result.verification = {"status": "unknown", "reason": "planning_failed", "checks": []}
            return result

        for step_no in range(1, self.max_steps + 1):
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
            decision = self.decide(messages)
            for call in decision.calls:
                call.setdefault("actor_role", "decision")
                call.setdefault("provenance", "sampled")
            if decision.action is None:
                # 动作解析/校验失败：空转一步继续（对齐 Agent S3 的 wait 语义），
                # 不提前终止任务——9B 模型偶尔漏 next_action，下一轮往往能恢复。
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
                    "next_observation": copy.deepcopy(before),
                })
                self._emit("transition", transition=result.trajectory[-1])
                state.memory.add_event("⚠️ 上一步动作解析失败，请重新输出合法的 next_action JSON")
                continue
            ground_started = time.perf_counter()
            action = self.ground(decision.action, state)
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
            execution = self.execute(action, state)
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
                    "verification": copy.deepcopy(state.verification),
                    "terminated": bool(should_break and (result.success or execution.done)),
                    "truncated": bool(should_break and not (result.success or execution.done)),
                    "timing_ms": {"ground": round(ground_ms, 2), "execute_observe": round(execute_ms, 2),
                                  "total": round((time.perf_counter() - started) * 1000, 2)},
                })
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
            "dialog", "tabs", "active_modals", "browser_events", "artifacts", "screenshot_stale")
            if k in observation_info}
        if browser_events:
            blocks.insert(0, {"kind": "browser_events", "text": json.dumps(browser_events, ensure_ascii=False)})
        if state.obs and state.obs.text:
            blocks.append({"kind": "desktop_a11y", "text": state.obs.text})
        # 显式场景状态：让模型明确知道「有结构走 DOM / 无结构走视觉」
        hint = scene_hint(state.obs)
        blocks.insert(0, {"kind": "scene", "text": "当前场景: " + json.dumps(hint, ensure_ascii=False)})
        auxiliary = json.dumps({"skills": state.skill_results, "terminal": state.terminal_result,
                                "code_agent": state.code_agent_result, "page_source": page_source,
                                "browser_result": browser_result}, ensure_ascii=False)
        images = self._build_images(state) if getattr(self.model, "vision", True) else []
        if observation_info.get("dialog"):
            # Native JS dialogs may block a new screenshot. Their fresh event
            # permits only dialog recovery; any retained image is explicitly stale.
            images = [(label, shot) for label, shot in images if shot is not None]
        user_text, images, metadata = self.compiler.compile(
            system=self.system_prompt, task=state.task, memory=state.memory_text,
            blocks=blocks, auxiliary=auxiliary,
            required=json.dumps({"plan": state.memory.plan,
                                 "task_profile": state.task_profile, "recovery": state.recovery,
                                 "phase_events": state.phase_events[-8:],
                                 "completion_spec": self.completion.public_spec}, ensure_ascii=False),
            images=images,
            require_image=bool(getattr(self.model, "vision", True)) and not bool(observation_info.get("dialog")))
        state.prompt_metadata = metadata
        state.prompt_text = user_text
        if state.obs:
            state.obs.info["exposed_refs"] = metadata["exposed_refs"]
        return [
            {"role": "system", "content": self.system_prompt},
            self.model.build_vision_message(user_text, images),
        ]

    def decide(self, messages: List[dict]) -> Decision:
        raw, action = "", None
        screen_analysis = observed_effect = action_status = expected_effect = ""
        state_update: Dict[str, Any] = {}
        calls: List[dict] = []
        for _ in range(max(1, self.max_retries)):
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
            call.update({"messages": copy.deepcopy(messages), "response": raw, "valid_action": False})
            calls.append(call)
            obj = parse_decision(raw)
            action = action_from_decision(obj or {})
            if action is not None:
                err = self._validate_action(action)
                patch = (obj or {}).get("state_update") or {}
                if err is None and (not isinstance(patch, dict)
                        or not isinstance(patch.get("known_fact_add", {}), dict)
                        or not isinstance(patch.get("completed_add", []), list)
                        or not isinstance(patch.get("pending_add", []), list)):
                    err = "state_update 必须是对象，known_fact_add 是对象，completed_add/pending_add 是列表"
                if err is None:
                    call["valid_action"] = True
                    screen_analysis = str((obj or {}).get("screen_analysis") or "")
                    observed_effect = str((obj or {}).get("observed_effect") or "")
                    action_status = str((obj or {}).get("action_status") or "")
                    state_update = (obj or {}).get("state_update") or {}
                    expected_effect = str((obj or {}).get("expected_effect") or "")
                    self._record_call(call)
                    return Decision(raw=raw, action=action, screen_analysis=screen_analysis,
                                    observed_effect=observed_effect, action_status=action_status,
                                    state_update=state_update, expected_effect=expected_effect,
                                    calls=calls)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": build_format_feedback(err)})
            else:
                messages.append({"role": "assistant", "content": raw})
                if obj and "next_action" not in obj:
                    feedback = "你的 JSON 里漏了 next_action 字段（screen_analysis/observed_effect/state_update 都写了，就缺 next_action）。请补上 next_action，里面有 type 和对应参数。"
                elif obj and "next_action" in obj and not isinstance(obj["next_action"], dict):
                    feedback = "next_action 必须是一个对象（含 type 字段），现在不是合法对象。请改成 {\"type\": \"动作名\", ...}。"
                else:
                    feedback = "没有找到合法的 JSON 动作（需含 next_action）。"
                messages.append({"role": "user", "content": build_format_feedback(feedback)})
            self._record_call(call)
        return Decision(raw=raw, action=None, calls=calls)

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
                   observation=self._observation_record(step_result.observation, "after_dom"))
        if (info.get("status") != "rejected" or info.get("dispatched") is not False
                or info.get("error_code") not in {"not_found", "unsupported"}
                or action.action not in {"click", "double_click", "right_click", "type", "select", "drag_and_drop"}):
            return step_result
        fresh = self._observe()
        self._emit("observation", observation_id="recovery", observation=self._observation_record(fresh, "recovery"))
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
        self._emit("execution_attempt", action=self._compact_action(visual), execution_info=recovered.info)
        recovered.info = dict(recovered.info or {})
        recovered.info["recovery"] = {"from": copy.deepcopy(info),
                                      "action": self._compact_action(visual),
                                      "observation": self._observation_record(fresh, "recovery")}
        state.resolved_action = visual
        state.last_grounding_coord = self._coordinate_of(visual)
        return recovered

    def execute(self, action: Action, state: StepState) -> Execution:
        name = action.action
        # browser_dom 通道：路由成功后走 DOM 定位执行（跳过 grounding）
        if state.route and state.route.get("channel") == "browser_dom":
            try:
                step_result = self.env.execute_routed(action, state.route)
                step_result = self._recover_browser(action, step_result, state)
                info = step_result.info if isinstance(step_result.info, dict) else {}
                # 下载是单项执行证据；提示模型继续核对其余目标，不能据此宣告整项任务完成。
                if info.get("action") == "download":
                    import os as _os
                    path = info.get("path", "")
                    if info.get("status") == "executed":
                        if info.get("verified"):
                            state.memory.add_event(
                                f"文件已下载：{path}（{info.get('bytes', 0)} 字节）。继续检查其余任务目标。")
                            state.memory.known_facts["downloaded_file"] = path
                        else:
                            state.memory.add_event(
                                f"⚠️ 下载写入异常：{path} 落盘校验失败，可能需要重新下载。")
                    elif info.get("status") == "already_downloaded":
                        # 程序侧核实文件还在不在（不靠模型自评）
                        exists = bool(path) and _os.path.exists(path)
                        if exists:
                            state.memory.add_event(f"文件已在本地存在：{path}。继续检查其余任务目标。")
                        else:
                            state.memory.add_event(
                                f"⚠️ 文件已下载过但当前不存在：{path}，需要重新下载。")
                        state.memory.known_facts["downloaded_file"] = path
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
                                   "error_code": state.route.get("error_code", "stale")})

        if is_skill(name):
            out = run_skill(name, action.args, state.skill_ctx)
            state.skill_results[name] = out
            return Execution(kind="skill", executed=f"技能 {name}", outcome="结果将在下一步返回")

        spec = get_action_spec(name)
        kind = spec.kind if spec else "env"

        if kind == "shell":
            command = str(action.args.get("command", ""))
            output = self.env.run_command(command)
            # 对齐 AgentS3 的 run_in_terminal 语义：原框架是「打开终端 → pyautogui 真打字」，
            # 命令会写进 ~/.bash_history；我们 subprocess 执行不写。但有些 evaluator 检查
            # 「是否从终端执行」是靠 grep ~/.bash_history（如 force-quit 任务查 kill 命令），
            # 这里补写一行，保证 shell 命令在 history 里可见，不改变命令本身的行为。
            try:
                self.env.run_python(
                    "import os\n"
                    "_h = os.path.expanduser('~/.bash_history')\n"
                    f"with open(_h, 'a') as _f: _f.write({command!r} + '\\n')\n"
                )
            except Exception:  # noqa: BLE001  补写失败不影响命令已执行的事实
                pass
            state.terminal_result = output[:4000] + ("\n...(输出过长已截断)" if len(output) > 4000 else "")
            return Execution(kind="shell", executed=f"终端命令: {command}",
                             outcome="命令已执行，输出将在下一步返回", obs=self._observe())

        if kind == "browser":
            if name == "get_page_source":
                result = self.env.browser_action(name, action.args)
                state.page_source = result[:12000] + ("\n...(源码过长已截断)" if len(result) > 12000 else "")
                return Execution(kind="browser", executed="浏览器: get_page_source",
                                 outcome="页面源码将在下一步返回")
            step_result = self.env.step(action)
            result = json.dumps(step_result.info or {}, ensure_ascii=False)
            state.browser_result = result
            return Execution(kind="browser", executed=f"浏览器: {name}",
                             outcome=result, info=step_result.info or {}, obs=step_result.observation,
                             done=step_result.done, reward=step_result.reward)

        if kind == "subagent":
            if self.code_agent is None:
                return Execution(kind="subagent", executed="子Agent", outcome="子Agent未配置")
            task = str(action.args.get("task", "") or state.task)
            result = self.code_agent.execute(task)
            state.code_agent_result = self._format_code_agent_result(result)
            for k, v in (result.get("notes") or {}).items():
                state.memory.known_facts[k] = v
            return Execution(kind="subagent", executed=f"子Agent: {task[:60]}",
                             outcome="子Agent已执行，结果将在下一步返回",
                             obs=self._observe(),
                             info={"action": "call_code_agent", "task": task,
                                   "result": state.code_agent_result[:1000]})

        if kind == "terminal":
            if name == "fail":
                return Execution(kind="terminal", executed=name, outcome="任务失败",
                                 done=True, success=False, answer="")
            # request_finish：不立即结束，交给 postprocess 验证
            return Execution(kind="finish_request", executed=name, outcome="请求结束，待验证",
                             obs=self._observe(), done=False, success=False,
                             answer=str(action.args.get("answer", "")))

        # env / uno 动作：此时 action 已坐标化。
        # uno = UNO 直写（set_cell_values / save，经 LibreOffice 2002 端口），与 GUI 点击
        # 共用 env.step 通道，但验证方式不同（uno 读回单元格 / 确认落盘，env 看屏幕变化）。
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

        # UNO 直写动作的写后校验（程序侧，不靠模型自觉）
        if kind == "uno" and action.action == "set_cell_values":
            try:
                from .adapters.vmware_env import verify_cell_values
                ok, detail = verify_cell_values(
                    self.env,
                    action.args.get("new_cell_values") or {},
                    str(action.args.get("app_name") or ""),
                    str(action.args.get("sheet_name") or "Sheet1"),
                )
                outcome += f"；写入校验：{'通过' if ok else '失败'}"
            except Exception:  # noqa: BLE001  校验失败不影响主流程
                pass

        return Execution(kind=kind, executed=self._compact_action(action),
                         outcome=outcome,
                         obs=step_result.observation, done=step_result.done,
                         info=step_result.info or {}, reward=step_result.reward)

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
                state.failure_count += 1
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
        # 6) env 动作更新观察
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
        self._update_plan_progress(state, execution)
        # 6.5) 单步动作有效性校验：env 动作执行后屏幕无变化 → 当场提示换策略
        # （不等到连续 2 步 STALL，每一步点空/点错都立即提示，避免反复点同一无效目标）
        if execution.kind == "env" and execution.obs is not None and before_screenshot is not None:
            if self._screen_diff(before_screenshot, execution.obs.screenshot) < _NO_CHANGE_DIFF_THRESHOLD:
                state.memory.add_event(
                    "⚠️ 这一步动作后屏幕没有可见变化：可能点空 / 点错 / 焦点不对。"
                    "下一步请换一种方式（键盘快捷键 / 不同菜单路径 / 重新定位目标），不要重复同样的点击。")
        # 7) 控制状态 + 确定性卡死检测
        self._compute_control_state(state, decision, execution, before_screenshot)
        # 7.5) 持久卡死（连续 STALL 3 次）→ 硬停止，不再只警告
        should_break = False
        if state.memory.control_state == "STALL" and state.memory.stall_count >= self.early_stop_repeat:
            repaired = False
            if self._replans < self.max_replans:
                self._replans += 1
                repaired = self._plan(state.task, state, repair=True)
                if repaired:
                    result.plan = state.memory.plan
                    state.memory.control_state = "RECOVER"
                    state.memory.stall_count = 0
                    state.no_change_count = 0
                    state.action_keys.clear()
                    state.memory.add_event("已根据实际失败证据重新规划；继续使用当前环境，不重置任务。")
            if not repaired:
                result.trajectory.append({"step": step_no, "early_stop": "持续卡死，强制停止"})
                should_break = True
                result.termination_reason = "stalled"
        # 8) 设置下一步的 expected_effect（当前决策的 expected_effect → 下一步对比用）
        state.memory.expected_effect = decision.expected_effect
        # All accepted task conditions must be backed by framework evidence.
        if state.verification.get("status") == "passed" and execution.kind != "terminal":
            result.success = True
            result.final_answer = execution.answer or "框架已逐项验证所有任务完成条件。"
            result.termination_reason = "verified"
            should_break = True
        if not should_break and execution.kind == "finish_request":
            confirmed, reason = self._verify_finish(state, execution.answer)
            if confirmed:
                result.success = True
                result.termination_reason = "verified"
                result.final_answer = execution.answer
                should_break = True
            elif getattr(self.env, "evaluation_available", False):
                # OSWorld evaluation may run postconfig; invoke it once at episode end.
                result.termination_reason = "evaluation_requested"
                result.final_answer = execution.answer
                should_break = True
            elif not self.completion.spec.get("checks"):
                result.termination_reason = "completion_unverifiable"
                result.final_answer = "执行结束，但系统尚未验证任务完成。模型答复：" + execution.answer
                should_break = True
            else:
                state.memory.add_event(f"request_finish 未通过系统验证：{reason}")
        if execution.done:
            if execution.kind == "terminal":
                result.success = False
                result.termination_reason = "agent_failed"
            elif not result.success:
                result.termination_reason = "environment_ended"
            result.final_answer = execution.answer or result.final_answer
            should_break = True
        # （reflection 已关闭：旧实现调了模型但结果被丢弃，属无效调用，浪费一次推理）

        return state, should_break

    # ================= 内部工具 ================= #
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

    def _plan(self, task: str, state: StepState, repair=False) -> bool:
        """Mandatory task decomposition, with one bounded repair attempt.

        A plan proposes actions and evidence; it cannot create or weaken the
        trusted completion contract. Only existing check IDs may be attached.
        """
        prompt = build_planner_prompt(task, self.completion.public_spec, state.task_profile)
        if repair:
            prompt += "\n当前计划受阻，请保留原目标，按以下已发生事实修订剩余步骤：\n" + json.dumps({
                "previous_plan": state.memory.plan, "recovery": state.recovery,
                "phase_events": state.phase_events, "feedback": self.completion.public_report(state.verification),
                "recent_actions": state.memory.action_history[-5:]}, ensure_ascii=False)
        messages = [{"role": "user", "content": prompt}]
        if not repair:
            self.planning_calls = []
        check_ids = {c["id"] for c in self.completion.public_spec.get("checks", [])}
        for attempt in range(2):
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
                if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
                    raise ValueError("steps 必须含 1~12 个具体步骤")
                normalized = []
                for i, step in enumerate(steps, 1):
                    if not isinstance(step, dict) or not isinstance(step.get("goal"), str) or not step["goal"].strip():
                        raise ValueError("每一步需要非空 goal")
                    if not isinstance(step.get("expected_evidence"), str) or not step["expected_evidence"].strip():
                        raise ValueError("每一步必须给出非空 expected_evidence，说明实际可以观察到的结果")
                    ids = step.get("check_ids", [])
                    if not isinstance(ids, list) or any(not isinstance(x, str) or x not in check_ids for x in ids):
                        raise ValueError("check_ids 只能引用给定的系统验收条件；没有时用 []")
                    normalized.append({"idx": i, "goal": step["goal"].strip(),
                                       "expected_evidence": str(step.get("expected_evidence") or ""),
                                       "check_ids": ids, "phase": str(step.get("phase") or ""), "status": "pending"})
                if state.task_profile.get("kind") == "paper_download":
                    phases = {step["phase"] for step in normalized}
                    required = {phase["phase"] for phase in state.task_profile["workflow"]}
                    if not required.issubset(phases):
                        raise ValueError("论文下载规划缺少流程阶段：" + ",".join(sorted(required - phases)))
                    ordered = [item["phase"] for item in state.task_profile["workflow"]]
                    indexes = [next(i for i, step in enumerate(normalized) if step["phase"] == phase) for phase in ordered]
                    if indexes != sorted(indexes):
                        raise ValueError("流程阶段顺序必须为 " + " → ".join(ordered))
                    if not isinstance(obj.get("keywords"), list) or not any(isinstance(k, str) and k.strip() for k in obj["keywords"]):
                        raise ValueError("论文搜索必须提供非空 keywords")
                state.memory.plan = {"steps": normalized,
                    "keywords": obj.get("keywords") if isinstance(obj.get("keywords"), list) else [],
                    "final_check": str(obj.get("final_check") or ""),
                    "completion_spec": copy.deepcopy(self.completion.public_spec)}
                state.memory.task_state["goal"] = task
                state.memory.task_state["current_subgoal"] = normalized[0]["goal"]
                normalized[0]["status"] = "doing"
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

    def _update_plan_progress(self, state: StepState, execution: Execution) -> None:
        """Only evidence for explicitly linked conditions can mark a step done."""
        if not state.memory.plan:
            return
        checks = {c["id"]: c.get("status") for c in state.verification.get("checks", [])}
        current = state.memory.task_state.get("current_subgoal", "")
        info = execution.info or {}
        evidence = None
        if info.get("status") == "executed" and info.get("action") == "goto":
            evidence = {"event": "page_opened", "url": info.get("url"), "step": state.step_id}
        artifacts = list(info.get("artifacts", [])) + list((state.obs.info if state.obs else {}).get("artifacts", []))
        if artifacts:
            evidence = {"event": "download_completed", "download_ids": [a.get("download_id") for a in artifacts
                        if isinstance(a, dict) and a.get("completed") is True], "step": state.step_id}
        if state.verification.get("status") == "passed":
            evidence = {"event": "all_requirements_verified", "step": state.step_id}
        if evidence and (not state.phase_events or evidence["event"] != state.phase_events[-1].get("event")
                         or evidence.get("download_ids") != state.phase_events[-1].get("download_ids")):
            state.phase_events.append(evidence)
            self._emit("phase_evidence", evidence=evidence)
        for step in state.memory.plan.get("steps", []):
            ids = step.get("check_ids", [])
            if ids and all(checks.get(i) == "passed" for i in ids):
                step["status"] = "done"
            elif step["goal"] == current:
                step["status"] = "doing"
            else:
                step["status"] = "pending"

    def _programmatic_done(self, state: StepState, execution: Execution) -> str:
        return "contract" if state.verification.get("status") == "passed" else ""

    def _build_memory_text(self, state: StepState) -> str:
        public = self.completion.public_report(state.verification)
        report = json.dumps({"status": public.get("status"),
            "checks": [{"id": c.get("id"), "status": c.get("status"), "reason": c.get("reason", "")[:120]}
                       for c in public.get("checks", [])], "reason": public.get("reason", "")}, ensure_ascii=False)
        return ("## 系统验收（模型不能修改）\n" + report + "\n\n"
                + self._program_info_text(state) + "\n\n" + state.memory.to_text(include_plan=False))

    def _program_info_text(self, state: StepState) -> str:
        coord = state.last_grounding_coord
        return "\n".join([
            "## 程序信息(program)",
            f"- step_id: {state.step_id}",
            f"- last_action: {state.last_action or '（无，这是第一步）'}",
            f"- last_target: {state.last_target or '（无）'}",
            f"- grounding_coordinate: {list(coord) if coord else '（无）'}",
            f"- execution_status: {state.execution_status or '（无）'}",
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
                               execution: Execution, before_screenshot) -> str:
        stall_reason = self._detect_stall(state, decision, execution, before_screenshot)
        if stall_reason:
            state.memory.control_state = "STALL"
            state.memory.stall_count += 1
            sig = self._stable_action_key(decision.action)
            state.memory.add_forbidden_repeat(sig)
            state.memory.add_event(f"⚠️ STALL：{stall_reason}。反复同一操作且屏幕无变化，任务可能【已经完成】——先看界面/文档是否已达标，达标就 request_finish；否则换一种完全不同的方法。")
            return "STALL"
        state.memory.stall_count = 0
        status = decision.action_status
        if status == "failure":
            state.memory.control_state = "RECOVER"
        elif status == "success":
            state.memory.control_state = "ADVANCE"
        else:
            state.memory.control_state = "CONTINUE"
        return state.memory.control_state

    @staticmethod
    def _brief_outcome(execution) -> str:
        """把执行结果精简成一句话，供动作历史使用（避免冗长 JSON 撑爆记忆预算）。"""
        info = execution.info if isinstance(execution.info, dict) else {}
        status = info.get("status")
        if status == "rejected":
            return f"失败：{str(info.get('error', ''))[:100]}"
        if status == "uncertain":
            return "结果不确定，需要重新观察并验证，不能直接重复执行"
        if info.get("action") == "download":
            if info.get("status") == "already_downloaded":
                return "已下载过（跳过）"
            return f"下载成功 {info.get('bytes', '?')}B"
        if info.get("action") == "goto":
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
        import re
        a = action.action
        target = str(action.args.get("target", "") or action.args.get("from_target", ""))
        target = re.sub(r"\(href[^)]*\)", "", target)          # 去掉 (href ...) 括号形式
        target = re.sub(r"href\s+[^\s,)]*", "href", target)    # 归一化 "href URL" → "href"
        target = re.sub(r"\[\d+\]", "", target)                # 去掉方括号数字 ref
        target = re.sub(r"\s+", " ", target).strip()
        stable = {"action": a, "target": target[:80]}
        for k in ("text", "option", "key", "url", "amount", "direction", "overwrite"):
            if k in action.args:
                stable[k] = action.args[k]
        return json.dumps(stable, ensure_ascii=False, sort_keys=True)

    def _detect_stall(self, state: StepState, decision: Decision,
                      execution: Execution, before_screenshot) -> Optional[str]:
        """确定性卡死检测：屏幕无变化 + shell 输出无变化 + 动作循环 + 禁止重复命中。返回原因或 None。"""
        # 1) 屏幕无变化（仅 env 动作，对比动作前后截图）
        if execution.kind == "env" and execution.obs is not None:
            after = execution.obs.screenshot
            if self._screen_diff(before_screenshot, after) < _NO_CHANGE_DIFF_THRESHOLD:
                state.no_change_count += 1
            else:
                state.no_change_count = 0
            if state.no_change_count >= _NO_CHANGE_CONSECUTIVE_THRESHOLD:
                return f"连续 {state.no_change_count} 步屏幕无可见变化"
        # 1b) shell 动作：命令在 VM 后台执行、不改屏幕，屏幕对比对它无效。
        #     改用「连续多次终端输出完全相同」判卡死（典型：反复执行同一失败命令）。
        if execution.kind == "shell":
            state.shell_outputs.append(state.terminal_result or "")
            recent = state.shell_outputs[-_REPEAT_ACTION_THRESHOLD:]
            if (len(recent) >= _REPEAT_ACTION_THRESHOLD
                    and all(o == recent[-1] for o in recent)):
                return f"连续 {len(recent)} 次终端命令输出完全相同（命令未产生新进展）"
        # 2) 连续相同动作（连续 N 次相同 action+target；用稳定签名，忽略每次变化的 ref/id）
        sig = self._stable_action_key(decision.action)
        state.action_keys.append(sig)
        recent = state.action_keys[-max(1, self.early_stop_repeat):]
        if len(recent) >= max(1, self.early_stop_repeat) and all(k == sig for k in recent):
            return f"连续 {self.early_stop_repeat} 次相同动作 {sig[:60]}"
        # 3) 周期循环（A,B,A,B 等，周期 2）
        keys = state.action_keys[-_LOOP_HISTORY_LEN:]
        if len(keys) >= _LOOP_HISTORY_LEN:
            for period in _LOOP_PERIODS:
                if all(keys[i] == keys[i - period] for i in range(period, len(keys))):
                    return f"动作形成 {period} 步周期循环"
        # 4) 禁止重复命中
        if sig and sig in state.memory.forbidden_repeats:
            return "重复了被禁止的策略"
        return None

    def _verify_finish(self, state: StepState, answer: str) -> Tuple[bool, str]:
        """The decision model's answer is not evidence and is never a verifier."""
        report = self.completion.evaluate()
        state.verification = report
        self._verification = copy.deepcopy(report)
        public = self.completion.public_report(report)
        missing = [f"{c.get('id')}: {c.get('reason', c.get('status'))}"
                   for c in public.get("checks", []) if c.get("status") != "passed"]
        return report.get("status") == "passed", "; ".join(missing) or public.get("reason", "缺少可验证完成条件")

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
            state.execution_status = "executed"

    def _validate_action(self, action: Action) -> Optional[str]:
        name = action.action
        if is_skill(name):
            return validate_skill(name, action.args)
        if name not in ACTION_REGISTRY:
            return (f"未知动作类型「{name}」。可选动作：{', '.join(ACTION_REGISTRY)}；"
                    f"可选技能：{', '.join(enabled_skills())}")
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
