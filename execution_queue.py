"""Framework-owned sequential plan. Policy memory is never completion evidence."""
from __future__ import annotations

import copy
from urllib.parse import urlsplit

# Resource guard for pathological model output, not a requested plan length.
MAX_PLAN_STEPS = 32


def normalize_verification(value):
    """Normalize unambiguous planner aliases without weakening predicates.

    Scalar predicates may use ``value`` for their one operand. Element states
    have multiple operands, so guessing what ``value`` means is not allowed.
    The original sampled plan stays untouched for training and diagnosis.
    """
    if value is None:
        return [{"kind": "visual"}]
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("verification 必须含 1~8 项核查条件")
    allowed = {"visual", "url_equals", "url_host", "text_present", "app_open", "element_state"}
    required = {"url_equals": "url", "url_host": "host", "text_present": "text", "app_open": "app", "element_state": "name"}
    result = []
    for item in value:
        if not isinstance(item, dict) or item.get("kind") not in allowed:
            raise ValueError("未知的步骤 verification.kind")
        kind = item["kind"]
        key = required.get(kind)
        if kind in {"url_equals", "url_host", "text_present", "app_open"} and "value" in item:
            alias = item["value"]
            if not isinstance(alias, str) or not alias.strip() or len(alias) > 1000:
                raise ValueError(f"{kind}.value 别名需要非空字符串，对应规范字段 {key}")
            if key in item and (not isinstance(item[key], str) or item[key].strip() != alias.strip()):
                raise ValueError(f"{kind} 的 {key} 与 value 冲突，请只保留表达原意的 {key} 字段")
            item = {**item, key: alias}
        elif kind == "element_state" and "value" in item:
            raise ValueError("element_state 不支持通用 value；请明确 name 和 states，文本输入值不属于 states")
        if key and (not isinstance(item.get(key), str) or not item[key].strip() or len(item[key]) > 1000):
            raise ValueError(f"{kind} 需要非空 {key}")
        clean = {"kind": kind}
        if key:
            clean[key] = item[key].strip()
        if kind == "url_equals" and urlsplit(clean["url"]).scheme not in {"https", "http"}:
            raise ValueError("url_equals 需要完整 http(s) URL")
        if kind == "url_host" and ("/" in clean["host"] or ":" in clean["host"] or " " in clean["host"]):
            raise ValueError("url_host.host 只填写域名")
        if kind == "element_state":
            states = item.get("states")
            if (not isinstance(states, dict) or not states or
                    any(k not in {"checked", "selected", "expanded", "disabled", "focused"} or type(v) is not bool
                        for k, v in states.items())):
                raise ValueError("element_state.states 需要明确的布尔状态")
            clean["states"] = copy.deepcopy(states)
            if item.get("role"):
                clean["role"] = str(item["role"])[:80]
        result.append(clean)
    return result


class ExecutionQueue:
    def __init__(self, plan, revision=0):
        self.plan = plan
        self.revision = revision
        self.cursor = 0
        for i, step in enumerate(plan["steps"]):
            step.update(id=f"p{revision}_{i + 1}", status="doing" if i == 0 else "pending")

    @property
    def current(self):
        steps = self.plan["steps"]
        return steps[self.cursor] if self.cursor < len(steps) else None

    def accept_review(self, report):
        step = self.current
        if step is None or report.get("plan_step_id") != step["id"]:
            return False
        step["last_review"] = copy.deepcopy(report)
        if report.get("status") != "passed":
            return False
        step["status"] = "done"
        self.cursor += 1
        if self.current:
            self.current["status"] = "doing"
        return True

    def window(self):
        steps = self.plan["steps"]
        def public(step):
            return {k: copy.deepcopy(step[k]) for k in (
                "id", "idx", "phase", "goal", "expected_evidence", "check_ids", "verification", "actions", "status") if k in step}
        return {"revision": self.revision, "cursor": self.cursor, "total": len(steps),
                "current": public(self.current) if self.current else None,
                "previous": public(steps[self.cursor - 1]) if self.cursor else None,
                "next": public(steps[self.cursor + 1]) if self.cursor + 1 < len(steps) else None,
                "all_steps_reviewed": self.current is None}

    def sync_memory(self, memory):
        memory.task_state["current_subgoal"] = self.current["goal"] if self.current else "核查整个任务的全部要求"
        memory.task_state["completed"] = [s["goal"] for s in self.plan["steps"] if s["status"] == "done"]
        memory.task_state["pending"] = [s["goal"] for s in self.plan["steps"][self.cursor:]]

    def snapshot(self):
        return {"revision": self.revision, "cursor": self.cursor, "total": len(self.plan["steps"]),
                "steps": [{k: copy.deepcopy(s[k]) for k in ("id", "idx", "goal", "status")} for s in self.plan["steps"]]}
