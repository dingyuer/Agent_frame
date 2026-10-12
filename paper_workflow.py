"""Bounded framework assistance for an explicit paper-download goal.

The original model decision is never rewritten. Identity resolution and file
operations are separate framework events and consume wall time, not model steps.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field

from .env import Observation

from .paper_resolver import ADAPTER_VERSION, arxiv_id_from_url, resolve_arxiv_candidates

WORKFLOW_VERSION = "paper-workflow-v1"


@dataclass
class WorkflowResult:
    observation: object
    operations: list = field(default_factory=list)
    messages: list = field(default_factory=list)


def observed_paper_urls(observation):
    """Prefer the selected paper; search pages retain all observed candidates."""
    info = getattr(observation, "info", {}) or {}
    blocks = getattr(observation, "context", []) or []
    urls = [info.get("url", "")]
    urls.extend(b.get("url", "") for b in blocks if b.get("kind") == "browser_ax")
    selected = [url for url in urls if arxiv_id_from_url(url)]
    if selected:
        return list(dict.fromkeys(selected))
    for container in [info, *blocks]:
        for candidate in container.get("download_candidates", []):
            if isinstance(candidate, dict):
                urls.append(candidate.get("url", ""))
        for node in container.get("nodes", []):
            if isinstance(node, dict):
                urls.append(node.get("href", ""))
    return list(dict.fromkeys(url for url in urls if arxiv_id_from_url(url)))


def binding_group_hash(bindings):
    """Stable evaluation identity, excluding timestamps and random artifact IDs."""
    semantic = {key: {field: value.get(field) for field in
                      ("query", "canonical_id", "full_title", "authors", "dois", "publication_years", "source_url", "pdf_url", "adapter_version")}
                for key, value in sorted(bindings.items())}
    return hashlib.sha256(json.dumps({"adapter": ADAPTER_VERSION, "bindings": semantic},
                                   sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class PaperWorkflow:
    def __init__(self, *, resolver=None, max_resolution_rounds=8):
        self.resolver = resolver or resolve_arxiv_candidates
        self.max_resolution_rounds = max_resolution_rounds
        self._resolutions = set()
        self._downloads = set()

    def advance(self, observation, completion, env, emit, *, remaining_s=None, allow_download=True):
        result = WorkflowResult(observation)
        session = getattr(env, "_browser_session", None)
        if observation is None or session is None or not callable(getattr(session, "download", None)):
            return result
        started = time.monotonic()

        def remaining():
            return float("inf") if remaining_s is None else remaining_s - (time.monotonic() - started)

        def record(event, **payload):
            payload.update(actor_role="framework", source="framework", policy_trainable=False)
            emit(event, **payload)

        urls = observed_paper_urls(observation)
        for goal in completion.pending_paper_goals():
            key = (goal["check_id"], tuple(urls))
            if (not urls or key in self._resolutions or len(self._resolutions) >= self.max_resolution_rounds
                    or remaining() <= 0):
                continue
            self._resolutions.add(key)
            resolution = self.resolver(goal["query"], urls, timeout=min(5.0, remaining()),
                                       total_timeout=min(12.0, remaining()))
            record("paper_resolution", check_id=goal["check_id"], observed_urls=urls,
                   status=resolution.status, reason=resolution.reason)
            if resolution.status != "resolved" or resolution.binding is None:
                result.messages.append("论文身份尚未确认：" + resolution.reason)
                continue
            try:
                completion.bind_paper(goal["check_id"], resolution.binding)
            except ValueError as exc:
                record("paper_binding_rejected", check_id=goal["check_id"], reason=str(exc))
                result.messages.append("候选论文不满足任务约束：" + str(exc))
                continue
            record("goal_binding", check_id=goal["check_id"], binding=resolution.binding.to_dict())
            result.messages.append("框架已根据官方证据确认论文：" + resolution.binding.full_title)

        completion.observe(observation)
        if not allow_download:
            return result
        metadata = session.event_metadata()
        if metadata.get("dialogs") or metadata.get("dialog"):
            return result
        reports = {c["id"]: c for c in completion.public_report(completion.evaluate()).get("checks", [])}
        for goal in completion.paper_goals():
            binding = goal.get("binding")
            if not binding or reports.get(goal["check_id"], {}).get("status") == "passed":
                continue
            if goal.get("count", 1) != 1 or remaining() <= 0:
                continue
            url = binding["pdf_url"]
            key = (goal["check_id"], url, goal.get("filename"))
            if key in self._downloads:
                continue
            metadata = session.event_metadata()
            pending = metadata.get("pending_downloads", []) or metadata.get("download_events", [])
            if any(p.get("status") == "in_progress" and str(p.get("url", "")).split("#", 1)[0] == url.split("#", 1)[0]
                   for p in pending):
                continue
            self._downloads.add(key)
            args = {"url": url}
            if goal.get("filename"):
                args["filename"] = goal["filename"]
            record("framework_operation_started", operation="download", check_id=goal["check_id"], args=args)
            operation_started = time.monotonic()
            options = getattr(session, "options", None)
            previous_timeout = getattr(options, "download_timeout_s", None)
            if previous_timeout is not None:
                options.download_timeout_s = min(previous_timeout, max(0.01, remaining()))
            try:
                info = session.download(**args)
            except Exception as exc:
                info = {"status": "uncertain", "error": str(exc), "action": "download",
                        "reason": "下载调用异常，派发与完成状态需要重新观察确认"}
            finally:
                if previous_timeout is not None:
                    options.download_timeout_s = previous_timeout
            if info.get("status") == "in_progress":
                # A raced download event is not a failed attempt. After settlement
                # allow the completed cache to apply the requested filename.
                self._downloads.discard(key)
            operation = {"operation": "download", "check_id": goal["check_id"], "args": args,
                         "result": copy.deepcopy(info), "actor_role": "framework", "policy_trainable": False,
                         "elapsed_s": round(time.monotonic() - operation_started, 3)}
            result.operations.append(operation)
            record("framework_operation_finished", **operation)
            completion.observe(None, info)
            result.messages.append("框架下载结果：" + str(info.get("status", "unknown")) +
                                   ("；" + str(info["error"]) if info.get("error") else ""))
        if result.operations:
            try:
                result.observation = env.observe()
            except Exception:
                # Never leave stale references/coordinates available to execution.
                result.observation = Observation(screenshot=None, context=[],
                    info={"observation_error": "Refresh after framework operation failed",
                          "screenshot_stale": True, **session.event_metadata()})
            completion.observe(result.observation)
        return result
