"""Read-only paper identity binding. All downloads originate from UI interaction."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from .paper_resolver import ADAPTER_VERSION, arxiv_id_from_url, resolve_arxiv_candidates

WORKFLOW_VERSION = "paper-workflow-v2-ui-only"


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

    def advance(self, observation, completion, env, emit, *, remaining_s=None, allow_download=False):
        result = WorkflowResult(observation)
        session = getattr(env, "_browser_session", None)
        if observation is None or session is None:
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
            pdf = resolution.binding.pdf_url
            filename = goal.get("filename", "")
            result.messages.append(
                "框架已确认论文身份：" + resolution.binding.full_title
                + "。请打开 PDF 页面 " + pdf
                + "，在浏览器里点击「下载 / Download」按钮保存 PDF"
                + (("，交付文件名应为 " + filename) if filename else "")
                + "；点击后浏览器会自动下载，框架会等待并核验文件名与内容。")

        completion.observe(observation)
        return result
