"""Semantic selector validation; no remote model requests."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from osworld_agent.paper_identity import PaperIdentitySelector
from osworld_agent.paper_resolver import parse_arxiv_metadata


ID = "2502.08047"
TITLE = "WorldGUI: An Interactive Benchmark for Desktop GUI Automation"


def candidate():
    return parse_arxiv_metadata("https://arxiv.org/abs/" + ID,
        '<meta name="citation_title" content="' + TITLE + '">'
        '<meta name="citation_author" content="Example Author">'
        '<meta name="citation_pdf_url" content="https://arxiv.org/pdf/' + ID + '">')


def answer(**changes):
    obj = {"status": "matched", "selected_id": ID, "reason": "简称和候选完整标题对应",
           "evidence": [{"candidate_id": ID, "field": "full_title", "quote": TITLE}]}
    obj.update(changes)
    return obj


class Model:
    def __init__(self, reply):
        self.reply = reply
        self.cfg = SimpleNamespace(timeout=99.0, max_tokens=2048, temperature=.7)
        self.requests = []
        self.last_call = {}

    def chat(self, messages, **kwargs):
        self.requests.append((messages, kwargs, vars(self.cfg).copy()))
        if isinstance(self.reply, Exception):
            raise self.reply
        return json.dumps(self.reply)


def test_model_can_resolve_a_paraphrase_without_literal_title_matching():
    model = Model(answer())
    selector = PaperIdentitySelector(model)
    result = selector.select("桌面 GUI 任意起点的 WorldGUI 基准论文", [candidate()], remaining_s=2)
    assert result["status"] == "matched" and result["selected_id"] == ID
    assert model.requests[0][1] == {"retries": 0, "json_mode": True}
    assert model.requests[0][2] == {"timeout": 2.0, "max_tokens": 768, "temperature": 0.0}
    assert vars(model.cfg) == {"timeout": 99.0, "max_tokens": 2048, "temperature": .7}
    assert selector.calls[0]["selection_result"] == result
    assert selector.calls[0]["actor_role"] == "verifier"


@pytest.mark.parametrize("reply", [
    answer(selected_id="1706.03762"),
    answer(evidence=[]),
    answer(evidence=[{"candidate_id": ID, "field": "authors", "quote": "Vaswani"}]),
    answer(evidence=[{"candidate_id": ID, "field": "source_url", "quote": "https://arxiv.org"}]),
    answer(status="different", selected_id=ID),
])
def test_invented_candidate_or_unsupported_citation_cannot_authorize_binding(reply):
    assert PaperIdentitySelector(Model(reply)).select("WorldGUI", [candidate()])["status"] == "uncertain"


def test_unattested_metadata_never_reaches_model():
    model = Model(answer())
    result = PaperIdentitySelector(model).select("WorldGUI", [replace(candidate(), _proof=None)])
    assert result["status"] == "uncertain" and not model.requests


def test_explicit_constraints_are_supplied_without_invented_hard_constraints():
    model = Model(answer())
    PaperIdentitySelector(model).select("WorldGUI", [candidate()], task="下载作者 Example Author 的论文",
        constraints={"author": "Example Author", "year": 2025, "guessed_id": "not evidence"})
    payload = json.loads(model.requests[0][0][1]["content"])
    assert payload["constraints"] == {"author": "Example Author", "year": 2025}
    assert payload["task"] == "下载作者 Example Author 的论文"


def test_no_budget_or_no_model_stays_uncertain_without_requests():
    model = Model(answer())
    assert PaperIdentitySelector(model).select("WorldGUI", [candidate()], remaining_s=0)["status"] == "uncertain"
    assert not model.requests
    assert PaperIdentitySelector().select("WorldGUI", [candidate()])["status"] == "uncertain"


def test_model_error_is_audited_without_exposing_exception_message():
    selector = PaperIdentitySelector(Model(RuntimeError("private transport data")))
    result = selector.select("WorldGUI", [candidate()])
    assert result["status"] == "uncertain"
    assert "private transport data" not in str(result)
    assert selector.calls[0]["selection_result"]["error_type"] == "RuntimeError"
