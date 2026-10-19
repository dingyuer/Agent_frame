"""Model intent selection is separate from deterministic citation provenance.

These are protocol/safety tests, not evidence of real model accuracy. Every
candidate is parsed through the same official citation adapter used at runtime.
"""
import json
from html import escape

import pytest

from osworld_agent.paper_identity import PaperIdentitySelector
from osworld_agent.paper_resolver import parse_arxiv_metadata, query_matches_title


WORLD_ID = "2502.08047"
WORLD_TITLE = ("WorldGUI: An Interactive Benchmark for Desktop GUI Automation "
               "from Any Starting Point")
ATTENTION_ID = "1706.03762"
ATTENTION_TITLE = "Attention Is All You Need"


def metadata(identifier=WORLD_ID, title=WORLD_TITLE, abstract=None):
    abstract = abstract or "An interactive benchmark for desktop GUI automation."
    html = (f'<meta name="citation_title" content="{escape(title, quote=True)}">'
            '<meta name="citation_author" content="Example Researcher">'
            f'<meta name="citation_arxiv_id" content="{identifier}">'
            f'<meta name="citation_pdf_url" content="https://arxiv.org/pdf/{identifier}">'
            '<meta name="citation_doi" content="10.1234/example">'
            '<meta name="citation_date" content="2025/02/11">'
            f'<meta name="description" content="{escape(abstract, quote=True)}">')
    return parse_arxiv_metadata("https://arxiv.org/abs/" + identifier, html)


def decision(status="matched", identifier=WORLD_ID, *, field="full_title", quote=WORLD_TITLE):
    return {"status": status, "selected_id": identifier,
            "reason": "The supplied metadata supports the user's intended paper.",
            "evidence": [{"candidate_id": identifier, "field": field, "quote": quote}]}


class RecordingModel:
    """Exercise ChatModel's string JSON boundary without any provider request."""
    vision = False

    def __init__(self, answer=None, error=None):
        self.answer = answer if answer is not None else decision()
        self.error = error
        self.calls = []
        self.last_call = {}

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        self.last_call = {"messages": messages, "response": self.answer}
        if self.error:
            raise self.error
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


@pytest.mark.parametrize("query, candidate, answer", [
    # A partial name need not coincide with the complete title or colon prefix.
    ("GUI Automation from Any Starting Point", metadata(), decision()),
    # A translation cannot pass the old normalized string equality check.
    ("注意力就是你所需要的一切", metadata(ATTENTION_ID, ATTENTION_TITLE),
     decision(identifier=ATTENTION_ID, quote=ATTENTION_TITLE)),
    # Descriptive user language may be grounded in the observed abstract.
    ("从任意起点开始的桌面交互基准论文", metadata(),
     decision(field="abstract", quote="interactive benchmark for desktop GUI automation")),
])
def test_semantic_selection_accepts_grounded_nonliteral_names(query, candidate, answer):
    assert not query_matches_title(query, candidate.full_title)
    model = RecordingModel(answer)
    result = PaperIdentitySelector(model).select(query, [candidate], task="搜索并下载这篇论文")
    assert result["status"] == "matched"
    assert result["selected_id"] == candidate.canonical_id
    assert result["evidence"]
    assert len(model.calls) == 1
    prompt = json.dumps(model.calls[0]["messages"], ensure_ascii=False)
    assert query in prompt
    assert candidate.full_title in prompt
    assert candidate.abstract in prompt


def test_model_can_reject_a_lexically_similar_but_different_paper():
    title = "Attention Is All You Need (to Avoid Spurious Oscillations)"
    candidate = metadata("2609.13531", title,
                         "We study spurious oscillations in numerical fluid simulation.")
    answer = decision("different", candidate.canonical_id,
                      field="abstract", quote="numerical fluid simulation")
    answer["selected_id"] = None  # No candidate is selected for a negative answer.
    model = RecordingModel(answer)
    result = PaperIdentitySelector(model).select("Attention is all you need", [candidate])
    assert result["status"] == "different"
    assert result["evidence"]


def test_identical_title_still_requires_explicit_model_intent_selection():
    model = RecordingModel({"status": "uncertain", "selected_id": None, "reason": "Ambiguous request", "evidence": []})
    result = PaperIdentitySelector(model).select(WORLD_TITLE, [metadata()])
    assert result["status"] == "uncertain"
    assert len(model.calls) == 1


def test_unobserved_selected_id_cannot_create_a_binding():
    answer = decision(identifier="9999.99999", quote=WORLD_TITLE)
    result = PaperIdentitySelector(RecordingModel(answer)).select("WorldGUI", [metadata()])
    assert result["status"] == "uncertain"
    assert result.get("selected_id") != "9999.99999"


@pytest.mark.parametrize("evidence", [
    [],
    [{"candidate_id": WORLD_ID, "field": "full_title", "quote": "A fabricated title not in metadata"}],
    [{"candidate_id": "9999.99999", "field": "full_title", "quote": WORLD_TITLE}],
    [{"candidate_id": WORLD_ID, "field": "not_a_metadata_field", "quote": WORLD_TITLE}],
    [{"candidate_id": WORLD_ID, "field": "full_title", "quote": ""}],
    [{"candidate_id": WORLD_ID, "field": "full_title", "quote": 123}],
    ["a free-form unsupported evidence claim"],
])
def test_matched_without_verifiable_input_citations_is_uncertain(evidence):
    answer = decision()
    answer["evidence"] = evidence
    result = PaperIdentitySelector(RecordingModel(answer)).select("WorldGUI", [metadata()])
    assert result["status"] == "uncertain"


def test_evidence_for_a_different_candidate_cannot_support_selected_paper():
    answer = decision()
    answer["evidence"] = [{"candidate_id": ATTENTION_ID, "field": "full_title", "quote": ATTENTION_TITLE}]
    result = PaperIdentitySelector(RecordingModel(answer)).select(
        "WorldGUI", [metadata(), metadata(ATTENTION_ID, ATTENTION_TITLE)])
    assert result["status"] == "uncertain"


@pytest.mark.parametrize("field,quote", [
    ("full_title", "An Interactive Benchmark"),
    ("abstract", "desktop GUI automation"),
    ("authors", "Example Researcher"),
    ("dois", "10.1234/example"),
    ("publication_years", "2025"),
])
def test_supported_metadata_fields_can_supply_exact_citations(field, quote):
    result = PaperIdentitySelector(RecordingModel(decision(field=field, quote=quote))).select("WorldGUI", [metadata()])
    assert result["status"] == "matched"


def test_no_model_returns_uncertain_even_for_exact_title():
    result = PaperIdentitySelector(None).select(WORLD_TITLE, [metadata()])
    assert result["status"] == "uncertain"


@pytest.mark.parametrize("answer", [
    "not JSON",
    "[]",
    {"status": "verified", "selected_id": WORLD_ID, "evidence": decision()["evidence"]},
    {"selected_id": WORLD_ID, "evidence": decision()["evidence"]},
    {"status": "matched", "selected_id": [WORLD_ID], "evidence": decision()["evidence"]},
])
def test_malformed_or_unknown_model_output_is_uncertain(answer):
    result = PaperIdentitySelector(RecordingModel(answer)).select("WorldGUI", [metadata()])
    assert result["status"] == "uncertain"


def test_model_exception_is_uncertain_not_a_download_permit():
    model = RecordingModel(error=TimeoutError("provider unavailable"))
    result = PaperIdentitySelector(model).select("WorldGUI", [metadata()])
    assert result["status"] == "uncertain"
    assert len(model.calls) == 1


@pytest.mark.parametrize("remaining_s", [0, -1])
def test_exhausted_operation_budget_never_calls_the_model(remaining_s):
    model = RecordingModel()
    result = PaperIdentitySelector(model).select("WorldGUI", [metadata()], remaining_s=remaining_s)
    assert result["status"] == "uncertain"
    assert model.calls == []


def test_no_candidates_does_not_invoke_semantic_guessing():
    model = RecordingModel()
    result = PaperIdentitySelector(model).select("WorldGUI", [])
    assert result["status"] == "uncertain"
    assert model.calls == []


def test_explicit_user_constraints_reach_the_model_with_actual_metadata():
    constraints = {"author": "Example Researcher", "year": 2025}
    model = RecordingModel()
    result = PaperIdentitySelector(model).select("WorldGUI", [metadata()],
        task="下载2025年的WorldGUI论文", constraints=constraints)
    assert result["status"] == "matched"
    prompt = json.dumps(model.calls[0]["messages"], ensure_ascii=False)
    assert "Example Researcher" in prompt and "2025" in prompt
    assert "下载2025年的WorldGUI论文" in prompt
    payload = json.loads(model.calls[0]["messages"][-1]["content"])
    assert payload["constraints"] == constraints
