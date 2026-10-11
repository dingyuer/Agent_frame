"""Transport lifecycle and retry limits, with no endpoint requests."""
from types import SimpleNamespace

import pytest

from osworld_agent.config import ModelConfig
from osworld_agent.model.decision_model import ChatModel


def test_reuses_client_and_disables_nested_sdk_retries(monkeypatch):
    import openai
    created, closed, requests = [], [], []
    def factory(**kwargs):
        created.append(kwargs)
        def create(**request):
            requests.append(request)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"ok":true}'), finish_reason="stop")])
        def close():
            closed.append(True)
            kwargs["http_client"].close()
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close)
    monkeypatch.setattr(openai, "OpenAI", factory)
    model = ChatModel(ModelConfig(url="http://example.invalid/v1"))
    for _ in range(2):
        assert model.chat([{"role": "user", "content": "test"}], json_mode=True) == '{"ok":true}'
    assert len(created) == 1 and created[0]["max_retries"] == 0
    assert len(requests) == 2
    assert requests[0]["response_format"] == {"type": "json_object"}
    model.close()
    assert closed == [True] and model._client is None


def test_transport_attempt_limit_is_explicit(monkeypatch):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("offline")
    model = ChatModel(ModelConfig())
    model._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("osworld_agent.model.decision_model.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="offline"):
        model.chat([{"role": "user", "content": "test"}], retries=1)
    assert len(calls) == 2 and len(model.last_call["attempts"]) == 2


def rollout_model(metadata=None, **config):
    requests = []
    choice = SimpleNamespace(message=SimpleNamespace(content='  {"ok":true}\n'),
        finish_reason="stop", token_ids=[101, 102],
        logprobs=SimpleNamespace(content=[SimpleNamespace(logprob=-0.2), SimpleNamespace(logprob=-0.4)]))
    response = SimpleNamespace(choices=[choice], prompt_token_ids=[1, 2, 3], rollout_metadata=metadata or {})
    def create(**request):
        requests.append(request)
        return response
    model = ChatModel(ModelConfig(capture_rollout=True, tokenizer_revision="tok1",
        processor_revision="processor1", chat_template_revision="template1", **config))
    model._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return model, requests


def test_capture_actual_server_tokens_and_preserve_sampled_whitespace():
    model, requests = rollout_model({"policy_revision": "checkpoint-1"}, policy_revision="checkpoint-1")
    text = model.chat([{"role": "user", "content": "task"}], retries=0)
    assert text == '  {"ok":true}\n'
    assert requests[0]["logprobs"] is True
    assert requests[0]["extra_body"]["return_token_ids"] is True
    assert model.last_call["token_ids"] == [101, 102]
    assert model.last_call["prompt_token_ids"] == [1, 2, 3]
    assert model.last_call["token_logprobs"] == [-0.2, -0.4]
    assert model.last_call["rl_ready"] is True


@pytest.mark.parametrize("metadata", [{}, {"policy_revision": "wrong-checkpoint"},
    {"policy_revision": "checkpoint-1", "token_logprobs": [float("nan"), -0.1]}])
def test_missing_or_inconsistent_rollout_evidence_is_not_rl_ready(metadata):
    import json
    model, _ = rollout_model(metadata, policy_revision="checkpoint-1")
    model.chat([{"role": "user", "content": "task"}], retries=0)
    assert not model.last_call["rl_ready"]
    assert model.last_call["rollout_validation_errors"]
    json.dumps(model.last_call, allow_nan=False)


def test_server_processor_must_match_expected_training_processor():
    model, _ = rollout_model({"policy_revision": "checkpoint-1", "processor_revision": "other"})
    model.chat([{"role": "user", "content": "task"}], retries=0)
    assert not model.last_call["rl_ready"]
    assert "processor_revision mismatch" in model.last_call["rollout_validation_errors"]


def test_current_image_encoding_failure_never_silently_becomes_text_only():
    from osworld_agent.model.decision_model import ImageEncodingError
    with pytest.raises(ImageEncodingError):
        ChatModel(ModelConfig()).build_vision_message("task", [("current", object())])
