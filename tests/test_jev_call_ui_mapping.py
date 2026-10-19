"""Archive genuine System One exchanges beside the decision they reviewed."""
import copy
import json

from fastapi.testclient import TestClient

from osworld_agent.viz.server import _local_traj_to_frontend, app


def systemone_call(call_id, *, state=None, decision_step=None):
    call = {
        "call_id": call_id,
        "actor_role": "verifier",
        "model": "d3-flash",
        "latency_ms": 234.5,
        "request": {
            "model": "d3-flash",
            "state": json.dumps(state or {"goal": "打开候选论文"}, ensure_ascii=False),
            "questions": {
                "settled": {
                    "type": "choice",
                    "instructions": "刚才的动作是否执行完成？",
                    "criteria": {"complete": "已完成", "waiting": "仍在加载"},
                }
            },
        },
        "response": {
            "status": "answered",
            "answers": {"settled": {"label": "complete", "probabilities": {"complete": .8, "waiting": .2}}},
            "raw_response": {
                "model": "deployed-flash",
                "answers": {"settled": {"choice": "complete", "probabilities": {"complete": .8, "waiting": .2}}},
                "usage": {"input_tokens": 321, "output_tokens": 0},
            },
        },
    }
    if decision_step is not None:
        call["decision_step"] = decision_step
    return call


def model_event(call, parent=None, **fields):
    event = {"event_type": "model_call", "call": call, **fields}
    if parent is not None:
        event["parent_decision_id"] = parent
    return event


def calls(rows, step):
    return next(row for row in rows if row["step_num"] == step)["jev_calls"]


def state_object(call):
    state = call["request"]["state"]
    return json.loads(state) if isinstance(state, str) else state


def test_model_events_keep_systemone_exchanges_on_their_own_decision_and_dedupe():
    first, second = systemone_call("call-1"), systemone_call("call-2")
    generative = {
        "call_id": "vlm-review",
        "actor_role": "verifier",
        "request": {"messages": [{"role": "user", "content": "Review the screenshot"}]},
        "response": {"text": "The page is ready"},
    }
    trajectory = {
        "steps": [{"step": 1}, {"step": 2}],
        "events": [model_event(first, "a_1"), model_event(generative, "a_1"),
                   model_event(second, "a_2"), model_event(copy.deepcopy(first), "a_1"),
                   {"event_type": "jev_transition", "parent_decision_id": "a_1", "call": first}],
    }
    rows = _local_traj_to_frontend(trajectory)
    assert [call["call_id"] for call in calls(rows, 1)] == ["call-1"]
    assert [call["call_id"] for call in calls(rows, 2)] == ["call-2"]
    assert calls(rows, 1)[0]["request"]["questions"] == first["request"]["questions"]
    assert calls(rows, 2)[0]["response"]["raw_response"] == second["response"]["raw_response"]


def test_only_explicit_decision_step_fallback_is_allowed_for_unparented_events():
    fallback = systemone_call("fallback", decision_step=2)
    orphan = systemone_call("orphan")
    wrong_parent = systemone_call("unknown-parent", decision_step=1)
    trajectory = {
        "steps": [{"step": 1}, {"step": 2}],
        "events": [model_event(fallback), model_event(orphan),
                   model_event(wrong_parent, "a_99")],
    }
    rows = _local_traj_to_frontend(trajectory)
    assert calls(rows, 1) == []
    assert [call["call_id"] for call in calls(rows, 2)] == ["fallback"]


def test_legacy_step_calls_are_supported_without_duplication_from_events():
    shared, legacy = systemone_call("shared"), systemone_call("legacy-only")
    trajectory = {
        "steps": [{"step": 1, "jev_calls": [shared, legacy]}],
        "events": [model_event(copy.deepcopy(shared), "a_1")],
    }
    mapped = calls(_local_traj_to_frontend(trajectory), 1)
    assert {call["call_id"] for call in mapped} == {"shared", "legacy-only"}
    assert len(mapped) == 2
    assert next(call for call in mapped if call["call_id"] == "legacy-only")["response"] == legacy["response"]


def test_secret_keys_are_removed_recursively_in_json_state_without_removing_token_usage():
    state = {
        "goal": "等待下载完成",
        "Authorization": "Bearer authorization-secret",
        "frame": {"api_key": "nested-api-secret", "input_tokens": 120,
                  "nodes": [{"name": "Download", "Password": "nested-password-secret"}]},
        "history": [{"token": "session-token-secret", "output_tokens": 0}],
    }
    call = systemone_call("sanitized-state", state=state)
    call["request"]["headers"] = {"authorization": "Bearer header-secret", "Accept": "application/json"}
    mapped = calls(_local_traj_to_frontend({"steps": [{"step": 1}],
                       "events": [model_event(call, "a_1")]}), 1)[0]
    sanitized_state = state_object(mapped)
    assert sanitized_state == {
        "goal": "等待下载完成",
        "frame": {"input_tokens": 120, "nodes": [{"name": "Download"}]},
        "history": [{"output_tokens": 0}],
    }
    assert mapped["request"]["headers"] == {"Accept": "application/json"}
    assert mapped["response"]["raw_response"]["usage"] == {"input_tokens": 321, "output_tokens": 0}
    assert "secret" not in json.dumps(mapped)


def test_images_remain_renderable_and_serialization_does_not_mutate_original_archive():
    call = systemone_call("images", state={"task": "看截图", "api_key": "private-key"})
    call["request"]["images"] = [
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
        {"url": "/local/screenshot?run=fixture&step=2", "detail": "high"},
    ]
    call["request"]["state"] = {"task": "看截图", "frames": [{"token": "private-token", "ref": "17"}]}
    trajectory = {"steps": [{"step": 2}], "events": [model_event(call, "a_2")]}
    original = copy.deepcopy(trajectory)
    mapped = calls(_local_traj_to_frontend(trajectory, prefix="fixture_"), 2)[0]
    assert mapped["request"]["images"] == original["events"][0]["call"]["request"]["images"]
    assert state_object(mapped) == {"task": "看截图", "frames": [{"ref": "17"}]}
    assert trajectory == original


def test_error_response_and_raw_typed_answers_are_preserved_except_secret_fields():
    call = systemone_call("error")
    call["response"].update({
        "status": "unknown",
        "error_type": "HTTPStatusError",
        "http_status": 422,
        "detail": [{"loc": ["images", 0], "msg": "invalid media", "api_key": "error-detail-secret"}],
        "error": {"code": "invalid_question", "message": "Unsupported criterion", "token": "error-token-secret"},
    })
    call["response"]["raw_response"]["server"] = {"version": "fixture-v1", "Password": "raw-password-secret"}
    mapped = calls(_local_traj_to_frontend({"steps": [{"step": 1}],
                       "events": [model_event(call, "a_1")]}), 1)[0]
    assert mapped["response"]["answers"] == call["response"]["answers"]
    assert mapped["response"]["raw_response"]["answers"] == call["response"]["raw_response"]["answers"]
    assert mapped["response"]["raw_response"]["server"] == {"version": "fixture-v1"}
    assert mapped["response"]["error"] == {"code": "invalid_question", "message": "Unsupported criterion"}
    assert mapped["response"]["detail"] == [{"loc": ["images", 0], "msg": "invalid media"}]
    assert mapped["response"]["http_status"] == 422
    assert mapped["response"]["error_type"] == "HTTPStatusError"
    assert "secret" not in json.dumps(mapped)


def test_default_prompt_endpoint_exposes_all_mode_prompts_and_compatible_dom_prompt(monkeypatch):
    from osworld_agent import prompts
    from osworld_agent.viz import server

    requested = []
    mode_prompts = {"dom": "DOM: click(ref)", "gui_browser": "GUI browser: click(x,y)",
                    "gui_desktop": "GUI desktop: open app"}

    def build_modes(**kwargs):
        requested.append(kwargs)
        return copy.deepcopy(mode_prompts)

    def obsolete_bootstrap(**kwargs):
        raise AssertionError("Mode preview must not display the bootstrap prompt")

    monkeypatch.setattr(prompts, "build_mode_system_prompts", build_modes)
    monkeypatch.setattr(prompts, "build_system_prompt", obsolete_bootstrap)
    monkeypatch.setattr(server.sys, "platform", "win32")
    with TestClient(app) as client:
        response = client.get("/experiments/default-prompt", params={"browser_only": "true"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["prompts"] == mode_prompts
    assert body["prompt"] == mode_prompts["dom"]
    assert requested == [{"os_name": "windows", "browser_only": True}]
