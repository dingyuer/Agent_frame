"""Framework paper assistance is bounded and keeps policy provenance intact."""
from functools import partial
from types import SimpleNamespace

from osworld_agent.paper_resolver import resolve_arxiv_candidates
from osworld_agent.paper_workflow import PaperWorkflow, binding_group_hash, observed_paper_urls
from osworld_agent.verifiers import CompletionVerifier
from osworld_agent.actions import Action, validate_action
from osworld_agent.rl.rollout import _step_semantic
from osworld_agent.rl.credit import build_tree
from osworld_agent.trajectory import iter_sft_records
from osworld_agent.trajectory import build_trajectory
from osworld_agent.pipeline import Pipeline
from osworld_agent.browser_feedback import paper_download_preflight_error
from osworld_agent.env import Observation

from test_pipeline_completion import observation
from test_pipeline_completion import Model, Env
from test_pipeline_training import paper_plan
from test_paper_completion import pdf_file
from test_paper_resolver import fixture_selection
from test_training_collection import trajectory, credit_rollout

import pytest
import json


TITLE = "RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks"
URL = "https://arxiv.org/pdf/2603.11558"


def metadata_fetch(url, **kwargs):
    identifier = url.rsplit("/", 1)[-1]
    return {"url": url, "text": f'''<meta name="citation_title" content="{TITLE}">
        <meta name="citation_author" content="Alice Smith">
        <meta name="citation_pdf_url" content="https://arxiv.org/pdf/{identifier}">'''}


class Session:
    def __init__(self, tmp_path):
        self.directory = tmp_path
        self.calls = []
        self.artifacts = []
        self.options = SimpleNamespace(download_timeout_s=300)

    def event_metadata(self):
        return {"artifacts": self.artifacts, "pending_downloads": []}

    def download(self, url, filename=None):
        self.calls.append((url, filename))
        path = pdf_file(self.directory / (filename or "download.pdf"), TITLE, "Alice Smith", "Abstract")
        artifact = {"artifact_id": "stable-id", "download_id": "stable-id", "status": "completed",
                    "completed": True, "path": str(path), "url": url, "final_url": url}
        self.artifacts.append(artifact)
        return {"status": "executed", "action": "download", "artifacts": [artifact]}


def environment(tmp_path, url="https://arxiv.org/abs/2603.11558"):
    session = Session(tmp_path)
    env = SimpleNamespace(_browser_session=session)
    def observe():
        obs = observation(url)
        obs.info.update(session.event_metadata())
        return obs
    env.observe = observe
    return env


def test_short_query_and_filename_bind_identity_without_downloading(tmp_path):
    env = environment(tmp_path)
    verifier = CompletionVerifier("请下载一篇叫做Roboclaw的论文，并重命名为Roboclaw")
    workflow = PaperWorkflow(resolver=partial(resolve_arxiv_candidates, fetcher=metadata_fetch, identity_selector=fixture_selection))
    events = []
    result = workflow.advance(env.observe(), verifier, env, lambda name, **p: events.append((name, p)))
    assert not env._browser_session.calls
    assert verifier.bindings and verifier.evaluate()["status"] != "passed"
    assert not result.operations and result.messages
    assert all(e[1]["policy_trainable"] is False for e in events)
    assert "goal_binding" in [e[0] for e in events]
    workflow.advance(result.observation, verifier, env, lambda *a, **k: None)
    assert not env._browser_session.calls


def test_selected_paper_does_not_mix_in_bibliography_links():
    obs = observation("https://arxiv.org/abs/2603.11558")
    obs.info["download_candidates"] = [{"url": "https://arxiv.org/pdf/2603.99999"}]
    assert observed_paper_urls(obs) == ["https://arxiv.org/abs/2603.11558"]


def test_search_result_candidates_are_not_removed_by_literal_title_matching():
    from test_browser_result_cards import result_block
    block = result_block(exact=True)
    obs = Observation(context=[block], info={'url': block['url']})
    urls = observed_paper_urls(obs, 'Attention Is All You Need')
    assert len(urls) == 50 and 'https://arxiv.org/abs/2601.00049' in urls
    assert observed_paper_urls(obs, 'A Title Not Present') == urls


def test_paper_download_requires_verified_identity_before_dispatch():
    pdf = 'https://arxiv.org/pdf/2603.11558'
    other = 'https://arxiv.org/pdf/2603.99999'
    block = {'kind': 'browser_ax', 'url': 'https://arxiv.org/search/', 'nodes': [
        {'ref': '1', 'role': 'link', 'name': 'pdf', 'href': pdf, 'direct': True},
        {'ref': '2', 'role': 'link', 'name': 'A candidate paper',
         'href': 'https://arxiv.org/abs/2603.99999', 'direct': True}]}
    obs = Observation(context=[block], info={'url': block['url']})
    class Goals:
        def __init__(self, binding=None): self.binding = binding
        def paper_goals(self): return [{'query': 'RoboClaw', 'binding': self.binding}]
    pending = Goals()
    assert paper_download_preflight_error(Action('click', {'target': 'pdf', 'target_ref': '1'}),
                                          obs, pending, {'target_ref': '1'})
    assert not paper_download_preflight_error(Action('click', {'target': 'A candidate paper', 'target_ref': '2'}),
                                              obs, pending, {'target_ref': '2'})
    assert paper_download_preflight_error(Action('goto', {'url': pdf}), obs, pending)
    bound = Goals({'canonical_id': '2603.11558', 'pdf_url': pdf})
    assert not paper_download_preflight_error(Action('goto', {'url': pdf}), obs, bound)
    assert paper_download_preflight_error(Action('goto', {'url': other}), obs, bound)
    viewer = Observation(info={'url': pdf, 'document_kind': 'pdf'})
    assert not paper_download_preflight_error(Action('hotkey', {'keys': ['command', 's']}), viewer, bound)


def test_current_official_page_metadata_binds_without_second_network_fetch(tmp_path):
    env = environment(tmp_path)
    obs = env.observe()
    obs.info['url'] = 'https://arxiv.org/abs/2603.11558'
    obs.info['paper_metadata'] = {
        'source': 'observed_arxiv_meta', 'source_url': obs.info['url'],
        'canonical_id': '2603.11558', 'title': TITLE, 'authors': ['Alice Smith'],
        'pdf_url': URL}
    verifier = CompletionVerifier('请下载一篇叫做Roboclaw的论文')
    workflow = PaperWorkflow(resolver=partial(resolve_arxiv_candidates,
        fetcher=lambda *_args, **_kw: pytest.fail('current page metadata should avoid fetch'),
        identity_selector=fixture_selection))
    workflow.advance(obs, verifier, env, lambda *a, **k: None)
    assert verifier.bindings and not env._browser_session.calls


def test_ambiguous_search_and_exhausted_time_never_download(tmp_path):
    env = environment(tmp_path, "https://arxiv.org/search/")
    obs = env.observe()
    obs.info["download_candidates"] = [{"url": URL}, {"url": "https://arxiv.org/pdf/2603.99999"}]
    verifier = CompletionVerifier("请下载一篇叫做Roboclaw的论文")
    workflow = PaperWorkflow(resolver=partial(resolve_arxiv_candidates, fetcher=metadata_fetch, identity_selector=fixture_selection))
    workflow.advance(obs, verifier, env, lambda *a, **k: None)
    assert not verifier.bindings and not env._browser_session.calls
    env = environment(tmp_path)
    workflow.advance(env.observe(), verifier, env, lambda *a, **k: None, remaining_s=0)
    assert not env._browser_session.calls


def test_unavailable_resolution_is_not_retried_each_decision(tmp_path):
    calls = []
    def offline(*args, **kwargs):
        calls.append(args)
        raise OSError("offline")
    env = environment(tmp_path)
    verifier = CompletionVerifier("请下载一篇叫做Roboclaw的论文")
    workflow = PaperWorkflow(resolver=partial(resolve_arxiv_candidates, fetcher=offline, identity_selector=fixture_selection))
    for _ in range(3):
        workflow.advance(env.observe(), verifier, env, lambda *a, **k: None)
    assert len(calls) == 1 and not env._browser_session.calls


def test_same_url_new_trusted_metadata_can_recover_failed_resolution(tmp_path):
    env = environment(tmp_path)
    calls = []
    def offline(*args, **kwargs):
        calls.append(args)
        raise OSError('offline')
    verifier = CompletionVerifier('请下载一篇叫做Roboclaw的论文')
    workflow = PaperWorkflow(resolver=partial(resolve_arxiv_candidates, fetcher=offline, identity_selector=fixture_selection))
    first = env.observe()
    workflow.advance(first, verifier, env, lambda *a, **k: None)
    assert not verifier.bindings and len(calls) == 1
    first.info.update(url='https://arxiv.org/abs/2603.11558', captured_at=10,
        paper_metadata={'source': 'observed_arxiv_meta', 'source_url': 'https://arxiv.org/abs/2603.11558',
                        'canonical_id': '2603.11558', 'title': TITLE, 'authors': ['Alice Smith'], 'pdf_url': URL})
    workflow.advance(first, verifier, env, lambda *a, **k: None)
    assert verifier.bindings and len(calls) == 1
    assert workflow._resolution_rounds == 2 and not env._browser_session.calls
    first.info['captured_at'] = 11
    workflow.advance(first, verifier, env, lambda *a, **k: None)
    assert workflow._resolution_rounds == 2


def test_transient_retry_has_cooldown_and_per_evidence_and_total_limits(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr('osworld_agent.paper_workflow.time.monotonic', lambda: now[0])
    calls = []
    def offline(*args, **kwargs):
        calls.append(args)
        raise OSError('offline')
    env = environment(tmp_path)
    verifier = CompletionVerifier('请下载一篇叫做Roboclaw的论文')
    workflow = PaperWorkflow(resolver=partial(resolve_arxiv_candidates, fetcher=offline, identity_selector=fixture_selection), retry_delay_s=2,
                             max_transient_retries=1, max_resolution_rounds=3)
    obs = env.observe()
    for _ in range(5):
        workflow.advance(obs, verifier, env, lambda *a, **k: None)
    assert len(calls) == 1
    now[0] = 3
    workflow.advance(obs, verifier, env, lambda *a, **k: None)
    now[0] = 10
    workflow.advance(obs, verifier, env, lambda *a, **k: None)
    assert len(calls) == 2
    for identifier in ('2603.99998', '2603.99999'):
        changed = observation('https://arxiv.org/abs/' + identifier)
        workflow.advance(changed, verifier, env, lambda *a, **k: None)
    assert len(calls) == 3 and workflow._resolution_rounds == 3
    assert not verifier.bindings and not env._browser_session.calls


def test_confirmed_title_mismatch_without_new_evidence_is_not_retried(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr('osworld_agent.paper_workflow.time.monotonic', lambda: now[0])
    env = environment(tmp_path)
    calls = []
    def unrelated(url, **kwargs):
        calls.append(url)
        return {'url': url, 'text': metadata_fetch(url)['text'].replace(TITLE, 'A Different Paper')}
    verifier = CompletionVerifier('请下载一篇叫做Roboclaw的论文')
    workflow = PaperWorkflow(resolver=partial(resolve_arxiv_candidates, fetcher=unrelated, identity_selector=fixture_selection))
    for clock in (0, 3, 30):
        now[0] = clock
        workflow.advance(env.observe(), verifier, env, lambda *a, **k: None)
    assert len(calls) == 1 and not verifier.bindings and not env._browser_session.calls


def test_new_capture_timestamp_does_not_count_as_new_metadata_semantics(tmp_path):
    env = environment(tmp_path)
    verifier = CompletionVerifier('请下载一篇叫做Roboclaw的论文')
    obs = env.observe()
    obs.info.update(url='https://arxiv.org/abs/2603.11558', captured_at=10,
        paper_metadata={'source': 'observed_arxiv_meta', 'source_url': 'https://arxiv.org/abs/2603.11558',
                        'canonical_id': '2603.11558', 'title': 'A Different Paper',
                        'authors': ['Alice Smith'], 'pdf_url': URL})
    calls = []
    def resolver(*args, **kwargs):
        calls.append(kwargs['metadata'])
        return resolve_arxiv_candidates(*args, **kwargs,
            fetcher=lambda *_a, **_k: pytest.fail('metadata must be used'))
    workflow = PaperWorkflow(resolver=resolver)
    workflow.advance(obs, verifier, env, lambda *a, **k: None)
    obs.info['captured_at'] = 20
    workflow.advance(obs, verifier, env, lambda *a, **k: None)
    assert len(calls) == workflow._resolution_rounds == 1
    assert not verifier.bindings and not env._browser_session.calls


def test_pending_or_uncertain_execution_never_triggers_another_transfer(tmp_path):
    env = environment(tmp_path)
    verifier = CompletionVerifier("请下载一篇叫做Roboclaw的论文")
    workflow = PaperWorkflow(resolver=partial(resolve_arxiv_candidates, fetcher=metadata_fetch, identity_selector=fixture_selection))
    workflow.advance(env.observe(), verifier, env, lambda *a, **k: None, allow_download=False)
    assert verifier.bindings and not env._browser_session.calls
    env._browser_session.event_metadata = lambda: {"pending_downloads": [
        {"download_id": "in-flight", "url": URL, "status": "in_progress"}]}
    workflow.advance(env.observe(), verifier, env, lambda *a, **k: None)
    assert not env._browser_session.calls


def test_stale_click_cannot_trigger_automatic_download_or_positive_sft(tmp_path, monkeypatch):
    monkeypatch.setattr("osworld_agent.pipeline._viz_log_status", None)
    monkeypatch.setattr("osworld_agent.paper_workflow.resolve_arxiv_candidates",
                        partial(resolve_arxiv_candidates, fetcher=metadata_fetch, identity_selector=fixture_selection))
    browser = environment(tmp_path)
    env = Env()
    env._browser_session = browser._browser_session
    env.obs = browser.observe()
    env.observe = browser.observe
    env.route_action = lambda action, obs: {"channel": "reobserve", "reason": "invalid ref", "error_code": "stale"}
    model = Model([{"type": "click", "target": "Download PDF", "target_ref": "1000,17"}], paper_plan())
    pipeline = Pipeline(model=model, env=env, grounding=None, system_prompt="test", max_steps=1)
    pipeline.paper_identity_selector = SimpleNamespace(calls=[], select=fixture_selection)
    result = pipeline.run("请下载一篇叫做Roboclaw的论文，并重命名为Roboclaw")
    assert not result.success and result.steps == 1 and len(model.calls) == 2
    assert not browser._browser_session.calls
    assert result.trajectory[0]["execution_info"]["status"] == "rejected"
    assert result.run_manifest["goal_bindings"] and result.run_manifest["goal_binding_hash"]
    trajectory = build_trajectory("short-paper", "browser", result.task, result)
    assert not list(iter_sft_records(trajectory))
    assert any(e["event_type"] == "goal_binding" for e in trajectory["events"])


def test_semantic_extra_requirement_is_preserved_for_independent_review(monkeypatch):
    monkeypatch.setattr("osworld_agent.pipeline._viz_log_status", None)
    model, env = Model([{"type": "fail"}], paper_plan()), Env()
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test", max_steps=1).run(
        "请下载一篇叫做Roboclaw的论文，并发送给导师")
    assert not result.success and result.steps == 1
    unresolved = [e for e in result.events if e["event_type"] == "contract_unresolved"]
    assert unresolved and "发送给导师" in str(unresolved)


@pytest.mark.parametrize("phases,valid", [(["download", "verify"], True),
                                         (["verify", "download"], False), (["download"], False)])
def test_paper_replan_can_cover_only_remaining_phases(phases, valid):
    from osworld_agent.task_specs import task_profile
    plan = {"steps": [{"phase": phase, "goal": phase, "expected_evidence": "实际文件证据"} for phase in phases],
            "keywords": ["AAA"]}
    model = Model([])
    model.responses = [json.dumps(plan)] * 2
    pipe = Pipeline(model=model, env=Env(), grounding=None, system_prompt="test")
    task = "请下载一篇叫做AAA的论文"
    pipe.completion = CompletionVerifier(task)
    state = pipe._init_state(task)
    state.task_profile = task_profile(task, pipe.completion.public_spec)
    assert pipe._plan(task, state, repair=True) is valid


def test_binding_group_excludes_timestamps_but_keeps_paper_identity():
    binding = {"query": "RoboClaw", "canonical_id": "2603.11558", "retrieved_at": 1}
    first = binding_group_hash({"check_1": binding})
    binding["retrieved_at"] = 100
    assert binding_group_hash({"check_1": binding}) == first
    binding["canonical_id"] = "2603.99999"
    assert binding_group_hash({"check_1": binding}) != first


def test_replacement_is_not_sft_positive_and_wait_is_not_replacement():
    data = trajectory()
    data["steps"][0]["execution_info"]["framework_replaced_action"] = True
    assert not list(iter_sft_records(data))
    data["steps"][0]["execution_info"] = {"status": "executed", "framework_wait": {"elapsed_s": 60}}
    assert len(list(iter_sft_records(data))) == 1


def test_credit_groups_separate_wait_rules_and_paper_bindings():
    first, second = credit_rollout(), credit_rollout()
    second.runtime_config_hash = "different-recovery-rules"
    with pytest.raises(ValueError, match="runtime"):
        build_tree([first, second])
    second.runtime_config_hash = ""
    second.goal_binding_hash = "different-paper"
    with pytest.raises(ValueError, match="binding"):
        build_tree([first, second])


def test_rename_semantics_use_source_and_name_not_random_id():
    def step(artifact_id):
        return {"action": {"action": "rename_artifact", "args": {"artifact_id": artifact_id, "filename": "paper.pdf"}},
                "observation": {"info": {"artifacts": [{"artifact_id": artifact_id, "url": URL}]}}}
    assert _step_semantic(step("random-1")) == _step_semantic(step("random-2"))
    assert URL in _step_semantic(step("random-1"))
    assert _step_semantic({"decision_calls": [{"response": "invalid"}]}) == "invalid_decision()"


@pytest.mark.parametrize("name", ["../paper.pdf", "dir/paper.pdf", "dir\\paper.pdf", "", ".."])
def test_rename_rejects_paths(name):
    assert validate_action(Action("rename_artifact", {"artifact_id": "a", "filename": name}))
