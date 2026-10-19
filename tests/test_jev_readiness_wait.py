"""A black post-navigation frame is polled without another policy action."""
from types import SimpleNamespace

from PIL import Image, ImageDraw

from osworld_agent.actions import Action
from osworld_agent.env import Observation
from osworld_agent.pipeline import Execution, Pipeline
from osworld_agent.visual_readiness import likely_loading_page
from test_pipeline_completion import Env, Model


def page(screenshot, names=()):
    return Observation(screenshot=screenshot,
        context=[{'kind': 'browser_ax', 'url': 'https://example.org/search',
                  'nodes': [{'role': 'heading', 'name': name} for name in names]}],
        info={'url': 'https://example.org/search', 'active_surface': 'browser'})


def test_loading_wait_uses_one_frame_jev_and_returns_fresh_page(monkeypatch):
    black = page(Image.new('RGB', (600, 400), '#101827'))
    ready_image = Image.new('RGB', (600, 400), 'white')
    ImageDraw.Draw(ready_image).rectangle((20, 20, 220, 120), fill='blue')
    ready = page(ready_image, ('Search results', 'Attention Is All You Need'))
    assert likely_loading_page(black)
    assert not likely_loading_page(ready)

    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    pipe._task_time_remaining = lambda: 10
    pipe._observe = lambda: ready
    seen = []
    pipe.decision_judge = SimpleNamespace(readiness=lambda image, **kwargs:
        (seen.append(image) or {'status': 'answered', 'model': 'fixture', 'latency_ms': 10,
           'answers': {'readiness': {'label': 'waiting', 'confidence': .95}}}))
    monkeypatch.setattr('osworld_agent.pipeline.time.sleep', lambda duration: None)
    result = pipe._settle_page_readiness(Action('goto', {'url': 'https://example.org/search'}),
        Execution(kind='browser_dom', obs=black, info={'status': 'executed'}))
    assert len(seen) == 1 and result.obs.screenshot is ready.screenshot
    assert result.obs.context == ready.context and 'page_readiness_wait' not in ready.info
    assert result.info['page_readiness_wait']['status'] == 'ready_for_policy'


def test_error_page_does_not_trigger_readiness_poll():
    blocked = page(Image.new('RGB', (600, 400), 'white'), ('Rate exceeded.',))
    assert not likely_loading_page(blocked)
