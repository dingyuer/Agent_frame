"""Qwen visual fallback returns screenshot-space cells and rejects bad answers."""
from types import SimpleNamespace

from PIL import Image

from osworld_agent.adapters.qwen_grounding import QwenPlusGrounding


def _grounding(*answers):
    remaining = iter(answers)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_kwargs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=next(remaining)))]))))
    grounder = QwenPlusGrounding('https://example.test/v1', 'test', 'qwen3.7-plus')
    grounder._client = client
    return grounder


def test_grid_answer_maps_to_original_screenshot_pixels():
    result = _grounding('{"column":10,"row":2}', '{"column":6,"row":5}').locate(
        Image.new('RGB', (1440, 900)), 'blue Search button')
    assert (result.x, result.y, result.confidence) == (1275, 265, 1.0)


def test_invalid_grid_answer_fails_closed():
    result = _grounding('{"column":12,"row":2}').locate(Image.new('RGB', (1440, 900)), 'missing')
    assert result.confidence == 0 and (result.x, result.y) == (-1, -1)


def test_invalid_refinement_fails_closed():
    result = _grounding('{"column":10,"row":2}', '{"column":-1,"row":-1}').locate(
        Image.new('RGB', (1440, 900)), 'missing')
    assert result.confidence == 0
