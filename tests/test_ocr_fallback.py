"""OCR fallback stays deterministic and never invokes the trainable policy."""
from io import BytesIO
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

from osworld_agent.pipeline import Pipeline


@pytest.fixture
def agent():
    pipeline = object.__new__(Pipeline)
    pipeline.model = SimpleNamespace(chat=Mock(side_effect=AssertionError("hidden model call")))
    yield pipeline
    pipeline.model.chat.assert_not_called()


def mock_ocr(monkeypatch, texts, **overrides):
    count = len(texts)
    data = {"text": texts,
            "left": [10 + 30 * i for i in range(count)], "top": [10] * count,
            "width": [20] * count, "height": [20] * count,
            "page_num": [1] * count, "block_num": [1] * count,
            "par_num": [1] * count, "line_num": [1] * count,
            "word_num": list(range(1, count + 1)), **overrides}
    reader = Mock(return_value=data)
    monkeypatch.setitem(sys.modules, "pytesseract", SimpleNamespace(
        image_to_data=reader, Output=SimpleNamespace(DICT="dict")))
    return reader


def test_unique_exact_word_is_located_without_model(agent, monkeypatch):
    mock_ocr(monkeypatch, ["Cancel", "Submit"])
    assert agent._ocr_fallback(Image.new("RGB", (100, 80)), "submit") == (50, 20)


@pytest.mark.parametrize("target", ["  SIGN\tIN  ", "ＳＩＧＮ　ＩＮ", "Sign in"])
def test_unique_consecutive_phrase_and_normalization(agent, monkeypatch, target):
    mock_ocr(monkeypatch, ["Sign", "in", "Cancel"])
    assert agent._ocr_fallback(Image.new("RGB", (100, 80)), target) == (35, 20)


@pytest.mark.parametrize("texts,target", [
    (["Submit", "SUBMIT"], "Submit"),
    (["Sign", "in", "Sign", "in"], "Sign in"),
    (["Saved"], "Save"),
    (["Save"], "the Save button"),
    (["Sign", "securely", "in"], "Sign in"),
    ([], "Submit"),
])
def test_ambiguous_or_inexact_targets_fail_closed(agent, monkeypatch, texts, target):
    mock_ocr(monkeypatch, texts)
    assert agent._ocr_fallback(Image.new("RGB", (200, 80)), target) is None


@pytest.mark.parametrize("overrides", [
    {"line_num": [1, 2]}, {"block_num": [1, 2]}, {"word_num": [1, 3]},
])
def test_phrases_cannot_join_separate_lines_blocks_or_missing_words(agent, monkeypatch, overrides):
    mock_ocr(monkeypatch, ["Sign", "in"], **overrides)
    assert agent._ocr_fallback(Image.new("RGB", (100, 80)), "Sign in") is None


def test_repeated_text_on_different_lines_is_still_ambiguous(agent, monkeypatch):
    mock_ocr(monkeypatch, ["Submit", "Submit"], line_num=[1, 2], word_num=[1, 1])
    assert agent._ocr_fallback(Image.new("RGB", (100, 80)), "Submit") is None


def test_missing_python_ocr_dependency_returns_none(agent, monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    assert agent._ocr_fallback(Image.new("RGB", (100, 80)), "Submit") is None


def test_unavailable_ocr_binary_returns_none(agent, monkeypatch):
    reader = mock_ocr(monkeypatch, ["Submit"])
    reader.side_effect = OSError("tesseract is not installed")
    assert agent._ocr_fallback(Image.new("RGB", (100, 80)), "Submit") is None


@pytest.mark.parametrize("overrides", [{"left": [-1]}, {"width": [0]}, {"height": []}])
def test_invalid_ocr_data_returns_none(agent, monkeypatch, overrides):
    mock_ocr(monkeypatch, ["Submit"], **overrides)
    assert agent._ocr_fallback(Image.new("RGB", (100, 80)), "Submit") is None


def test_bytes_screenshot_is_decoded_before_ocr(agent, monkeypatch):
    reader = mock_ocr(monkeypatch, ["Submit"])
    data = BytesIO()
    Image.new("RGB", (100, 80)).save(data, "PNG")
    assert agent._ocr_fallback(data.getvalue(), "Submit") == (20, 20)
    assert reader.call_args.args[0].size == (100, 80)


def test_empty_target_does_not_run_ocr(agent, monkeypatch):
    reader = mock_ocr(monkeypatch, ["Submit"])
    assert agent._ocr_fallback(Image.new("RGB", (100, 80)), " \t ") is None
    reader.assert_not_called()
