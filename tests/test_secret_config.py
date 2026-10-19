"""Credential references stay out of tracked model configuration."""
import pytest

from osworld_agent import config


def test_resolve_secret_prefers_process_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setenv("MODEL_TEST_KEY", "from-environment")
    assert config.resolve_secret("env:MODEL_TEST_KEY") == "from-environment"
    assert config.resolve_secret("${MODEL_TEST_KEY}") == "from-environment"


def test_resolve_secret_reads_ignored_dotenv(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / ".env").write_text("MODEL_TEST_KEY='from-dotenv'\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.delenv("MODEL_TEST_KEY", raising=False)
    assert config.resolve_secret("env:MODEL_TEST_KEY") == "from-dotenv"


def test_missing_secret_has_named_error(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.delenv("MODEL_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_TEST_KEY"):
        config.resolve_secret("env:MODEL_TEST_KEY")
