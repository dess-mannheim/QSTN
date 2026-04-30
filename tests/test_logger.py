"""Tests for QSTN logging helpers."""

import logging

import pytest

from qstn import logger as qstn_logger


@pytest.fixture(autouse=True)
def restore_qstn_logger():
    """Keep logger configuration changes local to each test."""
    package_logger = qstn_logger.get_logger()
    original_handlers = list(package_logger.handlers)
    original_level = package_logger.level
    original_propagate = package_logger.propagate
    yield
    package_logger.handlers = original_handlers
    package_logger.setLevel(original_level)
    package_logger.propagate = original_propagate


def test_default_logger_uses_null_handler_and_writes_nothing(capsys):
    package_logger = qstn_logger.get_logger()

    assert any(isinstance(handler, logging.NullHandler) for handler in package_logger.handlers)

    package_logger.warning("hidden by default")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_configure_logging_emits_selected_level_with_tqdm(monkeypatch):
    written: list[str] = []
    monkeypatch.setattr(qstn_logger.tqdm, "write", lambda msg: written.append(msg))

    qstn_logger.configure_logging(level=logging.INFO, force=True)
    package_logger = qstn_logger.get_logger()
    package_logger.debug("debug hidden")
    package_logger.info("hello")

    assert len(written) == 1
    assert "INFO:qstn:hello" in written[0]


def test_tqdm_write_calls_tqdm(monkeypatch):
    written: list[str] = []
    monkeypatch.setattr(qstn_logger.tqdm, "write", lambda msg: written.append(msg))

    qstn_logger.tqdm_write("visible")

    assert written == ["visible"]
