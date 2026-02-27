"""Tests for package and docs version consistency."""

from importlib import util as importlib_util
from pathlib import Path

import qstn


def test_qstn_exposes_non_empty_version() -> None:
    assert hasattr(qstn, "__version__")
    assert isinstance(qstn.__version__, str)
    assert qstn.__version__


def test_docs_release_matches_runtime_version() -> None:
    conf_path = Path(__file__).resolve().parents[1] / "docs" / "conf.py"
    spec = importlib_util.spec_from_file_location("qstn_docs_conf", conf_path)
    assert spec is not None
    assert spec.loader is not None
    docs_conf = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(docs_conf)
    assert docs_conf.release == qstn.__version__
