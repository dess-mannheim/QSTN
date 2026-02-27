"""Tests for package and docs version consistency."""

import subprocess
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

    repo_root = Path(__file__).resolve().parents[1]
    try:
        output = subprocess.check_output(
            ["git", "tag", "--sort=-v:refname"],
            cwd=repo_root,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        output = ""

    latest_stable_tag = None
    for raw_tag in output.splitlines():
        tag = raw_tag.strip()
        if not tag:
            continue
        normalized = tag[1:] if tag.startswith("v") else tag
        parts = normalized.split(".")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            latest_stable_tag = normalized
            break

    expected_release = latest_stable_tag or qstn.__version__
    assert docs_conf.release == expected_release
