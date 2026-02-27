"""Public exports for main modules"""

from importlib.metadata import PackageNotFoundError, version as package_version

from . import prompt_builder, survey_manager

try:
    from ._version import __version__
except Exception:
    try:
        __version__ = package_version("qstn")
    except PackageNotFoundError:
        __version__ = "0+unknown"

__all__ = ["prompt_builder", "survey_manager", "__version__"]
