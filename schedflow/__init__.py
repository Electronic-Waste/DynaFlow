from importlib.metadata import version as _version

from . import config as config
from . import example as example
from . import interface as interface
from . import manager as manager

try:
    __version__ = _version("schedflow")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
