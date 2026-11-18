"""FineTunePilot: an agent that plans, runs, diagnoses and compares QLoRA fine-tuning runs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("finetunepilot")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"
