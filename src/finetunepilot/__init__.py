"""FineTunePilot: an agent that plans, runs, diagnoses and compares QLoRA fine-tuning runs."""

import warnings
from importlib.metadata import PackageNotFoundError, version

# Raised inside peft/trl (not our code) under transformers 5; it only adds noise to the CLI.
warnings.filterwarnings("ignore", message="`is_gradient_checkpointing` is deprecated")

try:
    __version__ = version("finetunepilot")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"
