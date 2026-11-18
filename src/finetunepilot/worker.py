"""Runs ONE training attempt in its own process: `python -m finetunepilot.worker req.json res.json`.

The runner starts a fresh process per attempt, so when an attempt dies from CUDA OOM all of
its GPU memory is released before the next attempt starts.

request.json: {"config": <TrainConfig>, "output_dir": str, "dry_run": bool,
               "simulate_oom_gpu_gb": float | null}
result.json:  {"status": "success", "result": <TrainResult>, "data_stats": {...}}
           or {"status": "failed", "error_type", "error", "traceback", "data_stats"}

`simulate_oom_gpu_gb` (dry runs only) raises a fake CUDA OOM when the VRAM estimate of the
REAL config does not fit that much memory, so the recovery loop can be demoed on a laptop.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def simulated_oom_check(cfg, gpu_memory_gb: float, seq_len: int | None) -> None:
    """Raise a fake CUDA OOM if the real config's estimate does not fit gpu_memory_gb."""
    import torch

    from finetunepilot.tools.vram import estimate_vram

    est = estimate_vram(cfg, gpu_memory_gb, seq_len=seq_len, safety_margin=0.0)
    if not est.fits:
        raise torch.OutOfMemoryError(
            f"CUDA out of memory (simulated in dry run): estimated {est.total_gb:.2f} GiB "
            f"for batch {cfg.per_device_batch_size}, but the GPU has {gpu_memory_gb:.2f} GiB."
        )


def run_attempt(request: dict[str, Any], result_path: Path) -> int:
    from finetunepilot.config import TrainConfig
    from finetunepilot.train import train

    data_stats: dict[str, int] = {}

    def on_stats(stats: dict[str, int]) -> None:
        data_stats.update(stats)
        _write(result_path, {"status": "running", "data_stats": data_stats})

    try:
        cfg = TrainConfig.model_validate(request["config"])
        if request.get("dry_run") and request.get("simulate_oom_gpu_gb"):
            simulated_oom_check(cfg, request["simulate_oom_gpu_gb"], request.get("seq_len"))
        result = train(
            cfg, output_dir=request["output_dir"], dry_run=request.get("dry_run", False),
            on_data_stats=on_stats,
        )  # fmt: skip
    except BaseException as e:  # noqa: BLE001 - report everything, including KeyboardInterrupt
        _write(
            result_path,
            {
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e)[:4000],
                "traceback": "".join(traceback.format_exception(e))[-8000:],
                "data_stats": data_stats,
            },
        )
        return 1
    _write(
        result_path, {"status": "success", "result": result.model_dump(), "data_stats": data_stats}
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m finetunepilot.worker request.json result.json", file=sys.stderr)
        return 2
    request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    return run_attempt(request, Path(argv[1]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
