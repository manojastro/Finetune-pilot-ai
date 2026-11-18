# FineTunePilot session `s-20261118-170453-8148a0`

> **DRY RUN.** Training and evaluation were simulated with a toy model of config quality. These numbers are NOT measurements.

- **Objective:** Pick the right tool and produce exactly the right JSON arguments
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct` (QLoRA, 4-bit)
- **Dataset:** `data/sample`
- **GPU:** t4
- **Budget:** 3 runs, 90 minutes
- **Status:** completed
- **Stopped because:** max runs (3) reached

## Results

| run | config change | train | attempts | final loss | json_valid | tool_name_accuracy | args_exact_match | run logs |
|---|---|---|---|---|---|---|---|---|
| base | - (no adapter) | - | - | - | 0.7500 | 0.4500 | 0.1500 | `s-20261118-170453-8148a0-base-dryrun-20261118-170454-877b3052` |
| 1 | initial plan | success | 1 | 0.4639 | 0.9000 | 0.8000 | 0.7000 | `s-20261118-170453-8148a0-run1-dryrun-20261118-170456-b7ef6460`, `s-20261118-170453-8148a0-run1-eval-dryrun-20261118-170456-bf81fdc2` |
| 2 | learning_rate=0.0002 | success | 1 | 0.3826 | 0.9500 | 0.8500 | 0.8000 | `s-20261118-170453-8148a0-run2-dryrun-20261118-170456-039e761f`, `s-20261118-170453-8148a0-run2-eval-dryrun-20261118-170456-29e4b1f1` |
| 3 | lora_r=16 | success | 1 | 0.3075 | 0.9500 | 0.9000 | 0.8500 | `s-20261118-170453-8148a0-run3-dryrun-20261118-170456-0789eb92`, `s-20261118-170453-8148a0-run3-eval-dryrun-20261118-170456-fd55ce0f` |

## Best run

Run 3: args_exact_match 0.8500 (+0.7000 vs base 0.1500), tool_name_accuracy 0.9000.
Adapter: `outputs/agent/s-20261118-170453-8148a0/run-3/s-20261118-170453-8148a0-run3-dryrun-20261118-170456-0789eb92/attempt-1/adapter`

## What changed and why

### Run 1 (plan)
- Reason: Dry run: a conservative first config (low learning rate, small LoRA rank).
- Config: learning_rate=0.0001, num_epochs=3.0, lora_r=8, lora_alpha=16, lora_dropout=0.05, per_device_batch_size=4, grad_accum_steps=2, max_seq_length=1024, gradient_checkpointing=False, warmup_ratio=0.0
- VRAM estimate before training: 11.277 GiB (an estimate)
- Training: success (succeeded on attempt 1)
- Diagnosis: set `learning_rate` = `0.0002`. Dry run: loss still high; raise lr.

### Run 2 (diagnose)
- Reason: Dry run: loss still high; raise lr.
- Config: learning_rate=0.0002
- VRAM estimate before training: 11.277 GiB (an estimate)
- Training: success (succeeded on attempt 1)
- Diagnosis: set `lora_r` = `16`. Dry run: more adapter capacity for arguments.

### Run 3 (diagnose)
- Reason: Dry run: more adapter capacity for arguments.
- Config: lora_r=16
- VRAM estimate before training: 11.322 GiB (an estimate)
- Training: success (succeeded on attempt 1)
- Diagnosis: set `num_epochs` = `5`. Dry run: tiny dataset, train longer.
