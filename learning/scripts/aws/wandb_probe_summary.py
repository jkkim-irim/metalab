"""Push a watcher probe result into the TRAINING run's W&B summary (``probe/*`` keys).

The SR-probing watcher (`watch_train.sh`) evaluates fresh checkpoints out-of-band, so its results
otherwise exist only in the watcher's local log — invisible on the run page. This writes them as
summary keys (probe/SR, probe/iter, probe/best_SR, probe/best_iter) via the public API: a k/v
summary update is safe alongside the live training writer (no second history writer).

Usage (on the node, wandb-authed):
    python wandb_probe_summary.py <run_display_name> <iter> <sr> <best_sr> <best_iter>
Project defaults to wiin2-wirobotics-inc/chrisryu-simrl (override with WANDB_PROBE_PROJECT).
"""
import os
import sys

import wandb


def main() -> None:
    name, it, sr, best, best_it = sys.argv[1:6]
    project = os.environ.get("WANDB_PROBE_PROJECT", "wiin2-wirobotics-inc/chrisryu-simrl")
    runs = wandb.Api().runs(project, {"display_name": name})
    matches = [r for r in runs]
    if not matches:
        print(f"PROBE_SUMMARY_SKIP no run named {name!r} in {project}")
        return
    run = matches[0]  # newest first; the live training run
    run.summary.update({"probe/iter": int(it), "probe/SR": float(sr),
                        "probe/best_SR": float(best), "probe/best_iter": int(best_it)})
    run.update()
    print(f"PROBE_SUMMARY_OK {run.id} probe/SR={sr} @ {it} (best {best} @ {best_it})")


if __name__ == "__main__":
    main()
