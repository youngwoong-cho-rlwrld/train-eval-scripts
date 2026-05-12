# train-eval-scripts

Cluster-agnostic train + eval orchestration for the GR00T baselines.

## Layout

```
.
├── submit                       # cluster-aware sbatch wrapper (entry point)
├── clusters/
│   ├── kakao.env                # in-house cluster paths/partition
│   └── skt.env                  # AWS cluster paths/partition
├── lib/
│   ├── _common.sh               # log(), find_available_port(), GPU detect
│   ├── train_body.sh            # Phase 1 — single-script training
│   └── eval_body.sh             # Phase 2 + 3 — eval over EVAL_SETS, then aggregate
├── experiments/                 # one dir per variant; runtime artifacts gitignored
│   ├── baseline_pretrained/
│   │   ├── config.sh            # variant knobs (GPUs, batch, --tune-visual etc.)
│   │   └── (gitignored: checkpoints/ eval_results/ logs/ wandb/ results.json data_config.yaml)
│   ├── baseline_pretrained_tunevisual/
│   ├── baseline_scratch/
│   └── baseline_scratch_tunevisual/
└── utils/
    ├── sqf.sh                   # interactive job & log viewer (see Monitoring)
    └── visualize_state_action.py
```

## Usage

From either cluster (no need to know which one — auto-detected):

```bash
./submit train baseline_pretrained
./submit eval  baseline_pretrained
./submit eval  baseline_pretrained_tunevisual
```

The wrapper:
1. Detects cluster by checking which network FS is mounted (`/fsx/rlwrld` → skt, `/rlwrld2/home` → kakao).
2. Sources `clusters/<name>.env` for partition + path overlay.
3. Sources `experiments/<variant>/config.sh` for GPU count + variant-specific train flags.
4. Calls `sbatch` against `lib/train_body.sh` or `lib/eval_body.sh` with the right cluster-specific options.

## Monitoring with `sqf`

`utils/sqf.sh` is an interactive job + log viewer for active and recent runs. Single-key menu with options for `sacct`/`squeue` status, live-tailed STDOUT / STDERR, per-eval Isaac Sim server logs, GPU availability, and `scancel`.

```bash
~/train-eval-scripts/utils/sqf.sh
```

Defaults: `$HOME/logs` for slurm logs, `$HOME/train-eval-scripts/experiments` for per-variant logs. Override via `SQF_LOG_DIR` / `SQF_EXP_DIR` env vars, or the `d` menu option (persists to `~/.config/sqf/config`).

## Adding a variant

1. Make `experiments/<new_variant>/config.sh` (copy an existing one, tweak `TRAIN_EXTRA_ARGS`).
2. Done. No new run/eval scripts needed.

## Adding a cluster

1. Make `clusters/<new>.env` with the same keys as kakao/skt envs.
2. Add a detection branch in `submit`.

## Cross-cluster eval matrix

The same checkpoint can be evaluated on both clusters because each cluster's repo working tree maintains its own `experiments/<variant>/checkpoints/` (gitignored). To compare L40S vs A100 renders for variant V:

1. Train on whichever cluster: `./submit train V` → produces `experiments/V/checkpoints/checkpoint-<MAX_STEPS>` on that cluster.
2. Copy `experiments/V/checkpoints/` to the other cluster's repo working tree manually.
3. Run `./submit eval V` on each cluster — eval results land in `experiments/V/eval_results/` and `experiments/V/results.json` on each side.
