# RLWRLD scripts

Slurm submission scripts for the two RLWRLD clusters (kakao + skt) plus utilities I use across both.

## Layout

```
train-eval-scripts/   # default name from `git clone <url>`
├── kakao/   submission scripts for kakao (in-house, A100/H100)
│   ├── baseline_pretrained/run.sh   # train + eval, no --random-diffusion
│   └── baseline_scratch/run.sh      # train + eval, with --random-diffusion
├── skt/     placeholder for skt (AWS, L40S/H200) — no scripts yet
└── utils/   cluster-agnostic helpers (job viewer, plotter, diagram lib)
```

All command examples below assume you are `cd`'d into the repo root.

## How a run.sh works

Each `kakao/<exp>/run.sh` is a single sbatch script that does training → evaluation → result aggregation in one job:

1. **Phase 1 — Training**: skipped if `<exp>/checkpoints/checkpoint-<MAX_STEPS>` already exists; otherwise runs `gr00t_finetune.py` to step `MAX_STEPS` (default `30000`).
2. **Phase 2 — Evaluation**: launches Isaac Sim's `server_v2.py` and runs `eval_allex.py` for each `EVAL_SETS × N_RUNS` (default `0cm,3cm × 3 = 6 runs of 70 episodes`).
3. **Phase 3 — Aggregation**: writes mean ± std across runs to `<exp>/results.json`.

All artifacts (`checkpoints/`, `eval_results/`, `logs/`, `wandb/`, the auto-generated `data_config.yaml`) are written **inside the script's own directory**. The repo's `.gitignore` excludes them so the working tree stays clean.

## Naming

The Slurm job-name and wandb run-name are composed at runtime as:

```
{parent_dir}_{gpu_instance}_{yyyymmddHHMMSS}
e.g. baseline_pretrained_a100_20260506081503
```

`gpu_instance` is detected via `nvidia-smi` (`a100`, `h100`, `h200`, `l40s`, `v100`, `unknown`). The job is renamed at runtime via `scontrol update JobName=…` so `squeue`/`sacct` reflect the dynamic name.

The Slurm log filename is fixed by `#SBATCH --output` to `~/logs/{static_name}_{jobid}.{out,err}` (Slurm parses `#SBATCH` directives before the script body runs, so the filename can't include the dynamic name; `%j` keeps it unique).

## Adapting for someone else

Paths are hardcoded for `youngwoong_cho`. Before reuse, edit each `run.sh`:

- `GROOT_DIR`, `ISAAC_DIR` — your gr00t and rlwrld_isaac trees
- `DATA_PATH` — your dataset path
- `WANDB_PROJECT` — wandb destination
- `#SBATCH --output` / `--error` — slurm log dir (not auto-derived)
- `#SBATCH --partition` if you target a different default

`EXP_DIR` and the dynamic `EXP_NAME` auto-detect — no edit needed.

## Quick start

```bash
ssh kakao-login-1
cd ~/train-eval-scripts                  # or wherever you cloned to
sbatch kakao/baseline_pretrained/run.sh  # train + eval pretrained
sbatch kakao/baseline_scratch/run.sh     # train + eval scratch
```

If `rlwrld` partition is full, route to `background` (preemptible):

```bash
sbatch -p background kakao/baseline_pretrained/run.sh
```

## Monitoring

```bash
./utils/sqf.sh   # interactive viewer: sacct/squeue, .out, .err, server logs, free GPUs
```
