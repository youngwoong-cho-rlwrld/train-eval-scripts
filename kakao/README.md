# Kakao cluster scripts

In-house cluster, A100. SSH: `kakao-login-1` / `-2` / `-3`. Home: `/rlwrld2/home/<user>`.

## Defaults (per `#SBATCH` headers in each `run.sh`)

| Field | Value |
|---|---|
| Default partition | `rlwrld` (A100, 8 GPU/node, 48 h max) |
| GPUs requested | 2 |
| Walltime | 48 h |

Override the partition (e.g., to use `background` when `rlwrld` is full, or `h100` for H100s) via the sbatch CLI:

```bash
sbatch -p background kakao/baseline_pretrained/run.sh
sbatch -p h100       kakao/baseline_pretrained/run.sh
```

## Layout

```
kakao/
├── baseline_pretrained/
│   └── run.sh          # train + eval, no --random-diffusion
└── baseline_scratch/
    └── run.sh          # train + eval, with --random-diffusion
```

Each experiment dir gains the following at runtime (all gitignored):

```
checkpoints/{checkpoint-10000, ...-20000, ...-30000, experiment_cfg/, ...}
eval_results/<set>/run_<i>/{results.json, videos/ep000.mp4..ep069.mp4}
logs/{run.log, server_<set>_run<i>.log}
wandb/run-*/
data_config.yaml          # auto-regenerated each run via heredoc
results.json              # aggregated mean ± std (Phase 3 output)
```

## Submit

```bash
# Full pipeline (train then eval)
sbatch kakao/baseline_pretrained/run.sh
sbatch kakao/baseline_scratch/run.sh

# Eval-only (Phase 1 short-circuits when checkpoint-30000 already exists)
sbatch kakao/baseline_pretrained/run.sh

# Different partition (preemptible, starts immediately when rlwrld is full)
sbatch -p background kakao/baseline_pretrained/run.sh
```

The Slurm job-name and wandb run-name are auto-composed at runtime (no env vars needed):

```
{parent_dir}_{gpu_instance}_{yyyymmddHHMMSS}
e.g. baseline_pretrained_a100_20260506081503
```

## Output

Per-run JSON + videos:
```
kakao/<exp>/eval_results/<set>/run_<i>/
    results.json     per-run summary (success_rate, per-episode list)
    videos/ep<NNN>.mp4
```

Aggregated JSON (the experiment's headline number):
```
kakao/<exp>/results.json
```

Slurm stdout/stderr (filename uses the static initial job-name + jobid because `#SBATCH` is parsed before the script body):
```
~/logs/baseline_<variant>_<jobid>.{out,err}
```
