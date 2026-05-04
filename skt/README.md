# SKT cluster scripts

AWS ParallelCluster (us-east-1), L40S + H200. SSH: `skt`. Home: `/fsx/rlwrld/<user>` (Lustre).

## Defaults

| Field | Value |
|---|---|
| Train partition | `l40s-gpu` (L40S, 4 GPU/node, 120 d max) |
| Eval partition | `l40s-gpu_background` (preemptible, 48 h max — eval convention) |
| Train GPUs | 2 |
| Eval GPUs | 1 |
| Train walltime | 48 h |
| Eval walltime | 12 h |
| Default `GPU_LABEL` | `l40s` |

The skt cluster has tiered partitions per GPU type (`*` / `*_urgent` / `*_premium` / `*_background`). See [`docs/slurm-structure.md`](../../docs/slurm-structure.md) for the policy.

## Submit

```bash
# train (L40S default)
sbatch ~/script/skt/youngwoong_onboarding_train_pretrained.sh
sbatch ~/script/skt/youngwoong_onboarding_train_scratch.sh

# eval — single distance
EVAL_SET=0cm sbatch ~/script/skt/youngwoong_onboarding_eval_pretrained.sh

# eval — sweep distances
for d in 0cm 1cm 3cm 5cm; do
    EVAL_SET=$d sbatch ~/script/skt/youngwoong_onboarding_eval_pretrained.sh
done

# eval on H200 instead — override partition + label at submit time
GPU_LABEL=h200 EVAL_SET=0cm sbatch -p rlwrld-gpu_background \
    ~/script/skt/youngwoong_onboarding_eval_pretrained.sh

# train on H200 instead
sbatch -p rlwrld-gpu ~/script/skt/youngwoong_onboarding_train_pretrained.sh
```

## Env-var knobs (eval scripts)

| Var | Default | Effect |
|---|---|---|
| `EVAL_SET` | `0cm` | Eval-set name; passed to Isaac `server_v2.py --eval-set` and embedded in the output dir |
| `GPU_LABEL` | `l40s` | Tag for the output dir, useful when comparing across GPU types |

## Output

```
/fsx/rlwrld/<user>/eval_results/<run_name>/<EVAL_SET>_<GPU_LABEL>_<yyyymmddHHMMSS>/
    server.log       Isaac Sim server stdout/stderr
    results.json     final per-episode + aggregate eval summary
    *.parquet        per-episode (state, action) trajectories (use utils/visualize_state_action.py)
```

Slurm logs land in `/fsx/rlwrld/<user>/logs/<jobname>_<jobid>.{out,err}` — create that dir first.

## Notes

- `/rlwrld[1-4]` does **not** exist on skt — paths under `/fsx/rlwrld/<user>` are the equivalents.
- Slurm CLI lives at `/opt/slurm/bin` and is auto-added to `PATH` by `/etc/profile.d/aws-pcluster-env.sh` for the `rlwrld` group.
- Default partition for the `rlwrld` group is `rlwrld-gpu` (H200), so explicit `-p l40s-gpu*` is required for L40S work — already set in the `#SBATCH` headers here.
