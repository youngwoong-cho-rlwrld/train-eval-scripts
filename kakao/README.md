# Kakao cluster scripts

In-house cluster, A100/H100. SSH: `kakao-login-1` / `-2` / `-3`. Home: `/rlwrld2/home/<user>`.

## Defaults

| Field | Value |
|---|---|
| Train partition | `rlwrld` (A100, 8 GPU/node, 48 h max) |
| Eval partition | `rlwrld` (in this script set; convention says `background`) |
| Train GPUs | 2 |
| Eval GPUs | 1 |
| Train walltime | 48 h |
| Eval walltime | 12 h |
| Default `GPU_LABEL` | `a100` |

## Submit

```bash
# train
sbatch ~/script/kakao/youngwoong_onboarding_train_pretrained.sh
sbatch ~/script/kakao/youngwoong_onboarding_train_scratch.sh

# eval — single distance
EVAL_SET=0cm sbatch ~/script/kakao/youngwoong_onboarding_eval_pretrained.sh

# eval — sweep distances
for d in 0cm 1cm 3cm 5cm; do
    EVAL_SET=$d sbatch ~/script/kakao/youngwoong_onboarding_eval_pretrained.sh
done

# eval — different GPU type (H100), tag the output
GPU_LABEL=h100 EVAL_SET=0cm sbatch -p h100 \
    ~/script/kakao/youngwoong_onboarding_eval_pretrained.sh
```

## Env-var knobs (eval scripts)

| Var | Default | Effect |
|---|---|---|
| `EVAL_SET` | `0cm` | Eval-set name; passed to Isaac `server_v2.py --eval-set` and embedded in the output dir |
| `GPU_LABEL` | `a100` | Tag for the output dir, useful when comparing across GPU types |

## Output

```
/rlwrld2/home/<user>/eval_results/<run_name>/<EVAL_SET>_<GPU_LABEL>_<yyyymmddHHMMSS>/
    server.log       Isaac Sim server stdout/stderr
    results.json     final per-episode + aggregate eval summary
    *.parquet        per-episode (state, action) trajectories (use utils/visualize_state_action.py)
```

Slurm logs land in `/rlwrld2/home/<user>/logs/<jobname>_<jobid>.{out,err}` — create that dir first.
