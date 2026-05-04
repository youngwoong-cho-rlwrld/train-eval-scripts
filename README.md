# RLWRLD scripts

Slurm submission scripts for the two RLWRLD clusters (kakao + skt) plus utilities I use across both.

## Layout

```
train-eval-scripts/   # default name from `git clone <url>`
├── kakao/   submission scripts for kakao (in-house, A100/H100)
├── skt/     submission scripts for skt (AWS, L40S/H200)
└── utils/   cluster-agnostic helpers (job viewer, plotter, diagram lib)
```

All command examples below assume you are `cd`'d into the repo root.

Each cluster directory has the same four scripts:

| File | Purpose |
|---|---|
| `youngwoong_onboarding_train_pretrained.sh` | Fine-tune from pretrained backbone |
| `youngwoong_onboarding_train_scratch.sh` | Train action head from scratch (`--random-diffusion`) |
| `youngwoong_onboarding_eval_pretrained.sh` | Evaluate the pretrained checkpoint |
| `youngwoong_onboarding_eval_scratch.sh` | Evaluate the scratch checkpoint |

The scripts are otherwise identical between clusters except for paths, partition, and the default `GPU_LABEL` — see each cluster's README.

## Adapting for someone else

Paths and account names are hardcoded for `youngwoong_cho`. Before reuse, edit:

- `GROOT_DIR`, `ISAAC_DIR`, `CHECKPOINT`, `OUTPUT_DIR` (paths)
- `#SBATCH --output` / `--error` (log dirs)
- `#SBATCH --job-name` (uniqueness in `squeue`)
- `WANDB_PROJECT` if you publish to a different project

## Quick start

```bash
# kakao: A100 eval at 0cm
ssh kakao-login-1
EVAL_SET=0cm sbatch ./kakao/youngwoong_onboarding_eval_pretrained.sh

# skt: L40S eval at 0cm
ssh skt
EVAL_SET=0cm sbatch ./skt/youngwoong_onboarding_eval_pretrained.sh
```
