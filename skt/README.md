# SKT cluster scripts

> **No scripts ported to skt yet.** Adapt `kakao/baseline_*/run.sh` as a template.

AWS ParallelCluster (us-east-1), L40S + H200. SSH: `skt`. Home: `/fsx/rlwrld/<user>` (Lustre).

## Differences from kakao

- **Home**: `/fsx/rlwrld/<user>` (Lustre), **not** `/rlwrld2/home/<user>` (NFS).
- `/rlwrld[1-4]` and `/rlwrld-dataset` do not exist on skt.
- Slurm CLI lives at `/opt/slurm/bin` (auto-added to `PATH` by `/etc/profile.d/aws-pcluster-env.sh` for the `rlwrld` group).
- Default partition for the `rlwrld` group is `rlwrld-gpu` (H200), so explicit `-p l40s-gpu*` is needed for L40S work.
- Tier policy is implemented: each base partition has `*` (normal) / `*_urgent` / `*_premium` / `*_background` flavors.

See `docs/slurm-structure.md` for the full picture.

## Adapting kakao scripts

Copy `kakao/baseline_pretrained/run.sh` (or `_scratch`) into `skt/<exp>/run.sh` and change:

- `GROOT_DIR`, `ISAAC_DIR` to live under `/fsx/rlwrld/<user>/workspace/`
- `#SBATCH --output` / `--error` to `/fsx/rlwrld/<user>/logs/...`
- `#SBATCH --partition=rlwrld` to `l40s-gpu` (or `rlwrld-gpu` for H200)
- `DATA_PATH` to a path that exists under `/fsx/rlwrld/...`

Everything else (auto-detect of `EXP_DIR`, dynamic `EXP_NAME` from `nvidia-smi`, gr00t fine-tune flags, Phase 1 skip guard) carries over unchanged. The `nvidia-smi` GPU detector already knows about L40S and H200.
