# Utilities

Cluster-agnostic helpers used alongside the train/eval scripts.

## `sqf.sh`

Interactive job & log viewer. Auto-tails Slurm logs; press `Ctrl-C` to return to the menu.

```bash
./utils/sqf.sh
```

Menu:

| | Action |
|---|---|
| `1` | Job status (sacct last 24 h + current squeue) |
| `2` | STDOUT (`.out`) — pick from slurm `*.out` logs |
| `3` | STDERR (`.err`) — pick from slurm `*.err` logs |
| `4` | Isaac Sim server log — pick from `<EXP_DIR>/<exp>/logs/server_<set>_run<i>.log` |
| `5` | GPU availability — free GPUs by partition + per-node detail |
| `d` | Set directories (override `LOG_DIR` / `EXP_DIR` mid-session) |
| `q` | Quit |

Env-var defaults (override before launch):

| Var | Default | Purpose |
|---|---|---|
| `SQF_LOG_DIR` | `$HOME/logs` | Slurm `.out` / `.err` location |
| `SQF_EXP_DIR` | `$HOME/scripts` | Per-experiment dirs containing `logs/server_<set>_run<i>.log` |

## `visualize_state_action.py`

Curses-based parquet picker that plots one episode's `observation.state` (solid) vs `action` (dashed), one trace per joint, on a single matplotlib figure.

```bash
python ./utils/visualize_state_action.py
# prompted for a directory containing .parquet files
```

Expects each parquet to have `observation.state` and `action` columns whose entries are equal-length 1-D arrays (one per joint).

> **Note**: The current `kakao/baseline_*/run.sh` does not pass `--save-data` to `eval_allex.py`, so per-episode parquet files aren't produced. To use this tool against new eval output, add `--save-data` to the `python scripts/eval_allex.py` invocation in `run.sh`.

## `diagram_lib/`

Standalone HTML/JS library for tensor-flow diagrams (one SVG per model). Open any `diagrams/*.html` in a browser; no build step.

| File | Notes |
|---|---|
| `diagram.css` / `diagram.js` | Renderer. `renderDiagram(svgEl, spec)` consumes a JSON-like spec of modules + nodes + edges. |
| `diagrams/<name>.html` | Page that loads the renderer and a per-model `*_data.js`. |
| `diagrams/<name>_data.js` | The model spec (modules, nodes, edges, plot labels). |

Current diagrams: `gr00t`, `pi0`, `ta_vla`. Add a new one by copying any pair (`*.html` + `*_data.js`) and editing the spec.
