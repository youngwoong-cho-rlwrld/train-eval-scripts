# Utilities

Cluster-agnostic helpers used alongside the train/eval scripts.

## `sqf.sh`

Interactive job & log viewer. Auto-tails Slurm logs; press `Ctrl-C` to return to the menu.

```bash
~/script/utils/sqf.sh
```

Env-var defaults (override before launch):

| Var | Default | Purpose |
|---|---|---|
| `SQF_LOG_DIR` | `$HOME/logs` | Slurm `.out` / `.err` location |
| `SQF_EVAL_DIR` | `$HOME/eval_results` | Eval rollout dirs (`server.log`, `results.json`, `*.parquet`) |

## `visualize_state_action.py`

Curses-based parquet picker that plots one episode's `observation.state` (solid) vs `action` (dashed), one trace per joint, on a single matplotlib figure.

```bash
python ~/script/utils/visualize_state_action.py
# prompted for a directory containing .parquet files
```

Expects each parquet to have `observation.state` and `action` columns whose entries are equal-length 1-D arrays (one per joint). Used to inspect per-episode rollouts produced by the eval scripts.

## `diagram_lib/`

Standalone HTML/JS library for tensor-flow diagrams (one SVG per model). Open any `diagrams/*.html` in a browser; no build step.

| File | Notes |
|---|---|
| `diagram.css` / `diagram.js` | Renderer. `renderDiagram(svgEl, spec)` consumes a JSON-like spec of modules + nodes + edges. |
| `diagrams/<name>.html` | Page that loads the renderer and a per-model `*_data.js`. |
| `diagrams/<name>_data.js` | The model spec (modules, nodes, edges, plot labels). |

Current diagrams: `gr00t`, `pi0`, `ta_vla`. Add a new one by copying any pair (`*.html` + `*_data.js`) and editing the spec.
