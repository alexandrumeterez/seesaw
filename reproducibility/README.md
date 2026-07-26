# Reproducing the Seesaw figures

This directory is a clean-room reconstruction of the paper figures. It does not
import or execute the exploratory plotting notebooks used during paper
development.

## What is pinned

`run_manifest.json` contains the exact W&B run ID and expected configuration
for every trace. The plotting code rejects a run whose state, group, recorded
Git commit, or important model/optimizer/scheduler fields differ from the
manifest. It then downloads the complete `wandb-history` artifact and reads
only the required metrics:

- `eval/c4_val/CrossEntropyLoss`
- `throughput/total_tokens`
- `total_batches_seen`
- `batch_size`
- `optim/learning_rate_group0`

There is no implicit mean, confidence interval, or duplicate-run aggregation.
The endpoint CSV records exactly which data were plotted.

The manifest's run-discovery policy is executable in `discover_runs.py`.
Discovery is intentionally separate from reproduction: do not regenerate the
manifest unless you mean to select from the current mutable W&B groups.

## Setup

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reproducibility/requirements.txt
wandb login
```

Authentication is read from W&B's normal credential store or the
`WANDB_API_KEY` environment variable. No API key is embedded in these files.

## Figures 1 and 2 notebook

Open and run `reproduce_figures.ipynb`. The notebook calls the independently
implemented plotting module and produces:

- `outputs/figure_1_main.pdf` and `.png`
- `outputs/figure_2_equivalence.pdf` and `.png`
- `outputs/plotted_endpoints.csv`
- `outputs/figure_1_step_reductions.csv`

The checked-in notebook has been executed once against the pinned manifest.
Regenerate its source deterministically with:

```bash
python reproducibility/build_notebook.py
```

## Command line and supplemental figures

To reproduce all supplied plots:

```bash
python reproducibility/reproduce_figures.py
```

To reproduce a subset:

```bash
python reproducibility/reproduce_figures.py --figures figure_1 figure_2
```

The supplemental outputs are `figure_3_large_batches` and
`appendix_weight_decay`. Cache files land in `reproducibility/cache/` and are
ignored by Git.

## Interpretation and provenance

The lower row of Figure 1 measures sequential optimizer steps, not observed
wall-clock time. The checked-in CSV shows a 38.3–38.4% step reduction for the
displayed cells. Real wall-clock acceleration depends on enough hardware to
process the increased global batch without increasing step time.

Some W&B runs were executed from dirty worktrees: their output contains source
markers absent from the commit recorded by W&B. The metrics and run
configurations are reproducible, but the exact training diff is not
recoverable. See `../AUDIT.md` for the full audit.
