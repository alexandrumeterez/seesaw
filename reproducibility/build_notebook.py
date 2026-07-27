#!/usr/bin/env python3
"""Build the executable Figures 1 and 2 reproduction notebook."""

from pathlib import Path

import nbformat as nbf


def main() -> None:
    target = Path(__file__).with_name("reproduce_figures.ipynb")
    notebook = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        }
    )
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            """# Seesaw Figures 1 and 2

This notebook reproduces the primary Seesaw paper figures from immutable W&B
run IDs in `run_manifest.json`. It does **not** use the exploratory notebooks
that produced the paper graphics.

The shared module validates each run's state, group, Git metadata, and logged
configuration, then reads its complete W&B history artifact. Each trace is
plotted once without Seaborn aggregation. Set `WANDB_API_KEY` or run
`wandb login` before executing."""
        ),
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
if not (repo_root / "reproducibility").is_dir():
    repo_root = repo_root.parent
assert (repo_root / "reproducibility").is_dir(), "Run from the repository root or reproducibility/"
sys.path.insert(0, str(repo_root))

from reproducibility.reproduce_figures import reproduce

repro_dir = repo_root / "reproducibility"
output_dir = repro_dir / "outputs"
cache_dir = repro_dir / "cache"
manifest_path = repro_dir / "run_manifest.json"
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Validate, download, and plot

The first execution downloads the full history artifacts. Later executions use
the local ignored cache but still validate the live W&B run metadata against
the manifest."""
        ),
        nbf.v4.new_code_cell(
            """endpoints = reproduce(
    manifest_path=manifest_path,
    output_dir=output_dir,
    cache_dir=cache_dir,
    figures=("figure_1", "figure_2"),
)
endpoints"""
        ),
        nbf.v4.new_markdown_cell("## Figure 1: Seesaw versus cosine"),
        nbf.v4.new_code_cell(
            """from IPython.display import Image, display

display(Image(filename=output_dir / "figure_1_main.png", width=1100))"""
        ),
        nbf.v4.new_markdown_cell("## Figure 2: equivalence-family schedules"),
        nbf.v4.new_code_cell("""display(Image(filename=output_dir / "figure_2_equivalence.png", width=1100))"""),
        nbf.v4.new_markdown_cell(
            """## Logged sequential-step reductions

These are optimizer-step reductions at equal token budgets, not measured
wall-clock speedups."""
        ),
        nbf.v4.new_code_cell(
            """import pandas as pd

pd.read_csv(output_dir / "figure_1_step_reductions.csv")"""
        ),
        nbf.v4.new_markdown_cell(
            """The command-line entry point can also reproduce the supplied
large-batch and weight-decay plots:

```bash
python reproducibility/reproduce_figures.py
```

See `reproducibility/README.md` for the run-selection policy and output
details."""
        ),
    ]
    nbf.write(notebook, target)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
