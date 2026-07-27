# Seesaw

Official implementation and figure-reproduction code for
**“Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size
Scheduling.”**

Seesaw replaces part of learning-rate decay with a gradual increase in global
batch size. At or below the critical batch size, this reduces the number of
sequential optimizer steps while matching the validation-loss dynamics of
cosine decay.

This training code is based on [OLMo](https://github.com/allenai/OLMo).

## Repository layout

- `olmo/` contains the model, optimizer, trainer, and Seesaw scheduler.
- `scripts/train.py` is the training entry point.
- `configs/kempner/sweeps/` contains the experiment sweep configurations.
- `reproducibility/` contains pinned W&B run metadata and independent plotting
  code for the paper figures.

## Reproduce the paper figures

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reproducibility/requirements.txt
wandb login
```

Open and run
[`reproducibility/reproduce_figures.ipynb`](reproducibility/reproduce_figures.ipynb)
to reproduce Figures 1 and 2. To reproduce all available figures, including
the large-batch and weight-decay appendix plots, run:

```bash
python reproducibility/reproduce_figures.py
```

The exact W&B run IDs and expected configurations are pinned in
[`reproducibility/run_manifest.json`](reproducibility/run_manifest.json). See
[`reproducibility/README.md`](reproducibility/README.md) for the selection
policy and output details.

## Training environment

```bash
conda create -n seesaw python=3.10
conda activate seesaw
pip install -e ".[all]"
```

## License

This repository is released under the [Apache 2.0 License](LICENSE).
