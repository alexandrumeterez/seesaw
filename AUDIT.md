# Seesaw reproducibility audit

## Scope and outcome

This audit covered:

- the `continuous-bs-sched` training-code snapshot at
  `42b929a41ae0b638766da9e6bb2786b235d33091`;
- `arxiv.tex` and the rendered paper at paper commit
  `c3fc5a3a19d37238d156713a8f20f7dd9d9e869d`;
- the three exploratory plotting notebooks supplied with the project; and
- the W&B configurations, metadata, logs, summaries, and full history artifacts
  underlying the displayed figures.

The central empirical result survives independent reproduction: for the
displayed Figure 1 cells, Seesaw and cosine reach nearly identical final
validation losses, while the logged optimizer-step count falls by 38.3–38.4%.
The Figure 2 ordering and loss dynamics also reproduce. This is evidence for a
reduction in sequential optimizer steps, not a direct measurement of wall-clock
speedup.

No findings below were repaired in the archival training tree. The independent
plotting package fixes the plotting and selection problems without changing
the run data.

## High-severity training-code findings

1. **Periodic checkpointing is disabled.** Both
   `Trainer.should_save_this_step()` and
   `Trainer.should_save_unsharded_this_step()` return `False` before their real
   implementations (`olmo/train.py:905` and `olmo/train.py:916`). A final
   checkpoint may still be written at normal termination, but periodic
   recovery checkpoints are not.

2. **Seesaw resume is not correct.** `StepCosineHalfWithWarmup` keeps mutable
   `_decay_tokens`, `_current_lr`, `_switch_ptr`, and `_built` state
   (`olmo/optim.py:2094`) that is absent from the trainer checkpoint. During
   restore, `load_trainer_state_dict()` calls `scheduler.get_lr()` before the
   schedule is built (`olmo/train.py:358`); a fresh post-warmup scheduler
   returns `None`. Restoring the dynamic loader also reaches the
   `Trainer.dataset` assertion for `IterableDataset` even though the loader uses
   `SimpleStream`.

3. **The released Seesaw path is single-process only.** `SimpleStream` asserts
   `world_size == 1` (`olmo/data/iterable_dataset.py:51`), while its builder
   passes `rank=0` for every process (`olmo/data/__init__.py:133`). This prevents
   the released implementation from demonstrating the assumed data-parallel
   wall-clock benefit.

4. **Current default/sweep configuration can fail before training.**
   `lr_decay_factor_for_token_counts` defaults to `1.0`
   (`olmo/config.py:595`). `get_decay_tokens()` uses it as the base of a
   logarithm (`olmo/optim.py:2105`), for which `1.0` raises
   `ZeroDivisionError`. The active Seesaw entries in, for example,
   `configs/kempner/sweeps/bsz_sweeps/150m_bsz_continuous.yaml` omit the field.

## Schedule correctness and robustness

1. **Batch and learning-rate transitions are one batch out of phase.** The
   batch scheduler checks its token counter before forming a batch, then
   advances the counter after the batch is formed
   (`olmo/data/iterable_dataset.py:267`). The trainer increments its token
   counter before `train_step()` and LR scheduling (`olmo/train.py:1157`).
   Consequently, the batch that crosses a milestone uses the old batch size
   with the new learning rate; the larger batch begins on the next update.

2. **Only one crossed milestone is consumed per call.** `FastBSScheduler.tick()`
   uses a single `if` (`olmo/data/iterable_dataset.py:35`) rather than consuming
   all milestones at or below the current token count. A sufficiently large
   batch can therefore delay subsequent increases.

3. **The LR scheduler mutates once per optimizer parameter group.** The trainer
   calls the stateful `get_lr()` inside a parameter-group loop
   (`olmo/train.py:730`). If a token jump spans more than one milestone, groups
   can receive different learning rates during the same optimizer step.

4. **Batch-size rounding is implementation-specific and `max_bs` is unused.**
   The code special-cases values whose floored batch size is `1 mod 32`
   (`olmo/data/iterable_dataset.py:39`), rather than implementing a documented
   general rounding rule. The constructor accepts `max_bs` but never applies it.

5. **The dynamic loader wires the wrong option.** Its `persistent_workers`
   argument is assigned `prefetch_factor` instead of the configured boolean
   (`olmo/data/__init__.py:138`). The builder's `decay_steps` parameter is also
   unused.

6. **The stopping condition can overshoot the token budget.** A batch is formed
   before the trainer checks `global_train_tokens_seen >= max_tokens`; dynamic
   batches therefore process up to one extra batch.

7. **No Seesaw-specific tests are present.** The existing optimizer, iterable
   dataset, and configuration tests all passed (15 tests), but none exercises
   the dynamic batch scheduler, boundary alignment, resume, or distributed
   execution.

## Configuration and maintenance findings

- `configs/kempner/sweeps/weight_decay/150m_weight_decay_fixed_seesaw.yaml`
  is byte-for-byte identical to the cosine sweep and does not configure
  Seesaw.
- The committed weight-decay sweeps contain only `1e-6`, `1e-5`, and `1e-4`,
  although the paper says the sweep extended through `1.0`.
- With `decouple_weight_decay: true`, the code passes
  `weight_decay / learning_rate` to AdamW (`olmo/optim.py:2338`). Thus a
  configured `1e-4` at LR `0.003` becomes an optimizer coefficient of about
  `0.0333`, producing a per-step multiplicative decay of `1 - 1e-4`. This is
  internally deliberate but should be documented because it differs from the
  usual interpretation of the AdamW configuration value.
- The `shampoo_orig` optimizer branch still calls `ShampooOrig`
  (`olmo/optim.py:2428`) although that implementation/import is absent.
- Hot-loop debug prints and unused `pdb` imports remain. W&B output logs for the
  audited runs range into tens or hundreds of megabytes.

## W&B and plotting findings

1. **The supplied notebooks contain a plaintext W&B API key.** The key is not
   present in this repository. It should be revoked/rotated.

2. **The original selection is not stable.** The notebooks query mutable groups,
   infer model size from the first four characters of the run name, concatenate
   repeated cells, and use the final row without a deterministic ordering rule.
   They do not pin run IDs or validate configurations.

3. **Histories were sampled and duplicates were implicitly aggregated.** The
   notebooks use the default `run.history()` sampling and Seaborn `lineplot`,
   whose default estimator averages repeated traces and adds an error band.
   The main group contained 185 runs (179 finished, 5 failed, 1 crashed); 103
   matched the plotting filters, representing 95 distinct cells and 8 exact
   duplicate cells.

4. **The recorded Git commit is not a complete source snapshot.** Relevant runs
   report three commits (`ff609819…`, `65e76eba…`, and `da0730ea…`), but output
   logs from runs reporting `ff609819…` contain `INSIDE WHILE LOOP` and
   `batch_by_scheduler` markers absent from that commit and added later. The
   runs were made from a dirty worktree, and W&B did not retain its diff. Exact
   training source therefore cannot be recovered from Git metadata alone.

5. **One weight-decay cell has duplicate-run ambiguity.** The paper's 512M
   Seesaw value `3.0588` matches older run `0whdwu56`; the latest repeated cell,
   `91aqogok`, ends at `3.059374`. The original plotting code averages repeated
   traces, while the table effectively selects the older run.

The replacement workflow in `reproducibility/` pins every run ID, validates its
state/group/commit/configuration, reads full W&B history artifacts, plots each
trace exactly once, and records all plotted endpoints.

## Paper findings

- The empirical panels use optimizer steps on the lower x-axis. No measured
  wall-clock data is plotted, and the audited runs used one GPU with larger
  batches implemented through microbatch accumulation. Statements that the
  experiments directly demonstrate a 36% wall-clock reduction should instead
  describe a sequential-step reduction under sufficient parallel scaling.
- Figure 1's caption says the top “plots” are logarithmic, but only the token
  x-axis is logarithmic; validation loss is linear.
- `\label{sec:exp_details}` appears after a table rather than on a section or
  subsection, so the Figure 1 “Section” cross-reference is attached to the
  wrong counter. The existing `sec:empirical_findings` label is the appropriate
  target.
- The weight-decay table caption says “weight decay 0.003”; the experiments use
  learning rate `0.003` and configured weight decay `0.0001`.
- The paper states exact `D = 20N` Chinchilla scaling, while the configured
  nominal model sizes and token budgets are approximate (for example, 571M
  non-embedding parameters with 13.2B tokens is about 23.1 tokens/parameter).
- Algorithm 1 is expressed in total steps and step milestones, whereas the
  implementation and equal-data comparison operate in tokens and terminate at
  a token budget. It should define a token budget and token milestones
  explicitly.
- The Adam and NSGD equations assign `theta_t` on both sides; the left side
  should be `theta_{t+1}`. Minor prose typos include “will a single,”
  “comapre,” and “agressive.”
- The reported curves are single-seed results and do not quantify uncertainty.

## Reproduction artifacts

- `reproducibility/run_manifest.json`: immutable run selection and expected
  configuration.
- `reproducibility/reproduce_figures.py`: clean-room plotting implementation.
- `reproducibility/reproduce_figures.ipynb`: executable Figures 1 and 2
  notebook.
- `reproducibility/outputs/plotted_endpoints.csv`: exact plotted endpoints.
- `reproducibility/outputs/figure_1_step_reductions.csv`: logged step-count
  comparison.

The cached W&B history artifacts are intentionally excluded from Git. Running
the notebook or script downloads them from W&B after authentication.
