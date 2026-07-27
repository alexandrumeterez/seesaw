#!/usr/bin/env python3
"""Discover and pin the W&B runs used by the Seesaw paper figures.

This script is intentionally separate from the plotting code. It applies a
documented selection policy once and writes immutable W&B run IDs to
``run_manifest.json``. The plotting notebook consumes only that manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import wandb

ENTITY = "harvardml"
PROJECT = "batch-size-sched"
VALIDATION_METRIC = "eval/c4_val/CrossEntropyLoss"
MAIN_GROUP = "final-bsz-adam-continuous"
LARGE_BATCH_GROUP = "final-large-bsz-adam-continuous"
COSINE_WD_GROUP = "weight-decay-adam-sweep"
SEESAW_WD_GROUP = "seesaw-weight-decay-adam-sweep"

MODEL_BY_ARCHITECTURE = {
    (12, 1024): "150M",
    (24, 1024): "300M",
    (24, 1408): "600M",
}
MAIN_BATCHES = {
    "150M": (128, 256, 512),
    "300M": (128, 256, 512),
    "600M": (256, 512, 1024),
}
LEARNING_RATES = (1e-3, 3e-3, 1e-2, 3e-2)


def nested_get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def close(left: Any, right: float, *, tolerance: float = 1e-7) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class RunRecord:
    run: Any
    model: str
    batch_size: int
    learning_rate: float
    scheduler: str
    batch_factor: float
    learning_rate_factor: float
    token_factor: float | None
    final_validation_loss: float
    commit: str | None

    @classmethod
    def from_run(cls, run: Any) -> RunRecord | None:
        if run.state != "finished" or "redone" in run.name.lower():
            return None
        config = run.config
        model_cfg = config.get("model", {})
        model = MODEL_BY_ARCHITECTURE.get((model_cfg.get("n_layers"), model_cfg.get("d_model")))
        final_loss = run.summary.get(VALIDATION_METRIC)
        if model is None or final_loss is None:
            return None
        scheduler_cfg = config.get("scheduler", {})
        optimizer_cfg = config.get("optimizer", {})
        metadata_git = (run.metadata or {}).get("git") or {}
        return cls(
            run=run,
            model=model,
            batch_size=int(config["global_train_batch_size"]),
            learning_rate=float(optimizer_cfg["learning_rate"]),
            scheduler=str(scheduler_cfg["name"]),
            batch_factor=float(scheduler_cfg.get("batch_size_increase_factor", 1.0)),
            learning_rate_factor=float(scheduler_cfg.get("lr_decay_factor", 1.0)),
            token_factor=(
                None
                if scheduler_cfg.get("lr_decay_factor_for_token_counts") is None
                else float(scheduler_cfg["lr_decay_factor_for_token_counts"])
            ),
            final_validation_loss=float(final_loss),
            commit=metadata_git.get("commit"),
        )


def load_group(api: wandb.Api, group: str) -> list[RunRecord]:
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"group": group, "state": "finished"},
    )
    records = [record for run in runs if (record := RunRecord.from_run(run)) is not None]
    if not records:
        raise RuntimeError(f"No usable runs found in W&B group {group!r}")
    return records


def latest(records: Iterable[RunRecord], predicate: Callable[[RunRecord], bool]) -> RunRecord:
    matches = [record for record in records if predicate(record)]
    if not matches:
        raise RuntimeError("No run matched a required experimental cell")
    return max(matches, key=lambda record: record.run.created_at)


def is_cosine(record: RunRecord) -> bool:
    return record.scheduler == "cosine_with_warmup"


def is_standard_seesaw(record: RunRecord) -> bool:
    return (
        record.scheduler == "step_cosine_half_with_warmup"
        and close(record.batch_factor, 1.1)
        and close(record.learning_rate_factor, 0.953462)
    )


def has_fixed_weight_decay(record: RunRecord) -> bool:
    return nested_get(record.run.config, "scheduler.adaptive_wd", False) is False


def same_cell(
    record: RunRecord,
    *,
    model: str,
    batch_size: int,
    learning_rate: float,
) -> bool:
    return record.model == model and record.batch_size == batch_size and close(record.learning_rate, learning_rate)


def expected_config(record: RunRecord) -> dict[str, Any]:
    config = record.run.config
    expected = {
        "global_train_batch_size": record.batch_size,
        "model.n_layers": nested_get(config, "model.n_layers"),
        "model.d_model": nested_get(config, "model.d_model"),
        "model.max_sequence_length": nested_get(config, "model.max_sequence_length"),
        "optimizer.name": nested_get(config, "optimizer.name"),
        "optimizer.learning_rate": record.learning_rate,
        "optimizer.beta_0": nested_get(config, "optimizer.beta_0"),
        "optimizer.beta_1": nested_get(config, "optimizer.beta_1"),
        "optimizer.eps": nested_get(config, "optimizer.eps"),
        "optimizer.weight_decay": nested_get(config, "optimizer.weight_decay"),
        "optimizer.decouple_weight_decay": nested_get(config, "optimizer.decouple_weight_decay"),
        "scheduler.name": record.scheduler,
        "scheduler.t_warmup": nested_get(config, "scheduler.t_warmup"),
        "scheduler.units": nested_get(config, "scheduler.units"),
        "scheduler.alpha_f": nested_get(config, "scheduler.alpha_f"),
        "scheduler.lr_decay_factor": record.learning_rate_factor,
        "scheduler.batch_size_increase_factor": record.batch_factor,
        "max_duration": config.get("max_duration"),
        "seed": config.get("seed"),
    }
    token_factor = nested_get(config, "scheduler.lr_decay_factor_for_token_counts")
    if token_factor is not None:
        expected["scheduler.lr_decay_factor_for_token_counts"] = token_factor
    return expected


def manifest_entry(
    record: RunRecord,
    *,
    figure: str,
    role: str,
    label: str | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "figure": figure,
        "run_id": record.run.id,
        "run_name": record.run.name,
        "group": record.run.group,
        "state": record.run.state,
        "created_at": record.run.created_at,
        "git_commit": record.commit,
        "model": record.model,
        "initial_batch_size": record.batch_size,
        "learning_rate": record.learning_rate,
        "role": role,
        "final_validation_loss": record.final_validation_loss,
        "expected_config": expected_config(record),
    }
    if label is not None:
        entry["label"] = label
    if alpha is not None:
        entry["alpha"] = alpha
    if beta is not None:
        entry["beta"] = beta
    return entry


def select_figure_1(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for model, batch_sizes in MAIN_BATCHES.items():
        for batch_size in batch_sizes:
            cosine_by_lr: dict[float, RunRecord] = {}
            for learning_rate in LEARNING_RATES:
                try:
                    cosine_by_lr[learning_rate] = latest(
                        records,
                        lambda record, m=model, b=batch_size, lr=learning_rate: (
                            same_cell(record, model=m, batch_size=b, learning_rate=lr) and is_cosine(record)
                        ),
                    )
                except RuntimeError:
                    continue
            if not cosine_by_lr:
                raise RuntimeError(f"No cosine candidates for {model}, batch {batch_size}")
            best_lr, cosine = min(cosine_by_lr.items(), key=lambda item: item[1].final_validation_loss)
            seesaw = latest(
                records,
                lambda record, m=model, b=batch_size, lr=best_lr: (
                    same_cell(record, model=m, batch_size=b, learning_rate=lr) and is_standard_seesaw(record)
                ),
            )
            selected.append(manifest_entry(cosine, figure="figure_1", role="Cosine"))
            selected.append(manifest_entry(seesaw, figure="figure_1", role="Seesaw"))
    return selected


def select_figure_2(records: Sequence[RunRecord], figure_1: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best_lrs = {
        (entry["model"], entry["initial_batch_size"]): entry["learning_rate"]
        for entry in figure_1
        if entry["role"] == "Cosine"
    }
    variants = (
        (2.0, 1.0, 0.5, 1.0),
        (2 ** (3 / 4), math.sqrt(2), 2 ** (-3 / 4), math.sqrt(2)),
        (math.sqrt(2), 2.0, 1 / math.sqrt(2), 2.0),
        (2 ** (1 / 4), 2 ** (3 / 2), 2 ** (-1 / 4), 2 ** (3 / 2)),
        (1.0, 4.0, 1.0, 4.0),
    )
    selected: list[dict[str, Any]] = []
    for batch_size in (256, 512):
        learning_rate = best_lrs[("150M", batch_size)]
        for alpha, beta, lr_factor, batch_factor in variants:
            record = latest(
                records,
                lambda candidate, b=batch_size, lr=learning_rate, lrf=lr_factor, bf=batch_factor: (
                    same_cell(candidate, model="150M", batch_size=b, learning_rate=lr)
                    and candidate.scheduler == "step_cosine_half_with_warmup"
                    and close(candidate.learning_rate_factor, lrf)
                    and close(candidate.batch_factor, bf)
                    and close(candidate.token_factor, 0.5)
                ),
            )
            label = rf"$\alpha={alpha:.6g},\ \beta={beta:.6g}$"
            selected.append(
                manifest_entry(
                    record,
                    figure="figure_2",
                    role="equivalence_variant",
                    label=label,
                    alpha=alpha,
                    beta=beta,
                )
            )
    return selected


def select_figure_3(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    def scheduler_role(record: RunRecord) -> str | None:
        if is_cosine(record):
            return "Cosine"
        if (
            record.scheduler == "step_cosine_half_with_warmup"
            and close(record.batch_factor, 1.1)
            and close(record.learning_rate_factor, 1.0)
            and close(record.token_factor, 1 / 1.1)
        ):
            return "Fixed LR"
        if is_standard_seesaw(record) and close(record.token_factor, 1 / 1.1):
            return "Seesaw"
        return None

    for batch_size in (1024, 2048, 4096, 8192):
        cosine_by_lr: dict[float, RunRecord] = {}
        for learning_rate in LEARNING_RATES:
            try:
                cosine_by_lr[learning_rate] = latest(
                    records,
                    lambda record, b=batch_size, lr=learning_rate: (
                        same_cell(record, model="150M", batch_size=b, learning_rate=lr)
                        and scheduler_role(record) == "Cosine"
                    ),
                )
            except RuntimeError:
                continue
        best_lr, cosine = min(cosine_by_lr.items(), key=lambda item: item[1].final_validation_loss)
        selected.append(manifest_entry(cosine, figure="figure_3", role="Cosine"))
        for role in ("Fixed LR", "Seesaw"):
            record = latest(
                records,
                lambda candidate, b=batch_size, lr=best_lr, r=role: (
                    same_cell(candidate, model="150M", batch_size=b, learning_rate=lr)
                    and scheduler_role(candidate) == r
                ),
            )
            selected.append(manifest_entry(record, figure="figure_3", role=role))
    return selected


def select_weight_decay(
    cosine_records: Sequence[RunRecord],
    seesaw_records: Sequence[RunRecord],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for batch_size in (128, 256, 512):
        cosine_by_pair: dict[tuple[float, float], RunRecord] = {}
        for record in cosine_records:
            if (
                record.model != "150M"
                or record.batch_size != batch_size
                or not is_cosine(record)
                or not has_fixed_weight_decay(record)
            ):
                continue
            pair = (
                record.learning_rate,
                float(nested_get(record.run.config, "optimizer.weight_decay")),
            )
            previous = cosine_by_pair.get(pair)
            if previous is None or record.run.created_at > previous.run.created_at:
                cosine_by_pair[pair] = record
        if not cosine_by_pair:
            raise RuntimeError(f"No fixed-weight-decay cosine runs for batch {batch_size}")
        cosine = min(
            cosine_by_pair.values(),
            key=lambda record: record.final_validation_loss,
        )
        weight_decay = float(nested_get(cosine.run.config, "optimizer.weight_decay"))
        seesaw = latest(
            seesaw_records,
            lambda record, b=batch_size, lr=cosine.learning_rate, wd=weight_decay: (
                same_cell(record, model="150M", batch_size=b, learning_rate=lr)
                and close(nested_get(record.run.config, "optimizer.weight_decay"), wd)
                and has_fixed_weight_decay(record)
                and is_standard_seesaw(record)
            ),
        )
        selected.append(manifest_entry(cosine, figure="weight_decay", role="Cosine"))
        selected.append(manifest_entry(seesaw, figure="weight_decay", role="Seesaw"))
    return selected


def build_manifest(api: wandb.Api) -> dict[str, Any]:
    main = load_group(api, MAIN_GROUP)
    large = load_group(api, LARGE_BATCH_GROUP)
    cosine_wd = load_group(api, COSINE_WD_GROUP)
    seesaw_wd = load_group(api, SEESAW_WD_GROUP)

    figure_1 = select_figure_1(main)
    figure_2 = select_figure_2(main, figure_1)
    # The supplied large-batch notebook sourced the B=1024 standard Seesaw
    # trace from the main group and the remaining traces from the large-batch
    # group, so discovery intentionally considers both groups here.
    figure_3 = select_figure_3([*main, *large])
    weight_decay = select_weight_decay(cosine_wd, seesaw_wd)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wandb": {"entity": ENTITY, "project": PROJECT},
        "validation_metric": VALIDATION_METRIC,
        "selection_policy": {
            "common": (
                "Finished runs only; exclude names containing 'redone'; identify model "
                "size from architecture; choose the most recently created run when an "
                "experimental cell was repeated."
            ),
            "figure_1": (
                "For each displayed model and initial batch size, choose the learning "
                "rate with the lowest final cosine validation loss, then use the same "
                "learning rate for Seesaw."
            ),
            "figure_2": (
                "Use the Figure 1 cosine-selected learning rate for the 150M model at "
                "batch sizes 256 and 512, then pin all five alpha/beta variants."
            ),
            "figure_3": (
                "For each initial batch size, choose the learning rate with the lowest "
                "final cosine validation loss, then use it for Fixed LR and Seesaw."
            ),
            "weight_decay": (
                "At each initial batch size, choose the fixed-weight-decay learning-rate "
                "and weight-decay pair with the lowest final cosine validation loss, then "
                "use the same pair for fixed-weight-decay Seesaw."
            ),
        },
        "figures": {
            "figure_1": figure_1,
            "figure_2": figure_2,
            "figure_3": figure_3,
            "weight_decay": weight_decay,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("run_manifest.json"),
    )
    args = parser.parse_args()
    manifest = build_manifest(wandb.Api())
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    counts = {figure: len(entries) for figure, entries in manifest["figures"].items()}
    print(f"Wrote {args.output} with pinned run counts: {counts}")


if __name__ == "__main__":
    main()
