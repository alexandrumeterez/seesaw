#!/usr/bin/env python3
"""Reproduce the Seesaw paper figures from pinned W&B runs.

Unlike the exploratory notebooks used during paper development, this module:

* consumes immutable W&B run IDs from ``run_manifest.json``;
* validates logged configurations before accepting any run;
* downloads full, unsampled W&B history artifacts;
* plots each trace directly (no implicit aggregation or confidence intervals);
* writes a machine-readable table of the exact plotted endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

VALIDATION = "eval/c4_val/CrossEntropyLoss"
TOKENS = "throughput/total_tokens"
STEPS = "total_batches_seen"
BATCH_SIZE = "batch_size"
LEARNING_RATE = "optim/learning_rate_group0"
HISTORY_COLUMNS = ("_step", VALIDATION, TOKENS, STEPS, BATCH_SIZE, LEARNING_RATE)

MODEL_ORDER = ("150M", "300M", "600M")
BATCH_COLORS = {
    128: "#1f77b4",
    256: "#ff7f0e",
    512: "#2ca02c",
    1024: "#d62728",
}
SCHEDULER_STYLES = {"Cosine": "-", "Seesaw": ":", "Fixed LR": "--"}


def nested_get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-7, abs_tol=1e-10)
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_run(run: Any, entry: Mapping[str, Any]) -> None:
    problems: list[str] = []
    if run.id != entry["run_id"]:
        problems.append(f"run ID: {run.id!r} != {entry['run_id']!r}")
    if run.state != "finished":
        problems.append(f"state is {run.state!r}, expected 'finished'")
    if run.group != entry["group"]:
        problems.append(f"group: {run.group!r} != {entry['group']!r}")

    logged_commit = ((run.metadata or {}).get("git") or {}).get("commit")
    if logged_commit != entry.get("git_commit"):
        problems.append(f"Git commit: {logged_commit!r} != {entry.get('git_commit')!r}")

    for path, expected in entry["expected_config"].items():
        actual = nested_get(run.config, path)
        if not values_match(actual, expected):
            problems.append(f"{path}: {actual!r} != {expected!r}")

    if problems:
        detail = "\n  - ".join(problems)
        raise RuntimeError(f"W&B run {run.id} failed validation:\n  - {detail}")


def history_artifact(run: Any) -> Any:
    artifacts = [artifact for artifact in run.logged_artifacts() if artifact.type == "wandb-history"]
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected one history artifact for run {run.id}, found {len(artifacts)}")
    return artifacts[0]


def load_history(run: Any, cache_dir: Path) -> pd.DataFrame:
    run_cache = cache_dir / run.id
    parquet_files = sorted(run_cache.glob("**/*.parquet"))
    if not parquet_files:
        run_cache.mkdir(parents=True, exist_ok=True)
        artifact = history_artifact(run)
        downloaded = Path(artifact.download(root=run_cache))
        parquet_files = sorted(downloaded.glob("**/*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No Parquet history files found for run {run.id}")

    frames: list[pd.DataFrame] = []
    for parquet_file in parquet_files:
        available = set(pd.read_parquet(parquet_file, columns=[]).columns)
        # PyArrow returns an empty column index for columns=[] in some versions.
        if not available:
            import pyarrow.parquet as pq

            available = set(pq.read_schema(parquet_file).names)
        columns = [column for column in HISTORY_COLUMNS if column in available]
        frames.append(pd.read_parquet(parquet_file, columns=columns))
    history = pd.concat(frames, ignore_index=True)
    missing = {VALIDATION, TOKENS, STEPS} - set(history.columns)
    if missing:
        raise RuntimeError(f"Run {run.id} history is missing {sorted(missing)}")

    evaluations = history.dropna(subset=[VALIDATION, TOKENS, STEPS]).copy()
    evaluations = evaluations.sort_values([TOKENS, "_step"])
    evaluations = evaluations.drop_duplicates(subset=["_step"], keep="last")
    if evaluations.empty:
        raise RuntimeError(f"Run {run.id} has no complete evaluation rows")
    return evaluations


def load_figure_data(
    api: wandb.Api,
    manifest: Mapping[str, Any],
    figure: str,
    cache_dir: Path,
) -> dict[str, tuple[Mapping[str, Any], pd.DataFrame]]:
    entity = manifest["wandb"]["entity"]
    project = manifest["wandb"]["project"]
    result: dict[str, tuple[Mapping[str, Any], pd.DataFrame]] = {}
    for entry in manifest["figures"][figure]:
        run = api.run(f"{entity}/{project}/{entry['run_id']}")
        validate_run(run, entry)
        result[entry["run_id"]] = (entry, load_history(run, cache_dir))
    return result


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.grid": True,
            "grid.color": "#b0b0b0",
            "grid.linewidth": 0.6,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def add_tail_inset(
    parent: mpl.axes.Axes,
    traces: Sequence[tuple[Mapping[str, Any], pd.DataFrame]],
    threshold: float,
    location: str,
) -> None:
    borderpad = 2 if location == "lower left" else 1
    inset = inset_axes(
        parent,
        width="42%",
        height="38%",
        loc=location,
        borderpad=borderpad,
    )
    plotted = False
    for entry, history in traces:
        tail = history[history[TOKENS] >= threshold]
        if tail.empty:
            continue
        inset.plot(
            tail[TOKENS],
            tail[VALIDATION],
            color=BATCH_COLORS[entry["initial_batch_size"]],
            linestyle=SCHEDULER_STYLES[entry["role"]],
            linewidth=1.7,
        )
        plotted = True
    if not plotted:
        inset.remove()
        return
    inset.tick_params(axis="both", labelsize=8)
    inset.tick_params(axis="x", which="both", labelbottom=False)
    inset.xaxis.get_offset_text().set_visible(False)
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.grid(True)
    connector_corners = (1, 3) if location == "lower left" else (2, 4)
    mark_inset(
        parent,
        inset,
        loc1=connector_corners[0],
        loc2=connector_corners[1],
        fc="none",
        ec="0.65",
        linewidth=0.8,
    )


def plot_figure_1(
    data: Mapping[str, tuple[Mapping[str, Any], pd.DataFrame]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.2))
    inset_locations = {"150M": "lower left", "300M": "upper right", "600M": "upper right"}

    for column, model in enumerate(MODEL_ORDER):
        traces = [pair for pair in data.values() if pair[0]["model"] == model]
        traces.sort(
            key=lambda pair: (
                pair[0]["initial_batch_size"],
                0 if pair[0]["role"] == "Cosine" else 1,
            )
        )
        for entry, history in traces:
            color = BATCH_COLORS[entry["initial_batch_size"]]
            linestyle = SCHEDULER_STYLES[entry["role"]]
            axes[0, column].plot(
                history[TOKENS],
                history[VALIDATION],
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
            )
            axes[1, column].plot(
                history[STEPS],
                history[VALIDATION],
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
            )

        axes[0, column].set_xlim(left=1e9)
        axes[0, column].set_title(model, fontsize=18)
        axes[0, column].set_xlabel("Tokens")
        axes[0, column].set_ylim(2.65, 3.5)
        axes[1, column].set_xlabel("Steps")
        axes[1, column].set_ylim(2.65, 3.5)
        final_tokens = max(float(history[TOKENS].max()) for _, history in traces)
        add_tail_inset(
            axes[0, column],
            traces,
            0.9 * final_tokens,
            inset_locations[model],
        )
        if column == 0:
            axes[0, column].set_ylabel("Validation Loss")
            axes[1, column].set_ylabel("Validation Loss")
        else:
            axes[0, column].set_ylabel("")
            axes[1, column].set_ylabel("")

    batch_handles = [
        Line2D([0], [0], color=BATCH_COLORS[batch], lw=3, label=str(batch)) for batch in (128, 256, 512, 1024)
    ]
    scheduler_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=3,
            linestyle=SCHEDULER_STYLES[role],
            label=role,
        )
        for role in ("Cosine", "Seesaw")
    ]
    handles = (
        [Line2D([], [], linestyle="none", label="Batch size:")]
        + batch_handles
        + [Line2D([], [], linestyle="none", label="Scheduler:")]
        + scheduler_handles
    )
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="upper center",
        ncol=9,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.08,
        top=0.88,
        wspace=0.17,
        hspace=0.28,
    )
    save_figure(fig, output_dir, "figure_1_main")


def alpha_beta_label(entry: Mapping[str, Any]) -> str:
    alpha = entry["alpha"]
    beta = entry["beta"]
    labels = {
        (2.0, 1.0): r"$\alpha=2,\ \beta=1$",
        (2 ** (3 / 4), math.sqrt(2)): r"$\alpha=2^{3/4},\ \beta=\sqrt{2}$",
        (math.sqrt(2), 2.0): r"$\alpha=\sqrt{2},\ \beta=2$",
        (2 ** (1 / 4), 2 ** (3 / 2)): r"$\alpha=2^{1/4},\ \beta=2^{3/2}$",
        (1.0, 4.0): r"$\alpha=1,\ \beta=4$",
    }
    for (expected_alpha, expected_beta), label in labels.items():
        if math.isclose(alpha, expected_alpha) and math.isclose(beta, expected_beta):
            return label
    return entry["label"]


def plot_figure_2(
    data: Mapping[str, tuple[Mapping[str, Any], pd.DataFrame]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    variants = sorted({entry["alpha"] for entry, _ in data.values()}, reverse=True)
    colors = dict(zip(variants, mpl.colormaps["tab10"].colors[: len(variants)]))

    for axis, batch_size in zip(axes, (256, 512)):
        traces = [pair for pair in data.values() if pair[0]["initial_batch_size"] == batch_size]
        traces.sort(key=lambda pair: pair[0]["alpha"], reverse=True)
        for entry, history in traces:
            tail = history[history[TOKENS] >= 1.9e9]
            axis.plot(
                tail[TOKENS],
                tail[VALIDATION],
                color=colors[entry["alpha"]],
                linewidth=2.8,
                label=alpha_beta_label(entry),
            )
        axis.set_title(f"B={batch_size}", fontsize=17)
        axis.set_xlabel("Tokens")
        axis.set_ylabel("Validation Loss")
        axis.ticklabel_format(axis="x", style="sci", scilimits=(9, 9))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, output_dir, "figure_2_equivalence")


def plot_figure_3(
    data: Mapping[str, tuple[Mapping[str, Any], pd.DataFrame]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3))
    axes = axes.ravel()
    role_colors = {
        "Cosine": "#1f77b4",
        "Fixed LR": "#ff7f0e",
        "Seesaw": "#2ca02c",
    }
    for axis, batch_size in zip(axes, (1024, 2048, 4096, 8192)):
        traces = [pair for pair in data.values() if pair[0]["initial_batch_size"] == batch_size]
        traces.sort(key=lambda pair: ("Cosine", "Fixed LR", "Seesaw").index(pair[0]["role"]))
        for entry, history in traces:
            tail = history[history[TOKENS] >= 3e8]
            axis.plot(
                tail[TOKENS],
                tail[VALIDATION],
                color=role_colors[entry["role"]],
                linewidth=2.5,
                label=entry["role"],
            )
        axis.set_xscale("log")
        axis.set_xticks(
            (3e8, 1e9, 3e9),
            labels=(r"$3\times10^8$", r"$10^9$", r"$3\times10^9$"),
        )
        axis.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        axis.set_title(f"B={batch_size}", fontsize=16)
        axis.set_xlabel("Tokens")
        axis.set_ylabel("Validation Loss")
        axis.set_ylim((3.1, 4.0) if batch_size <= 2048 else (3.5, 6.0))
    handles = [Line2D([0], [0], color=color, lw=3, label=role) for role, color in role_colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_dir, "figure_3_large_batches")


def plot_weight_decay(
    data: Mapping[str, tuple[Mapping[str, Any], pd.DataFrame]],
    output_dir: Path,
) -> None:
    with mpl.rc_context(
        {
            "axes.grid": False,
            "font.size": 14,
            "grid.linewidth": 0.8,
            "mathtext.fontset": "cm",
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        role_styles = {
            "Seesaw": ("Seesaw", "#1f77b4"),
            "Cosine": ("Cosine Decay", "#ff7f0e"),
        }
        for axis, batch_size in zip(axes, (128, 256, 512)):
            traces = {
                entry["role"]: (entry, history)
                for entry, history in data.values()
                if entry["initial_batch_size"] == batch_size
            }
            for role, (label, color) in role_styles.items():
                _, history = traces[role]
                tail = history[history[TOKENS] > 1e9]
                axis.plot(
                    tail[TOKENS],
                    tail[VALIDATION],
                    color=color,
                    linewidth=3,
                    label=label,
                )
            axis.grid(True, which="both")
            axis.set_xscale("log")
            axis.set_title(f"Batch Size: {batch_size}")
            axis.set_xlabel("Tokens")
            axis.set_ylabel("Validation Loss")
            axis.set_ylim(bottom=3.0)
        axes[0].legend(title="Scheduler")
        fig.tight_layout()
        save_figure(fig, output_dir, "appendix_weight_decay")


def endpoint_rows(figure_data: Mapping[str, Mapping[str, tuple[Mapping[str, Any], pd.DataFrame]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for figure, data in figure_data.items():
        for entry, history in data.values():
            final = history.iloc[-1]
            row = {
                "figure": figure,
                "run_id": entry["run_id"],
                "model": entry["model"],
                "initial_batch_size": entry["initial_batch_size"],
                "learning_rate": entry["learning_rate"],
                "role": entry["role"],
                "final_tokens": int(final[TOKENS]),
                "final_steps": int(final[STEPS]),
                "final_validation_loss": float(final[VALIDATION]),
                "git_commit": entry.get("git_commit"),
            }
            if "alpha" in entry:
                row["alpha"] = entry["alpha"]
                row["beta"] = entry["beta"]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["figure", "model", "initial_batch_size", "role"])


def write_step_reductions(endpoints: pd.DataFrame, output_dir: Path) -> None:
    figure_1 = endpoints[endpoints["figure"] == "figure_1"]
    rows: list[dict[str, Any]] = []
    for (model, batch_size), group in figure_1.groupby(["model", "initial_batch_size"]):
        steps = group.set_index("role")["final_steps"]
        rows.append(
            {
                "model": model,
                "initial_batch_size": batch_size,
                "cosine_steps": int(steps["Cosine"]),
                "seesaw_steps": int(steps["Seesaw"]),
                "step_reduction_fraction": 1 - steps["Seesaw"] / steps["Cosine"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "figure_1_step_reductions.csv", index=False)


PLOTTERS = {
    "figure_1": plot_figure_1,
    "figure_2": plot_figure_2,
    "figure_3": plot_figure_3,
    "weight_decay": plot_weight_decay,
}


def reproduce(
    manifest_path: Path,
    output_dir: Path,
    cache_dir: Path,
    figures: Iterable[str],
) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported run manifest schema")
    configure_matplotlib()
    api = wandb.Api()
    selected = list(figures)
    figure_data = {figure: load_figure_data(api, manifest, figure, cache_dir) for figure in selected}
    for figure, data in figure_data.items():
        PLOTTERS[figure](data, output_dir)
    endpoints = endpoint_rows(figure_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoints.to_csv(output_dir / "plotted_endpoints.csv", index=False)
    if "figure_1" in figure_data:
        write_step_reductions(endpoints, output_dir)
    return endpoints


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=base / "run_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=base / "outputs")
    parser.add_argument("--cache-dir", type=Path, default=base / "cache")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=tuple(PLOTTERS),
        default=list(PLOTTERS),
    )
    args = parser.parse_args()
    endpoints = reproduce(args.manifest, args.output_dir, args.cache_dir, args.figures)
    print(f"Validated and plotted {len(endpoints)} pinned runs into {args.output_dir}")


if __name__ == "__main__":
    main()
