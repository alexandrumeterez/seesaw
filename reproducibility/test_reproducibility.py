"""Fast, offline integrity checks for the checked-in reproduction artifacts."""

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent


def test_manifest_shape_and_per_figure_run_ids() -> None:
    manifest = json.loads((BASE / "run_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert {figure: len(entries) for figure, entries in manifest["figures"].items()} == {
        "figure_1": 18,
        "figure_2": 10,
        "figure_3": 12,
        "weight_decay": 6,
    }
    for entries in manifest["figures"].values():
        run_ids = [entry["run_id"] for entry in entries]
        assert len(run_ids) == len(set(run_ids))


def test_weight_decay_selection_uses_per_batch_optima() -> None:
    manifest = json.loads((BASE / "run_manifest.json").read_text())
    entries = manifest["figures"]["weight_decay"]
    selected = {
        (entry["initial_batch_size"], entry["role"]): (
            entry["run_id"],
            entry["learning_rate"],
            entry["expected_config"]["optimizer.weight_decay"],
        )
        for entry in entries
    }
    assert selected == {
        (128, "Cosine"): ("dtxikk43", 0.003, 0.0001),
        (128, "Seesaw"): ("048j3qdz", 0.003, 0.0001),
        (256, "Cosine"): ("m9z3gfs9", 0.003, 0.0001),
        (256, "Seesaw"): ("jllk7kdp", 0.003, 0.0001),
        (512, "Cosine"): ("uqdc3049", 0.01, 0.001),
        (512, "Seesaw"): ("n4w0iqig", 0.01, 0.001),
    }


def test_checked_in_endpoints_match_manifest_summaries() -> None:
    manifest = json.loads((BASE / "run_manifest.json").read_text())
    expected = {
        (figure, entry["run_id"]): entry["final_validation_loss"]
        for figure, entries in manifest["figures"].items()
        for entry in entries
    }
    endpoints = pd.read_csv(BASE / "outputs" / "plotted_endpoints.csv")
    assert len(endpoints) == 46
    for row in endpoints.itertuples(index=False):
        assert abs(row.final_validation_loss - expected[(row.figure, row.run_id)]) < 1e-9


def test_all_rendered_outputs_exist() -> None:
    stems = (
        "figure_1_main",
        "figure_2_equivalence",
        "figure_3_large_batches",
        "appendix_weight_decay",
    )
    for stem in stems:
        for suffix in (".pdf", ".png"):
            path = BASE / "outputs" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 10_000
