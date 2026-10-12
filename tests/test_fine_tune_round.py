import json

import pytest
import yaml

from src.modeling import evaluate, fine_tune_round, train

REFERENCE = {
    "learning_rate": 0.001,
    "steps": 1000,
    "batch_size": 32,
    "micro_batch_size": 16,
    "seed": 7,
    "window_start": 0,
    "checkpoint_every": 250,
    "device": "cuda",
    "max_seconds": 1800,
}


@pytest.fixture
def params(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "fine_tuning": REFERENCE,
                "fine_tuning_options": {
                    "1-reference": {},
                    "2-lower-lr": {"learning_rate": 0.0001},
                },
            }
        )
    )
    return path


def test_an_option_changes_only_its_own_settings_and_the_seed_is_per_run(params):
    settings = train.FineTuneConfig.from_params(params, "2-lower-lr", seed=9)
    assert settings.learning_rate == 0.0001 and settings.seed == 9
    assert settings.steps == REFERENCE["steps"]
    assert train.FineTuneConfig.from_params(params).seed == REFERENCE["seed"]


def test_unknown_options_and_seeds_inside_options_are_rejected(params):
    with pytest.raises(ValueError, match="Unknown option"):
        train.FineTuneConfig.from_params(params, "9-missing")
    values = yaml.safe_load(params.read_text())
    values["fine_tuning_options"]["2-lower-lr"]["seed"] = 8
    params.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="must not set the seed"):
        train.FineTuneConfig.from_params(params, "2-lower-lr")


def test_round_scores_every_run_and_continues_after_a_failure(params, tmp_path, monkeypatch):
    def fake_fine_tune(*, params_path, tracking_uri, option, seed):
        if (option, seed) == ("2-lower-lr", 7):
            raise FloatingPointError("Non-finite training loss at step 3")
        return {
            "run_id": f"train-{option}-{seed}",
            "checkpoint": {"path": f"ckpt/{option}-{seed}"},
        }

    scored = []

    def fake_evaluate(*, params_path, method, tracking_uri, checkpoint, device):
        scored.append((method, checkpoint.as_posix(), device))
        return {
            "results": {
                "chronos": {"run_id": f"eval-{checkpoint.name}", "metrics": {"mean.mase": 0.5}}
            }
        }

    monkeypatch.setattr(train, "fine_tune", fake_fine_tune)
    monkeypatch.setattr(evaluate, "evaluate", fake_evaluate)
    results = fine_tune_round.run_round(
        ["1-reference", "2-lower-lr"], [7, 8], params_path=params, summary_dir=tmp_path / "rounds"
    )

    assert [(r["option"], r["seed"]) for r in results] == [
        ("1-reference", 7),
        ("1-reference", 8),
        ("2-lower-lr", 7),
        ("2-lower-lr", 8),
    ]
    assert results[2]["error"] == "FloatingPointError: Non-finite training loss at step 3"
    assert "evaluation_run_id" not in results[2]
    assert results[3]["evaluation_run_id"] == "eval-2-lower-lr-8"
    assert scored[0] == ("chronos", "ckpt/1-reference-7", "cuda")
    (summary,) = (tmp_path / "rounds").iterdir()
    assert json.loads(summary.read_text()) == results


def test_round_starts_no_run_after_its_time_budget(params, tmp_path, monkeypatch):
    clock = iter([0.0, 0.0, 7200.0])
    monkeypatch.setattr(fine_tune_round.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        train, "fine_tune", lambda **kwargs: {"run_id": "t", "checkpoint": {"path": "ckpt"}}
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate",
        lambda **kwargs: {"results": {"chronos": {"run_id": "e", "metrics": {"mean.mase": 0.5}}}},
    )
    results = fine_tune_round.run_round(
        ["1-reference"], [7, 8], params_path=params, summary_dir=tmp_path, max_hours=1.0
    )
    assert results[0]["evaluation_run_id"] == "e"
    assert results[1] == {"option": "1-reference", "seed": 8, "skipped": "time budget"}
    # Skipped runs are in the summary file too, not only in the printed result.
    (summary,) = tmp_path.glob("round-*.json")
    assert json.loads(summary.read_text()) == results


def test_unknown_option_stops_the_round_before_any_training(params, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "fine_tune", lambda **kwargs: pytest.fail("trained"))
    with pytest.raises(ValueError, match="Unknown option"):
        fine_tune_round.run_round(["9-missing"], [7], params_path=params, summary_dir=tmp_path)
