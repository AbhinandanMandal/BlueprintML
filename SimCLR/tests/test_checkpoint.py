import csv
import json

import pytest
import torch

from simclr.checkpoint import (check_resume_compatibility, load_checkpoint, load_model_for_eval,
                               save_checkpoint)
from simclr.engine import fit
from simclr.utils import get_rng_state, set_rng_state
import random
import numpy as np


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One tiny 2-epoch CPU run on synthetic data shared by the tests below."""
    from simclr.config import TrainConfig
    tmp = tmp_path_factory.mktemp("ckpt")
    cfg = TrainConfig(dataset="synthetic", data_dir=str(tmp / "data"), output_dir=str(tmp / "run"),
                      train_subset_size=64, eval_subset_size=32, batch_size=16, epochs=2, warmup_epochs=0.5,
                      num_workers=0, projection_hidden_dim=256, device="cpu", amp=False, seed=0).validate()
    summary = fit(cfg)
    return cfg, summary, tmp / "run"


def test_fit_writes_expected_artifacts(trained):
    cfg, summary, out = trained
    for name in ("last_checkpoint.pt", "best_checkpoint.pt", "metrics.csv", "config.json", "environment.json",
                 "summary.json"):
        assert (out / name).is_file(), name
    assert not list(out.glob("*.tmp"))  # atomic writes leave no temp files
    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2"] and rows[-1]["global_step"] == "8"
    assert float(rows[-1]["knn_acc"]) >= 0.0 and float(rows[0]["images_per_sec"]) > 0
    assert json.loads((out / "config.json").read_text())["seed"] == 0
    assert summary["epochs_completed"] == 2 and torch.isfinite(torch.tensor(summary["final_train_loss"]))


def test_checkpoint_contents_and_model_restoration(trained):
    cfg, _, out = trained
    ckpt = load_checkpoint(out / "last_checkpoint.pt")
    for key in ("encoder", "projector", "optimizer", "scheduler", "scaler", "epoch", "global_step", "best_metric",
                "config", "seed", "rng_state", "environment"):
        assert key in ckpt, key
    assert ckpt["epoch"] == 2 and ckpt["global_step"] == 8
    model, loaded_cfg = load_model_for_eval(out / "last_checkpoint.pt", torch.device("cpu"))
    assert loaded_cfg.projection_hidden_dim == 256 and not model.training
    for k, v in ckpt["encoder"].items():
        assert torch.equal(model.encoder.state_dict()[k], v)


def test_resume_completed_run_is_a_noop(trained):
    cfg, _, out = trained
    before = (out / "last_checkpoint.pt").stat().st_mtime_ns
    cfg2 = TrainConfig_like(cfg, resume=str(out / "last_checkpoint.pt"))
    summary = fit(cfg2)
    assert summary["epochs_completed"] == 2 and (out / "last_checkpoint.pt").stat().st_mtime_ns == before


def TrainConfig_like(cfg, **kw):
    d = cfg.to_dict()
    d.update(kw)
    return type(cfg)(**d).validate()


def test_incompatible_resume_is_rejected(trained):
    cfg, _, out = trained
    ckpt_cfg = load_checkpoint(out / "last_checkpoint.pt")["config"]
    with pytest.raises(ValueError, match="encoder|temperature|batch_size"):
        check_resume_compatibility(ckpt_cfg, TrainConfig_like(cfg, temperature=0.2), False)
    with pytest.raises(ValueError, match="--extend-schedule"):
        check_resume_compatibility(ckpt_cfg, TrainConfig_like(cfg, epochs=3), False)
    check_resume_compatibility(ckpt_cfg, TrainConfig_like(cfg, epochs=3), True)  # allowed when extending
    with pytest.raises(ValueError):
        fit(TrainConfig_like(cfg, resume=str(out / "last_checkpoint.pt"), batch_size=8))


def test_resume_and_extend_schedule_continues_from_saved_step(trained):
    cfg, _, out = trained
    cfg3 = TrainConfig_like(cfg, epochs=3, resume="auto", extend_schedule=True)
    summary = fit(cfg3)
    ckpt = load_checkpoint(out / "last_checkpoint.pt")
    assert ckpt["epoch"] == 3 and ckpt["global_step"] == 12 and summary["epochs_completed"] == 3
    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]  # log appended, not overwritten
    assert ckpt["scheduler"]["step_count"] == 12


def test_resume_missing_checkpoint_raises(trained, tmp_path):
    cfg, _, _ = trained
    with pytest.raises(FileNotFoundError):
        fit(TrainConfig_like(cfg, output_dir=str(tmp_path / "other"), resume=str(tmp_path / "nope.pt")))


def test_corrupt_checkpoint_gives_actionable_error(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="Could not read checkpoint"):
        load_checkpoint(bad)


def test_atomic_save_roundtrip(tmp_path):
    save_checkpoint(tmp_path / "x.pt", {"a": torch.arange(3)})
    assert torch.equal(torch.load(tmp_path / "x.pt", weights_only=False)["a"], torch.arange(3))
    assert not list(tmp_path.glob("*.tmp"))


def test_rng_state_roundtrip():
    state = get_rng_state()
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_rng_state(state)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b
