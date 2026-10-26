import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simclr.config import TrainConfig  # noqa: E402


@pytest.fixture
def make_cfg(tmp_path):
    """Factory for tiny CPU-friendly configs on the offline synthetic dataset."""

    def _make(**overrides) -> TrainConfig:
        base = dict(dataset="synthetic", data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "run"),
                    train_subset_size=64, eval_subset_size=32, batch_size=16, epochs=2, warmup_epochs=0.5,
                    num_workers=0, projection_hidden_dim=256, device="cpu", amp=False, seed=0)
        base.update(overrides)
        return TrainConfig(**base).validate()

    return _make
