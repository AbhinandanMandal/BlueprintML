"""Default configuration."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:
    # ---- data ----------------------------------------
    data_dir: str = "UPDATE_WITH_DATA_DIRECTORY"            
    image_size: int = 128           # transforms.Resize((128, 128))
    train_size: int = 8000
    val_size: int = 1000
    batch_size: int = 32
    num_workers: int = 2
    split_seed: int = 0            
    # ---- optimisation / loss ------------------------------------
    epochs: int = 20
    adv_start_epoch: int = 2        # epochs 1..2 reconstruction only; adversarial from epoch 3
    lambda_rec: float = 0.999
    lambda_adv: float = 0.001
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    mask_kinds: List[str] = field(default_factory=lambda: ["central", "block", "region"])
    max_steps: Optional[int] = None  # None = full epoch; small int for smoke tests

    # ---- reproducibility ----------------------------------------
    seed: int = 0                   # torch.manual_seed(0)
    mask_seed: int = 0              # np.random.default_rng(0) for training masks

    # ---- evaluation / visualisation -----------------------------
    eval_seed: int = 123
    eval_max_batches: int = 8
    vis_every: int = 2
    vis_seed: int = 7

    # ---- output -----------------------------------------------------------
    output_dir: str = "outputs"    
    ckpt_name: str = "context_encoder_last.pt"
