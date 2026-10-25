"""Seeding, device selection and checkpoint helpers."""
import os
import tempfile

import torch


def set_seed(seed: int = 0) -> None:
    """Notebook behaviour: ``torch.manual_seed(seed)`` (masks use their own numpy Generator)."""
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def save_checkpoint(path, G, D, epoch: int) -> None:
    """Save ``{"G", "D", "epoch"}`` - the exact format written by the notebook. Write is atomic."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    os.close(fd)
    try:
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch}, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_checkpoint(path, G, D=None, map_location="cpu", strict: bool = True) -> int:
    """Load a checkpoint into ``G`` (and ``D`` if given); returns the stored epoch.

    Uses ``weights_only=True`` so arbitrary pickled code in an untrusted file is not executed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: '{path}'.")
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "G" not in ckpt:
        raise KeyError(f"'{path}' has no 'G' entry; found keys {list(ckpt)}.")
    G.load_state_dict(ckpt["G"], strict=strict)
    if D is not None:
        if "D" not in ckpt:
            raise KeyError(f"'{path}' has no 'D' entry; found keys {list(ckpt)}.")
        D.load_state_dict(ckpt["D"], strict=strict)
    return int(ckpt.get("epoch", 0))
