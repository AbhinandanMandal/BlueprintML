import torch

from src.context_encoders.model import ContextEncoder, Discriminator


def test_generator_shapes_params_and_grads():
    G = ContextEncoder()
    assert abs(sum(p.numel() for p in G.parameters()) - 4_343_171) == 0   # value printed by the notebook (4.343171 M)
    x = torch.randn(2, 3, 128, 128)
    assert G.encoder(x).shape == (2, 256, 8, 8)
    out = G(x)
    assert out.shape == (2, 3, 128, 128) and out.dtype == torch.float32
    assert out.min() >= -1 and out.max() <= 1
    out.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in G.parameters())


def test_discriminator_shape_and_grads():
    D = Discriminator()
    out = D(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 1)
    out.sum().backward()
    assert all(p.grad is not None for p in D.parameters())


def test_state_dict_key_names_match_notebook_checkpoints():
    keys = set(ContextEncoder().state_dict())
    assert {"encoder.net.0.0.weight", "decoder.cw_fc.weight", "decoder.cw_fc.bias",
            "decoder.mix.weight", "decoder.up.4.weight"} <= keys
    dkeys = set(Discriminator().state_dict())
    assert {"features.0.0.weight", "features.1.0.weight", "classifier.weight"} <= dkeys
