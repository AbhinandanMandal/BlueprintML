import pytest
import torch
import torch.nn as nn

from simclr.config import TrainConfig
from simclr.models import FEATURE_DIMS, ProjectionHead, build_encoder, build_model


@pytest.mark.parametrize("name", ["resnet18", "resnet50"])
def test_encoder_returns_features_not_logits(name):
    enc, dim = build_encoder(name, cifar_stem=True)
    assert dim == FEATURE_DIMS[name]
    enc.eval()
    out = enc(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, dim)
    assert isinstance(enc.fc, nn.Identity)


def test_cifar_stem_adaptation():
    enc, _ = build_encoder("resnet18", cifar_stem=True)
    assert enc.conv1.kernel_size == (3, 3) and enc.conv1.stride == (1, 1)
    assert isinstance(enc.maxpool, nn.Identity)
    std, _ = build_encoder("resnet18", cifar_stem=False)
    assert std.conv1.kernel_size == (7, 7) and not isinstance(std.maxpool, nn.Identity)


def test_random_init_no_pretrained_weights():
    torch.manual_seed(0)
    a, _ = build_encoder("resnet18")
    torch.manual_seed(1)
    b, _ = build_encoder("resnet18")
    assert not torch.equal(a.conv1.weight, b.conv1.weight)
    for p in a.parameters():
        assert torch.isfinite(p).all()
    assert a.conv1.weight.std() > 0


def test_unknown_encoder_raises():
    with pytest.raises(ValueError):
        build_encoder("vgg16")


def test_projection_head_architecture_and_output_dim():
    head = ProjectionHead(512, hidden_dim=2048, out_dim=128, use_bn=True)
    assert head(torch.randn(4, 512)).shape == (4, 128)
    linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(512, 2048), (2048, 128)]
    assert any(isinstance(m, nn.ReLU) for m in head.modules())
    no_bn = ProjectionHead(512, use_bn=False)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in no_bn.modules())


def test_encoder_and_head_are_separate():
    model = build_model(TrainConfig()).eval()
    x = torch.randn(3, 3, 32, 32)
    h, z = model(x)
    assert h.shape == (3, 512) and z.shape == (3, 128)
    assert torch.allclose(model.encode(x), h)
    assert model.encoder is not model.projector


def test_all_trainable_parameters_receive_gradients():
    model = build_model(TrainConfig(projection_hidden_dim=64)).train()
    _, z = model(torch.randn(8, 3, 32, 32))
    z.pow(2).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing
