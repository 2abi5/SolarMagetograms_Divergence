"""Model tests (training_plan.md Phase 5): shapes, x4, 1 -> 3 channels, no BatchNorm."""
import pytest
import torch
import torch.nn as nn

from sr.models import build_model, count_params

NAMES = ("highresnet", "unet", "edsr")


@pytest.mark.parametrize("name", NAMES)
def test_patch_shapes(name):
    m = build_model(name)
    y = m(torch.randn(2, 1, 16, 16))
    assert y.shape == (2, 3, 64, 64)


@pytest.mark.parametrize("name", NAMES)
def test_full_region_shapes_at_the_required_multiple(name):
    m = build_model(name).eval()
    k = m.lr_multiple
    h, w = 20 * k, 28 * k            # a region already padded to the model's multiple
    with torch.no_grad():
        assert m(torch.randn(1, 1, h, w)).shape == (1, 3, 4 * h, 4 * w)


@pytest.mark.parametrize("name", ("highresnet", "edsr"))
def test_fully_convolutional_models_take_any_size(name):
    m = build_model(name).eval()
    assert m.lr_multiple == 1
    with torch.no_grad():
        assert m(torch.randn(1, 1, 37, 61)).shape == (1, 3, 148, 244)


@pytest.mark.parametrize("name", NAMES)
def test_no_batchnorm_anywhere(name):
    m = build_model(name)
    assert not any(isinstance(x, nn.modules.batchnorm._BatchNorm) for x in m.modules())


def test_highresnet_reuses_the_converter_blocks():
    from source.models.highresnet_rprcdo import FusionBlock
    from source.models.model_utils_RPRCDO import Encoder, ResidualBlock
    m = build_model("highresnet")
    assert isinstance(m.encode, Encoder) and isinstance(m.fuse, FusionBlock)
    assert sum(isinstance(x, ResidualBlock) for x in m.modules()) == 3   # 2 encoder + 1 fusion


def test_highresnet_fusion_is_inert_at_k1_and_active_params_are_counted():
    m = build_model("highresnet")
    m(torch.randn(2, 1, 16, 16)).sum().backward()
    active = sum(p.numel() for p in m.parameters() if p.grad is not None)
    total = count_params(m)
    assert active == count_params(m, active_only=True) == 222409
    assert total == 222409 + 368963                 # FusionBlock(64) parameters, never used


@pytest.mark.parametrize("name", ("unet", "edsr"))
def test_baselines_are_in_the_highresnet_parameter_ballpark(name):
    ref = count_params(build_model("highresnet"), active_only=True)
    n = count_params(build_model(name))
    assert ref / 3 < n < ref * 3


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("resnet")
