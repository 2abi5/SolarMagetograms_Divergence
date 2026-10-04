"""Model factory: every model maps (B, 1, h, w) MDI -> (B, 3, 4h, 4w) SHARP (Bp, Bt, Br)."""

from sr.models.edsr import EDSR
from sr.models.highresnet import HighResNetSR
from sr.models.unet import UNetSR

MODELS = {"highresnet": HighResNetSR, "unet": UNetSR, "edsr": EDSR}


def build_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def count_params(model, active_only=False):
    """All parameters, or only those used in a forward pass (HighRes-Net's
    FusionBlock is inert at K = 1)."""
    skip = getattr(model, "inactive_prefixes", ()) if active_only else ()
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith(skip))
