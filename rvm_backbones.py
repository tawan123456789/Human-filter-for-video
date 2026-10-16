"""
RVM (Robust Video Matting) segmentation network with swappable MobileNet encoders.

The official RVM only ships `mobilenetv3` (= MobileNetV3-Large) and `resnet50`.
This module re-creates the RVM segmentation path

    encoder -> LR-ASPP -> Recurrent (ConvGRU) decoder -> project_seg

and lets the encoder be replaced with:

    "mobilenetv3_small"  : torchvision MobileNetV3-Small (ImageNet weights)
    "mobilenetv2"        : torchvision MobileNetV2      (ImageNet weights)
    "mobilenetv1"        : timm MobileNetV1-100          (ImageNet weights)

Everything after the encoder (LR-ASPP 128ch, decoder [80, 40, 32, 16],
Projection 16 -> 1) is identical to the official RVM-MobileNetV3 so that
the comparison isolates the effect of the backbone.

The call signature matches the official model so the same training /
evaluation / webcam code works for every variant:

    seg, r1, r2, r3, r4 = model(src, r1, r2, r3, r4, 1.0, True)
    seg, *rec          = model(src, segmentation_pass=True)

Only single frames (B, C, H, W) are supported - this is how the notebook
and webcam.py call the model (temporal memory is carried through r1..r4).

LR-ASPP / decoder / ConvGRU / Projection are adapted from
https://github.com/PeterL1n/RobustVideoMatting (GPL-3.0).
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ============================================================
# ENCODERS
#
# Each encoder returns 4 feature maps at strides
#   f1: /2, f2: /4, f3: /8, f4: /16
# (same strides as the official RVM MobileNetV3-Large encoder,
#  the last stage is dilated instead of downsampled to /32)
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _Encoder(nn.Module):
    """Base class: input in [0, 1] RGB, ImageNet normalisation inside."""

    channels: List[int]

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(std).view(1, 3, 1, 1), persistent=False
        )

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class _StagedEncoder(_Encoder):
    """Encoder made of 4 sequential stages (torchvision models)."""

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.normalize(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class MobileNetV3SmallEncoder(_StagedEncoder):
    """
    torchvision mobilenet_v3_small, dilated=True (last stage /16).

    features[0]      stem             16ch  /2   -> f1
    features[1]      bneck (SE)       16ch  /4   -> f2
    features[2:4]    bneck            24ch  /8   -> f3
    features[4:]     bnecks + 1x1    576ch  /16  -> f4
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import (
            MobileNet_V3_Small_Weights,
            mobilenet_v3_small,
        )

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v3_small(weights=weights, dilated=True).features

        self.stage1 = features[0]
        self.stage2 = features[1]
        self.stage3 = features[2:4]
        self.stage4 = features[4:]
        self.channels = [16, 16, 24, 576]


class MobileNetV2Encoder(_StagedEncoder):
    """
    torchvision mobilenet_v2. The /32 stage (features[14:18]) is dilated
    (stride 1, dilation 2) so the output stride is /16 like RVM.
    The final 1x1 conv to 1280ch (features[18]) is dropped (DeepLabV3 style).

    features[0:2]    stem + IR        16ch  /2   -> f1
    features[2:4]    IR               24ch  /4   -> f2
    features[4:7]    IR               32ch  /8   -> f3
    features[7:18]   IR              320ch  /16  -> f4
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v2(weights=weights).features

        # /32 -> /16 : replace stride with dilation in the depthwise 3x3 convs
        for module in features[14:18].modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
                module.stride = (1, 1)
                module.dilation = (2, 2)
                module.padding = (2, 2)

        self.stage1 = features[0:2]
        self.stage2 = features[2:4]
        self.stage3 = features[4:7]
        self.stage4 = features[7:18]
        self.channels = [16, 24, 32, 320]


class MobileNetV1Encoder(_Encoder):
    """
    timm mobilenetv1_100 (ImageNet), features_only, output_stride=16.

    Feature maps at /2, /4, /8, /16 (the /32 stage is dilated to /16 and
    used as f4):  64, 128, 256, 1024 channels.
    """

    TIMM_NAME = "mobilenetv1_100.ra4_e3600_r224_in1k"

    def __init__(self, pretrained: bool = True):
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "MobileNetV1 needs `timm`  ->  pip install timm"
            ) from e

        net = timm.create_model(
            self.TIMM_NAME,
            pretrained=pretrained,
            features_only=True,
            output_stride=16,
        )
        cfg = getattr(net, "pretrained_cfg", None) or {}
        super().__init__(
            mean=cfg.get("mean", IMAGENET_MEAN),
            std=cfg.get("std", IMAGENET_STD),
        )
        self.net = net

        # pick one feature per stride: /2, /4, /8 and the LAST /16 map
        reductions = net.feature_info.reduction()
        channels = net.feature_info.channels()
        idx = [reductions.index(2), reductions.index(4), reductions.index(8)]
        idx.append(len(reductions) - 1 - reductions[::-1].index(16))
        self.out_idx = idx
        self.channels = [channels[i] for i in idx]

    def forward(self, x: Tensor) -> List[Tensor]:
        feats = self.net(self.normalize(x))
        return [feats[i] for i in self.out_idx]


ENCODERS = {
    "mobilenetv3_small": MobileNetV3SmallEncoder,
    "mobilenetv2": MobileNetV2Encoder,
    "mobilenetv1": MobileNetV1Encoder,
}


# ============================================================
# RVM HEAD  (adapted from RobustVideoMatting, single-frame only)
# ============================================================

class LRASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.aspp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.aspp1(x) * self.aspp2(x)


class ConvGRU(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid(),
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, h: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        if h is None:
            h = torch.zeros_like(x)
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h


class BottleneckBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)

    def forward(self, x: Tensor, r: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        a, b = x.split(self.channels // 2, dim=1)
        b, r = self.gru(b, r)
        return torch.cat([a, b], dim=1), r


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, src_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels,
                      out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = ConvGRU(out_channels // 2)

    def forward(self, x: Tensor, f: Tensor, s: Tensor, r: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = x[:, :, : s.size(2), : s.size(3)]
        x = self.conv(torch.cat([x, f, s], dim=1))
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        return torch.cat([a, b], dim=1), r


class OutputBlock(nn.Module):
    def __init__(self, in_channels: int, src_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = x[:, :, : s.size(2), : s.size(3)]
        return self.conv(torch.cat([x, s], dim=1))


class RecurrentDecoder(nn.Module):
    def __init__(self, feature_channels: List[int], decoder_channels: List[int]):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self, s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3, r4


class Projection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


# ============================================================
# FULL MODEL
# ============================================================

class RVMSegmentation(nn.Module):
    """RVM segmentation network with a swappable MobileNet encoder."""

    ASPP_CHANNELS = 128
    DECODER_CHANNELS = [80, 40, 32, 16]

    def __init__(self, variant: str = "mobilenetv3_small",
                 pretrained_backbone: bool = True):
        super().__init__()
        if variant not in ENCODERS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(ENCODERS)}"
            )
        self.variant = variant
        self.backbone = ENCODERS[variant](pretrained_backbone)

        c1, c2, c3, c4 = self.backbone.channels
        self.aspp = LRASPP(c4, self.ASPP_CHANNELS)
        self.decoder = RecurrentDecoder(
            [c1, c2, c3, self.ASPP_CHANNELS], self.DECODER_CHANNELS
        )
        self.project_seg = Projection(self.DECODER_CHANNELS[-1], 1)

    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1.0,
                segmentation_pass: bool = True) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        if src.ndim != 4:
            raise ValueError("RVMSegmentation expects (B, 3, H, W) input")

        if downsample_ratio != 1.0:
            src = F.interpolate(src, scale_factor=downsample_ratio,
                                mode="bilinear", align_corners=False,
                                recompute_scale_factor=False)

        f1, f2, f3, f4 = self.backbone(src)
        f4 = self.aspp(f4)
        hid, r1, r2, r3, r4 = self.decoder(src, f1, f2, f3, f4, r1, r2, r3, r4)
        seg = self.project_seg(hid)
        return seg, r1, r2, r3, r4

    def param_groups(self, lr_backbone: float, lr_head: float):
        """Optimizer param groups: low LR for the ImageNet encoder."""
        head = [p for n, p in self.named_parameters()
                if not n.startswith("backbone.")]
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": head, "lr": lr_head},
        ]


def build_rvm(variant: str, pretrained_backbone: bool = True) -> RVMSegmentation:
    return RVMSegmentation(variant, pretrained_backbone)


def load_rvm_checkpoint(path, device="cpu") -> Tuple[nn.Module, dict]:
    """
    Load any checkpoint produced by the notebooks.

    - checkpoints with config["variant"] in ENCODERS -> RVMSegmentation
    - otherwise -> official RVM-MobileNetV3(-Large) from torch.hub
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    variant = checkpoint.get("config", {}).get("variant", "mobilenetv3")

    if variant in ENCODERS:
        model = build_rvm(variant, pretrained_backbone=False)
    else:
        model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3",
                               pretrained=False, trust_repo=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


if __name__ == "__main__":
    # quick shape / parameter check:  python rvm_backbones.py
    for name in ENCODERS:
        m = build_rvm(name, pretrained_backbone=False).eval()
        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            seg, *rec = m(x)
            seg2, *_ = m(x, *rec, 1.0, True)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:18s} enc={m.backbone.channels} params={n/1e6:.2f}M "
              f"seg={tuple(seg.shape)} rec={[tuple(r.shape) for r in rec]}")
