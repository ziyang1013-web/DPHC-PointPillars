"""
Small Object High-Resolution Auxiliary BEV Backbones.

Four experiment / aux-backbone types:
  - block1_head:       Features from backbone block 1 (stride 2), light conv refinement.
  - fullres_branch:    Full-resolution BEV features (stride 1), downsample + refine.
  - fullres_context:   Full-res BEV + backbone multi-scale context fusion.
  - identity:          Raw spatial_features passthrough — no conv, no params.
                       Used by AUX_BACKBONE_TYPE: identity.

All inputs are expected to be already detached by the caller (PointPillarAux).
"""

import torch
import torch.nn as nn


class AuxBEVBackboneBlock1(nn.Module):
    """
    Experiment 1: block1_head.

    Minimal adaptation layer that isolates the aux head from the backbone:
      Conv 3×3  Cin → 64  stride=1  padding=1
      BatchNorm2d(64)
      ReLU

    Input:  backbone block 1 output at stride 2  (B, Cin, H/2, W/2)
    Output: aux feature map at stride 2            (B, 64, H/2, W/2)
    """

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        use_bn = model_cfg.get('USE_BATCH_NORM', True)

        # Single Conv-BN-ReLU — simplest possible isolation layer.
        # Even when input_channels == 64 we keep this layer to
        # decouple aux-head gradients from the backbone features.
        layers = []
        layers.append(nn.Conv2d(input_channels, 64, kernel_size=3, stride=1,
                                padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        self.convs = nn.Sequential(*layers)
        self.num_bev_features = 64

    def forward(self, aux_input_dict):
        """
        Args:
            aux_input_dict:
                block1_features: (B, Cin, H/2, W/2)
        Returns:
            aux_spatial_features: (B, 64, H/2, W/2)
        """
        x = aux_input_dict['block1_features']
        x = self.convs(x)
        return x


class AuxBEVBackboneFullRes(nn.Module):
    """
    Experiment 2: fullres_branch.

    Full-resolution lightweight branch using depthwise separable conv.
    No spatial downsampling — output is at the same stride-1 resolution
    as the input spatial_features.

    Structure:
      Pointwise Conv 1×1  Cin → hidden  (hidden=32)
      BatchNorm2d + ReLU
      Depthwise Conv 3×3  hidden → hidden  stride=1  padding=1  groups=hidden
      BatchNorm2d + ReLU
      Pointwise Conv 1×1  hidden → 64
      BatchNorm2d + ReLU

    Input:  spatial_features_fullres  (B, Cin, H, W)
    Output: aux_spatial_features      (B, 64, H, W)  — same resolution, stride=1
    """

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        hidden_channels = model_cfg.get('HIDDEN_CHANNELS', 32)
        use_bn = model_cfg.get('USE_BATCH_NORM', True)

        # 1. Pointwise: Cin → hidden
        self.pw1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=not use_bn),
            nn.BatchNorm2d(hidden_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # 2. Depthwise: hidden → hidden
        self.dw = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      stride=1, padding=1, groups=hidden_channels, bias=not use_bn),
            nn.BatchNorm2d(hidden_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # 3. Pointwise: hidden → 64
        self.pw2 = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, kernel_size=1, bias=not use_bn),
            nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.num_bev_features = 64

    def forward(self, aux_input_dict):
        """
        Args:
            aux_input_dict:
                spatial_features_fullres: (B, Cin, H, W)
        Returns:
            aux_spatial_features: (B, 64, H, W)  — stride=1
        """
        x = aux_input_dict['spatial_features_fullres']
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        return x


class AuxBEVBackboneFullResContext(nn.Module):
    """
    Experiment 3 / Exp5: fullres_context.

    Full-resolution branch with parallel dilated depthwise convolutions
    for multi-scale context — no attention, no gating, no learnable scaling.

    Supports arbitrary number of dilation branches via DILATIONS config.

    Structure:
      Input: spatial_features_fullres  (B, Cin, H, W)   stride=1

      Pointwise Conv 1×1  Cin → hidden  (hidden=32)
        BatchNorm2d + ReLU

      For each d in DILATIONS:
        └─ Branch: DWConv 3×3  dil=d  groups=hidden  → (B, hidden, H, W)
             BatchNorm2d + ReLU

      Concat all branches  →  (B, hidden * len(DILATIONS), H, W)

      Pointwise Conv 1×1  fusion_in → 64
        BatchNorm2d + ReLU

      [optional] Plain residual:  output = fused + input
        (only when USE_PLAIN_RESIDUAL=True and input_channels == 64)

    Default DILATIONS: [1, 2]  (backward compatible with original Exp3/Exp5)

    Output: aux_spatial_features  (B, 64, H, W)   stride=1
    """

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        hidden_channels = model_cfg.get('HIDDEN_CHANNELS', 32)
        dilations = model_cfg.get('DILATIONS', [1, 2])
        use_bn = model_cfg.get('USE_BATCH_NORM', True)
        use_residual = model_cfg.get('USE_PLAIN_RESIDUAL', True)

        # Input projection: Cin → hidden
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1,
                      bias=not use_bn),
            nn.BatchNorm2d(hidden_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Parallel dilated depthwise branches (dynamic count)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                          stride=1, padding=d, dilation=d,
                          groups=hidden_channels, bias=not use_bn),
                nn.BatchNorm2d(hidden_channels) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
            ))

        fusion_in = hidden_channels * len(dilations)

        # Output projection: fusion_in → 64
        self.output_proj = nn.Sequential(
            nn.Conv2d(fusion_in, 64, kernel_size=1, bias=not use_bn),
            nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.use_residual = use_residual and (input_channels == 64)
        self.num_bev_features = 64

    def forward(self, aux_input_dict):
        """
        Args:
            aux_input_dict:
                spatial_features_fullres: (B, Cin, H, W)
        Returns:
            aux_spatial_features: (B, 64, H, W)  — stride=1
        """
        identity = aux_input_dict['spatial_features_fullres']

        x = self.input_proj(identity)            # (B, hidden, H, W)

        # Run all dilation branches in parallel
        branch_outputs = [branch(x) for branch in self.branches]

        fused = torch.cat(branch_outputs, dim=1)  # (B, hidden*N, H, W)
        out = self.output_proj(fused)              # (B, 64, H, W)

        if self.use_residual:
            out = out + identity

        return out


class AuxBEVBackboneIdentity(nn.Module):
    """
    Identity pass-through backbone — no convolutions, no parameters.

    Used when AUX_BACKBONE_TYPE: identity.
    Passes raw spatial_features directly to the aux detection head
    to measure whether Exp5's gains come from the AuxBEVBackbone design
    or simply from adding a stride-1 detection head.

    Input:  spatial_features_fullres  (B, Cin, H, W)
    Output: aux_spatial_features      (B, Cin, H, W)  — identical to input
    """

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = input_channels

    def forward(self, aux_input_dict):
        """
        Args:
            aux_input_dict:
                spatial_features_fullres: (B, Cin, H, W)
        Returns:
            aux_spatial_features: (B, Cin, H, W)  — identity
        """
        return aux_input_dict['spatial_features_fullres']
