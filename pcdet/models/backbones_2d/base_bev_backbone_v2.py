"""
BaseBEVBackboneWithIntermediate — a subclass of BaseBEVBackbone that additionally
stores per-block intermediate feature maps in data_dict so that downstream
auxiliary branches can access them.

Keys written to data_dict (in addition to parent behaviour):
    spatial_features_2x   — after block 1  (stride 2)
    spatial_features_4x   — after block 2  (stride 4)
    spatial_features_8x   — after block 3  (stride 8)

Usage:
    BACKBONE_2D:
        NAME: BaseBEVBackboneWithIntermediate
        ...  # all other BaseBEVBackbone parameters

When not needed, keep using the original BaseBEVBackbone — its behaviour is
completely unchanged.
"""

import torch
from .base_bev_backbone import BaseBEVBackbone


class BaseBEVBackboneWithIntermediate(BaseBEVBackbone):
    """
    Same as BaseBEVBackbone but also exposes per-block intermediate features
    via data_dict keys ('spatial_features_2x', 'spatial_features_4x',
    'spatial_features_8x').

    This subclass is intentionally minimal — it only adds the data_dict writes.
    All other behaviour (weights, config, serialisation) is inherited unchanged.
    """

    def forward(self, data_dict):
        """
        Identical to BaseBEVBackbone.forward() except that intermediate
        feature maps are written to data_dict (in addition to the local
        ret_dict that the parent already populates).
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)


            stride = int(spatial_features.shape[2] / x.shape[2])
            # === ONLY DIFFERENCE from parent: expose to downstream ===
            data_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)


        data_dict['spatial_features_2d'] = x
        return data_dict
