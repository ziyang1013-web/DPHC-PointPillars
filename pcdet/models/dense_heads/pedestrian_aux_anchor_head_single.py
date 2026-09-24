"""
Pedestrian-only Auxiliary Anchor Head (Exp5).

Single-class detection head for Pedestrian only — no Car, no Cyclist.

Global class IDs: bg=0, Car=1, Pedestrian=2, Cyclist=3
Aux local label:  Pedestrian=1

Key differences from AuxAnchorHeadSingle (2-class):
  - num_class = 1
  - class_names = ['Pedestrian']
  - Only Pedestrian anchors
  - GT filtering: global 2 → local 1, all other GT removed
  - conv_cls output = num_anchors_per_location (not ×2)
"""

import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class PedestrianAuxAnchorHeadSingle(nn.Module):
    """
    Single-class auxiliary anchor head — Pedestrian only.
    Standalone, does not inherit from AnchorHeadTemplate or AuxAnchorHeadSingle.
    """

    def __init__(self, model_cfg, input_channels, num_class, class_names,
                 grid_size, point_cloud_range, predict_boxes_when_training=True, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class   # 1
        self.class_names = class_names  # ['Pedestrian']
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = False

        # --- Box coder ---
        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        # --- Anchors: Pedestrian only ---
        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]

        # --- Target assigner (Pedestrian-only class_names) ---
        self.target_assigner = AxisAlignedTargetAssigner(
            model_cfg=self.model_cfg,
            class_names=self.class_names,   # ['Pedestrian']
            box_coder=self.box_coder,
            match_height=anchor_target_cfg.get('MATCH_HEIGHT', False)
        )

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

        # --- Conv heads ---
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        # cls: num_anchors_per_location × 1 class
        self.conv_cls = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        # box: num_anchors_per_location × code_size
        self.conv_box = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

        self.init_weights()

    # ------------------------------------------------------------------
    #  Init helpers
    # ------------------------------------------------------------------

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride']
                            for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = \
            anchor_generator.generate_anchors(feature_map_size)
        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors
        return anchors_list, num_anchors_per_location_list

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(
                code_weights=losses_cfg.LOSS_WEIGHTS['code_weights']
            )
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    # ------------------------------------------------------------------
    #  Target assignment — Pedestrian only
    # ------------------------------------------------------------------

    def assign_targets(self, gt_boxes):
        """
        Filter gt_boxes to Pedestrian (global class 2) only.
        Remap: global 2 → local 1.
        Does NOT modify the original gt_boxes in batch_dict.
        """
        aux_gt_boxes = self._prepare_ped_gt_boxes(gt_boxes)
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, aux_gt_boxes
        )
        return targets_dict

    def _prepare_ped_gt_boxes(self, gt_boxes):
        """
        Create ped-only gt_boxes tensor.
        Keeps only global class 2 (Pedestrian), remaps to local 1.
        All other classes (Car=1, Cyc=3, bg=0) become padding.
        """
        batch_size = gt_boxes.shape[0]
        aux_gt_list = []

        for b in range(batch_size):
            gt = gt_boxes[b]  # (M, 8)
            valid_mask = (gt[:, :7].abs().sum(dim=-1) > 1e-6)
            if valid_mask.sum() == 0:
                aux_gt_list.append(gt.new_zeros(0, 8))
                continue

            valid_gt = gt[valid_mask]
            valid_cls = valid_gt[:, -1]

            # Keep only Pedestrian (global class 2)
            keep_mask = (valid_cls == 2)
            filtered = valid_gt[keep_mask].clone()

            if filtered.shape[0] > 0:
                # Remap global 2 → local 1
                filtered[:, -1] = 1.0

            aux_gt_list.append(filtered)

        max_gt = max((g.shape[0] for g in aux_gt_list), default=1)
        if max_gt == 0:
            max_gt = 1
        aux_gt_boxes = gt_boxes.new_zeros(batch_size, max_gt, 8)
        for b, g in enumerate(aux_gt_list):
            if g.shape[0] > 0:
                aux_gt_boxes[b, :g.shape[0]] = g
        return aux_gt_boxes

    # ------------------------------------------------------------------
    #  Loss
    # ------------------------------------------------------------------

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])

        cared = box_cls_labels >= 0
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        if self.num_class == 1:
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_targets = cls_targets.squeeze(dim=-1)

        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)

        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)
        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

        tb_dict = {'aux_rpn_loss_cls': cls_loss.item()}
        return cls_loss, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location)

        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

        box_loss = loc_loss
        tb_dict = {'aux_rpn_loss_loc': loc_loss.item()}

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['aux_rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        tb_dict['aux_rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    # ------------------------------------------------------------------
    #  Box decoding
    # ------------------------------------------------------------------

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Decode predictions to absolute boxes.
        cls_preds: (N, H, W, num_anchors_per_loc * 1) — single Ped class.
        Returns (batch_cls_preds, batch_box_preds) with local scores.
        """
        if isinstance(self.anchors, list):
            anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float()
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + \
                period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    # ------------------------------------------------------------------
    #  Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
            torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape), num_bins,
                dtype=anchors.dtype, device=dir_cls_targets.device
            )
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, data_dict):
        spatial_features_2d = data_dict['aux_spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)   # → (N, 2, H, W)  [1 class × 2 rot]
        box_preds = self.conv_box(spatial_features_2d)   # → (N, 14, H, W) [2 rot × 7 code]

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(gt_boxes=data_dict['gt_boxes'])
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds,
                dir_cls_preds=dir_cls_preds
            )
            # aux_batch_cls_preds: (B, N_aux, 1) — single Ped channel
            data_dict['aux_batch_cls_preds'] = batch_cls_preds
            data_dict['aux_batch_box_preds'] = batch_box_preds
            data_dict['aux_cls_preds_normalized'] = False

        return data_dict
