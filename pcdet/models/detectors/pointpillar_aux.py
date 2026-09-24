"""
PointPillarSmallAux — PointPillar with Small Object High-Resolution Auxiliary Branches.

Adds an independent auxiliary detection branch for Pedestrian / Cyclist
while keeping the original Density-only main branch completely frozen.

Three experiment types (set via SMALL_OBJECT_AUX.EXPERIMENT_TYPE):
  - block1_head:       backbone block-1 features at stride 2
  - fullres_branch:    full-resolution BEV features at stride 1
  - fullres_context:   full-res BEV + multi-scale backbone context

Three inference modes (set via SMALL_OBJECT_AUX.AUX_INFERENCE_MODE):
  - main_only:  original Density-only predictions (exact match)
  - aux_only:   only aux head Pedestrian / Cyclist predictions
  - merged:     Car from main head; Ped/Cyc merged from both + class-wise NMS

When SMALL_OBJECT_AUX.ENABLED is False (or the key is absent), this class
behaves identically to the original PointPillar detector.
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn

from .detector3d_template import Detector3DTemplate
from ..backbones_2d.aux_bev_backbone import (
    AuxBEVBackboneBlock1,
    AuxBEVBackboneFullRes,
    AuxBEVBackboneFullResContext,
    AuxBEVBackboneIdentity,
)
from ..dense_heads.aux_anchor_head_single import AuxAnchorHeadSingle
from ..model_utils import model_nms_utils

# Main-path module name prefixes (used for freezing + checkpoint categorisation)
_MAIN_PATH_PREFIXES = ('vfe.', 'map_to_bev_module.', 'backbone_2d.', 'dense_head.')
_AUX_PATH_PREFIXES = ('aux_backbone.', 'aux_dense_head.')


class PointPillarSmallAux(Detector3DTemplate):
    """
    PointPillar detector with optional small-object auxiliary branch.

    When SMALL_OBJECT_AUX.ENABLED is False (or the key is absent), this class
    behaves identically to the original PointPillar detector.
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # --- Read aux config ---
        aux_cfg = model_cfg.get('SMALL_OBJECT_AUX', None)
        self.aux_enabled = (aux_cfg is not None and aux_cfg.get('ENABLED', False))

        if self.aux_enabled:
            self._init_aux(aux_cfg)

    # ------------------------------------------------------------------
    #  Aux initialisation
    # ------------------------------------------------------------------

    def _init_aux(self, aux_cfg):
        """Build auxiliary backbone + head and freeze the main branch."""
        self.aux_cfg = aux_cfg
        self.experiment_type = aux_cfg.EXPERIMENT_TYPE
        self.aux_class_names = list(aux_cfg.CLASS_NAMES)
        self.aux_num_class = len(self.aux_class_names)
        self.detach_input = aux_cfg.get('DETACH_INPUT', True)
        self.freeze_main_branch = aux_cfg.get('FREEZE_MAIN_BRANCH', True)
        self.aux_loss_weight = aux_cfg.get('AUX_LOSS_WEIGHT', 1.0)
        self.aux_inference_mode = aux_cfg.get('AUX_INFERENCE_MODE', 'merged')
        self.aux_backbone_type = aux_cfg.get('AUX_BACKBONE_TYPE', 'context')
        default_input_source = (
            'block1' if self.experiment_type == 'block1_head' else 'fullres'
        )
        self.aux_input_source = aux_cfg.get(
            'AUX_INPUT_SOURCE', default_input_source
        )
        if self.aux_input_source not in ('fullres', 'block1'):
            raise ValueError(
                "SMALL_OBJECT_AUX.AUX_INPUT_SOURCE must be 'fullres' or "
                f"'block1', got '{self.aux_input_source}'"
            )

        self._build_aux_backbone()
        self._build_aux_head()

        if self.freeze_main_branch:
            self._freeze_main_branch()

    def _build_aux_backbone(self):
        """Instantiate the correct aux backbone based on AUX_BACKBONE_TYPE."""
        aux_type = self.aux_backbone_type

        if aux_type == 'identity':
            self.aux_backbone = AuxBEVBackboneIdentity(
                model_cfg=self.aux_cfg, input_channels=64
            )
            return

        # --- existing experiment_type-based dispatch (backward compatible) ---
        if self.experiment_type == 'block1_head':
            self.aux_backbone = AuxBEVBackboneBlock1(
                model_cfg=self.aux_cfg, input_channels=64
            )
        elif self.experiment_type == 'fullres_branch':
            self.aux_backbone = AuxBEVBackboneFullRes(
                model_cfg=self.aux_cfg, input_channels=64
            )
        elif self.experiment_type == 'fullres_context':
            self.aux_backbone = AuxBEVBackboneFullResContext(
                model_cfg=self.aux_cfg, input_channels=64
            )
        else:
            raise ValueError(f"Unknown SMALL_OBJECT_AUX.EXPERIMENT_TYPE: "
                             f"'{self.experiment_type}'")

    def _build_aux_head(self):
        """Build the aux detection head (AnchorHeadSingle-style, 2 classes)."""
        grid_size = self.dataset.grid_size
        aux_grid_size = np.array([grid_size[0], grid_size[1], grid_size[2]])
        aux_head_input_channels = self.aux_backbone.num_bev_features
        point_cloud_range = self.dataset.point_cloud_range
        aux_head_cfg = self._build_aux_head_cfg()

        self.aux_dense_head = AuxAnchorHeadSingle(
            model_cfg=aux_head_cfg,
            input_channels=aux_head_input_channels,
            num_class=self.aux_num_class,
            class_names=self.aux_class_names,
            grid_size=aux_grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=True,
        )

    def _build_aux_head_cfg(self):
        """Build config for the aux head — inherit from DENSE_HEAD, keep Ped/Cyc only."""
        head_cfg = copy.deepcopy(self.model_cfg.DENSE_HEAD)

        all_anchor_configs = head_cfg.ANCHOR_GENERATOR_CONFIG
        aux_anchor_configs = [
            acfg for acfg in all_anchor_configs
            if acfg['class_name'] in self.aux_class_names
        ]

        # Override feature_map_stride per the aux config (e.g. stride=1
        # for fullres_branch, stride=2 for block1_head).
        aux_stride = self.aux_cfg.get('FEATURE_MAP_STRIDE', None)
        if aux_stride is not None:
            for acfg in aux_anchor_configs:
                acfg['feature_map_stride'] = int(aux_stride)

        head_cfg.ANCHOR_GENERATOR_CONFIG = aux_anchor_configs

        aux_head_override = self.aux_cfg.get('AUX_HEAD_CFG', None)
        if aux_head_override is not None:
            for key, val in aux_head_override.items():
                head_cfg[key] = val

        return head_cfg

    # ------------------------------------------------------------------
    #  Freezing
    # ------------------------------------------------------------------

    def _freeze_main_branch(self):
        """Freeze all main-branch parameters and permanently set them to eval."""
        for name in ['vfe', 'map_to_bev_module', 'backbone_2d', 'dense_head']:
            module = getattr(self, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    def train(self, mode=True):
        """
        Override train() so that frozen main-branch modules stay in eval mode
        even when the overall model is set to train().
        """
        super().train(mode)
        if mode and self.aux_enabled and self.freeze_main_branch:
            for name in ['vfe', 'map_to_bev_module', 'backbone_2d', 'dense_head']:
                m = getattr(self, name, None)
                if m is not None:
                    m.eval()
        return self

    # ------------------------------------------------------------------
    #  Checkpoint loading with detailed report
    # ------------------------------------------------------------------

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        """
        Extended checkpoint loader that prints a categorised report:

          - Successfully loaded  main-path  parameters (count + names)
          - Missing  auxiliary  parameters  (expected when loading from
            a Density-only checkpoint — these will be randomly initialised)
          - Unexpected  parameters  (in checkpoint but not in model)
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Checkpoint not found: {filename}")

        logger.info(f'==> Loading parameters from checkpoint {filename} '
                     f'to {"CPU" if to_cpu else "GPU"}')
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type, weights_only=False)
        model_state_disk = checkpoint['model_state']

        if pre_trained_path is not None:
            pretrain_ckpt = torch.load(pre_trained_path, map_location=loc_type, weights_only=False)
            model_state_disk.update(pretrain_ckpt['model_state'])

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info(f'==> Checkpoint trained from version: {version}')

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        # ---- Categorise parameters ----
        ckpt_keys = set(model_state_disk.keys())
        model_keys = set(state_dict.keys())

        loaded_main = [k for k in (ckpt_keys & model_keys)
                       if k.startswith(_MAIN_PATH_PREFIXES)]
        loaded_other = [k for k in (ckpt_keys & model_keys)
                        if not k.startswith(_MAIN_PATH_PREFIXES)
                        and not k.startswith(_AUX_PATH_PREFIXES)]
        missing_aux = [k for k in model_keys
                       if k.startswith(_AUX_PATH_PREFIXES)
                       and k not in ckpt_keys]
        unexpected = [k for k in ckpt_keys if k not in model_keys]

        # Also catch main-path params that are NOT in the checkpoint
        missing_main = [k for k in model_keys
                        if k.startswith(_MAIN_PATH_PREFIXES)
                        and k not in ckpt_keys]

        # ---- Report ----
        logger.info('=' * 62)
        logger.info('  Checkpoint Loading Report')
        logger.info('=' * 62)
        logger.info(f'  Successfully loaded main-path params : {len(loaded_main):>6d}')
        logger.info(f'  Successfully loaded other params     : {len(loaded_other):>6d}')
        logger.info(f'  Missing   auxiliary   params         : {len(missing_aux):>6d}  '
                     '(expected — will be randomly init)')
        if missing_main:
            logger.warning(f'  *** WARNING: {len(missing_main)} main-path params MISSING '
                           f'in checkpoint! ***')

        if loaded_main:
            logger.info(f'  --- Loaded main-path params ({len(loaded_main)}) ---')
            for k in sorted(loaded_main):
                logger.info(f'    {k}  {list(state_dict[k].shape)}')

        if missing_aux:
            logger.info(f'  --- Missing aux params ({len(missing_aux)}) ---')
            for k in sorted(missing_aux):
                logger.info(f'    {k}  {list(state_dict[k].shape)}')

        if unexpected:
            logger.info(f'  --- Unexpected in checkpoint ({len(unexpected)}) ---')
            for k in sorted(unexpected):
                logger.info(f'    {k}')

        logger.info(f'  Total: {len(update_model_state)}/{len(state_dict)} '
                     f'parameters loaded')
        logger.info('=' * 62)

        return state_dict, update_model_state

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, batch_dict):
        # --- Phase 1: Main branch (VFE → Scatter → Backbone → DenseHead) ---
        if self.aux_enabled and self.freeze_main_branch and self.training:
            with torch.no_grad():
                for cur_module in self.module_list:
                    batch_dict = cur_module(batch_dict)
        else:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

        # --- Phase 2: Aux branch ---
        if self.aux_enabled:
            batch_dict = self._forward_aux(batch_dict)

        # --- Phase 3: Loss or post-processing ---
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def _forward_aux(self, batch_dict):
        """Gather intermediate features, detach, run aux backbone + head."""
        aux_input_dict = {}

        # Matched stride-2 context ablation: keep the FullResContext module
        # unchanged and only replace its stride-1 input with block-1 features.
        if (
            self.aux_input_source == 'block1'
            and self.experiment_type != 'block1_head'
        ):
            source_key = 'spatial_features_2x'
            if source_key not in batch_dict:
                raise KeyError(
                    "AUX_INPUT_SOURCE: block1 requires spatial_features_2x. "
                    "Use BaseBEVBackboneWithIntermediate."
                )
            feat = batch_dict[source_key]
            if self.detach_input:
                feat = feat.detach()
            aux_input_dict['spatial_features_fullres'] = feat

        if self.aux_backbone_type == 'identity':
            # Raw spatial_features → straight to aux head (no conv, no params).
            feat = batch_dict['spatial_features']
            if self.detach_input:
                feat = feat.detach()
            aux_input_dict['spatial_features_fullres'] = feat

        elif self.experiment_type == 'block1_head':
            feat = batch_dict['spatial_features_2x']
            if self.detach_input:
                feat = feat.detach()
            aux_input_dict['block1_features'] = feat

        elif (
            self.experiment_type in ('fullres_branch', 'fullres_context')
            and self.aux_input_source == 'fullres'
        ):
            # Both use raw BEV spatial_features (stride 1) — the difference
            # is in the aux backbone architecture, not the input source.
            feat = batch_dict['spatial_features']
            if self.detach_input:
                feat = feat.detach()
            aux_input_dict['spatial_features_fullres'] = feat

        aux_features = self.aux_backbone(aux_input_dict)
        batch_dict['aux_spatial_features_2d'] = aux_features
        batch_dict = self.aux_dense_head(batch_dict)
        return batch_dict

    # ------------------------------------------------------------------
    #  Training loss
    # ------------------------------------------------------------------

    def get_training_loss(self):
        disp_dict = {}

        if self.aux_enabled:
            loss_aux, tb_dict_aux = self.aux_dense_head.get_loss()
            tb_dict = {}
            tb_dict.update({f'aux_{k}': v for k, v in tb_dict_aux.items()})
            tb_dict['loss_aux'] = loss_aux.item()
            total_loss = self.aux_loss_weight * loss_aux
        else:
            loss_main, tb_dict = self.dense_head.get_loss()
            tb_dict['loss_rpn'] = loss_main.item()
            total_loss = loss_main

        return total_loss, tb_dict, disp_dict

    # ------------------------------------------------------------------
    #  Post-processing / Inference
    # ------------------------------------------------------------------

    def post_processing(self, batch_dict):
        if not self.aux_enabled:
            return super().post_processing(batch_dict)

        if self.aux_inference_mode == 'main_only':
            return super().post_processing(batch_dict)
        elif self.aux_inference_mode == 'aux_only':
            return self._post_processing_aux_only(batch_dict)
        elif self.aux_inference_mode == 'merged':
            return self._post_processing_merged(batch_dict)
        elif self.aux_inference_mode == 'selective':
            return self._post_processing_selective(batch_dict)
        else:
            raise ValueError(f"Unknown AUX_INFERENCE_MODE: "
                             f"'{self.aux_inference_mode}'")

    # ------------------------------------------------------------------
    #  aux_only inference
    # ------------------------------------------------------------------

    def _post_processing_aux_only(self, batch_dict):
        """Return only aux-head Pedestrian / Cyclist predictions (global labels)."""
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []

        for index in range(batch_size):
            batch_mask = index

            aux_cls_preds = batch_dict['aux_batch_cls_preds'][batch_mask]
            aux_box_preds = batch_dict['aux_batch_box_preds'][batch_mask]

            if not batch_dict['aux_cls_preds_normalized']:
                aux_cls_preds = torch.sigmoid(aux_cls_preds)

            local_to_global = torch.tensor([2, 3], dtype=torch.long,
                                           device=aux_cls_preds.device)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                final_scores, final_labels_local, final_boxes = \
                    model_nms_utils.multi_classes_nms(
                        cls_scores=aux_cls_preds, box_preds=aux_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                final_labels = local_to_global[final_labels_local]
            else:
                cls_preds_max, label_preds_local = torch.max(aux_cls_preds, dim=-1)
                label_preds_global = local_to_global[label_preds_local]

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds_max, box_preds=aux_box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                final_scores = selected_scores
                final_labels = label_preds_global[selected]
                final_boxes = aux_box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            pred_dicts.append({
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
            })

        return pred_dicts, recall_dict

    # ------------------------------------------------------------------
    #  merged inference
    # ------------------------------------------------------------------

    def _post_processing_merged(self, batch_dict):
        """
        Merged inference mode:

        - Car  (global 1): main head only.
        - Ped  (global 2): main + aux merged → class-wise NMS.
        - Cyc  (global 3): main + aux merged → class-wise NMS.

        NMS is per-class — different classes never suppress each other.
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []

        MAIN_CAR, MAIN_PED, MAIN_CYC = 0, 1, 2
        AUX_PED, AUX_CYC = 0, 1
        GLOBAL_CAR, GLOBAL_PED, GLOBAL_CYC = 1, 2, 3

        for index in range(batch_size):
            batch_mask = index

            main_cls = batch_dict['batch_cls_preds'][batch_mask]
            main_box = batch_dict['batch_box_preds'][batch_mask]
            if not batch_dict['cls_preds_normalized']:
                main_cls = torch.sigmoid(main_cls)

            aux_cls = batch_dict['aux_batch_cls_preds'][batch_mask]
            aux_box = batch_dict['aux_batch_box_preds'][batch_mask]
            if not batch_dict['aux_cls_preds_normalized']:
                aux_cls = torch.sigmoid(aux_cls)

            nms_cfg = post_process_cfg.NMS_CONFIG
            score_th = post_process_cfg.SCORE_THRESH

            # -- Car: main only --
            car_scores = main_cls[:, MAIN_CAR]
            car_sel, car_final_scores = model_nms_utils.class_agnostic_nms(
                box_scores=car_scores, box_preds=main_box,
                nms_config=nms_cfg, score_thresh=score_th
            )
            car_boxes = main_box[car_sel]
            car_labels = car_boxes.new_full((car_boxes.shape[0],), GLOBAL_CAR, dtype=torch.long)

            # -- Pedestrian: main + aux merged --
            ped_boxes, ped_scores, ped_labels = self._merge_class(
                main_cls[:, MAIN_PED], main_box,
                aux_cls[:, AUX_PED], aux_box,
                GLOBAL_PED, nms_cfg, score_th
            )

            # -- Cyclist: main + aux merged --
            cyc_boxes, cyc_scores, cyc_labels = self._merge_class(
                main_cls[:, MAIN_CYC], main_box,
                aux_cls[:, AUX_CYC], aux_box,
                GLOBAL_CYC, nms_cfg, score_th
            )

            final_boxes = torch.cat([car_boxes, ped_boxes, cyc_boxes], dim=0)
            final_scores = torch.cat([car_final_scores, ped_scores, cyc_scores], dim=0)
            final_labels = torch.cat([car_labels, ped_labels, cyc_labels], dim=0)

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            pred_dicts.append({
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
            })

        return pred_dicts, recall_dict

    # ------------------------------------------------------------------
    #  selective inference  (Car: main only, Ped: merged, Cyc: main only)
    # ------------------------------------------------------------------

    def _post_processing_selective(self, batch_dict):
        """
        Selective inference mode:

        - Car  (global 1): main head only.
        - Ped  (global 2): main + aux merged → class-wise NMS.
        - Cyc  (global 3): main head only (aux Cyclist excluded).

        Used to isolate whether Cyclist degradation comes from the
        auxiliary head's Cyclist predictions.
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []

        MAIN_CAR, MAIN_PED, MAIN_CYC = 0, 1, 2
        AUX_PED = 0  # aux channel 0 = Pedestrian
        GLOBAL_CAR, GLOBAL_PED, GLOBAL_CYC = 1, 2, 3

        for index in range(batch_size):
            batch_mask = index

            main_cls = batch_dict['batch_cls_preds'][batch_mask]
            main_box = batch_dict['batch_box_preds'][batch_mask]
            if not batch_dict['cls_preds_normalized']:
                main_cls = torch.sigmoid(main_cls)

            aux_cls = batch_dict['aux_batch_cls_preds'][batch_mask]
            aux_box = batch_dict['aux_batch_box_preds'][batch_mask]
            if not batch_dict['aux_cls_preds_normalized']:
                aux_cls = torch.sigmoid(aux_cls)

            nms_cfg = post_process_cfg.NMS_CONFIG
            score_th = post_process_cfg.SCORE_THRESH

            # -- Car: main only --
            car_sel, car_final_scores = model_nms_utils.class_agnostic_nms(
                box_scores=main_cls[:, MAIN_CAR], box_preds=main_box,
                nms_config=nms_cfg, score_thresh=score_th
            )
            car_boxes = main_box[car_sel]
            car_labels = car_boxes.new_full((car_boxes.shape[0],), GLOBAL_CAR, dtype=torch.long)

            # -- Pedestrian: main + aux merged --
            ped_boxes, ped_scores, ped_labels = self._merge_class(
                main_cls[:, MAIN_PED], main_box,
                aux_cls[:, AUX_PED], aux_box,
                GLOBAL_PED, nms_cfg, score_th
            )

            # -- Cyclist: main only (skip aux Cyc) --
            cyc_sel, cyc_final_scores = model_nms_utils.class_agnostic_nms(
                box_scores=main_cls[:, MAIN_CYC], box_preds=main_box,
                nms_config=nms_cfg, score_thresh=score_th
            )
            cyc_boxes = main_box[cyc_sel]
            cyc_labels = cyc_boxes.new_full((cyc_boxes.shape[0],), GLOBAL_CYC, dtype=torch.long)

            final_boxes = torch.cat([car_boxes, ped_boxes, cyc_boxes], dim=0)
            final_scores = torch.cat([car_final_scores, ped_scores, cyc_final_scores], dim=0)
            final_labels = torch.cat([car_labels, ped_labels, cyc_labels], dim=0)

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            pred_dicts.append({
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
            })

        return pred_dicts, recall_dict

    @staticmethod
    def _merge_class(main_scores, main_boxes, aux_scores, aux_boxes,
                     global_label, nms_cfg, score_thresh):
        """Merge main + aux predictions for one class, then NMS."""
        main_sel, _ = model_nms_utils.class_agnostic_nms(
            box_scores=main_scores, box_preds=main_boxes,
            nms_config=nms_cfg, score_thresh=score_thresh
        )
        aux_sel, _ = model_nms_utils.class_agnostic_nms(
            box_scores=aux_scores, box_preds=aux_boxes,
            nms_config=nms_cfg, score_thresh=score_thresh
        )

        n_main, n_aux = len(main_sel), len(aux_sel)
        if n_main + n_aux == 0:
            return (main_boxes.new_zeros(0, 7),
                    main_scores.new_zeros(0),
                    main_boxes.new_zeros(0, dtype=torch.long))

        all_boxes = torch.cat([main_boxes[main_sel][:, :7],
                                aux_boxes[aux_sel][:, :7]], dim=0)
        all_scores = torch.cat([main_scores[main_sel],
                                 aux_scores[aux_sel]], dim=0)

        keep, final_scores = model_nms_utils.class_agnostic_nms(
            box_scores=all_scores, box_preds=all_boxes,
            nms_config=nms_cfg, score_thresh=score_thresh
        )

        final_boxes = all_boxes[keep]
        final_scores = all_scores[keep]
        final_labels = final_boxes.new_full((final_boxes.shape[0],), global_label,
                                            dtype=torch.long)
        return final_boxes, final_scores, final_labels
