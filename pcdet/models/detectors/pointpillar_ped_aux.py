"""
PointPillarPedestrianAux — Exp5: Pedestrian-only High-Resolution Context Branch.

Inherits from PointPillarSmallAux.  Same frozen main-path + fullres_context
aux backbone as Exp3, but the aux detection head only predicts Pedestrian
(single class) instead of Pedestrian + Cyclist.

New inference mode:  ped_selective
  - Car   : main head only
  - Ped   : main + aux merged → class-wise NMS
  - Cyc   : main head only

Does NOT modify any existing Exp1/2/3/4 behaviour or checkpoints.
"""

import copy
import numpy as np
import torch
import torch.nn as nn

from .pointpillar_aux import PointPillarSmallAux
from ..backbones_2d.aux_bev_backbone import (
    AuxBEVBackboneBlock1,
    AuxBEVBackboneFullRes,
    AuxBEVBackboneFullResContext,
)
from ..dense_heads.pedestrian_aux_anchor_head_single import PedestrianAuxAnchorHeadSingle
from ..model_utils import model_nms_utils


class PointPillarPedestrianAux(PointPillarSmallAux):
    """
    Exp5: Pedestrian-only auxiliary branch.

    Identical to PointPillarSmallAux except:
      - aux head is PedestrianAuxAnchorHeadSingle (1 class, not 2)
      - adds ped_selective inference mode
      - config uses AUX_CLASS_NAMES: ['Pedestrian'] with NUM_CLASS: 1
    """

    def _build_aux_head(self):
        """Override: use PedestrianAuxAnchorHeadSingle (1 class)."""
        grid_size = self.dataset.grid_size
        aux_grid_size = np.array([grid_size[0], grid_size[1], grid_size[2]])
        aux_head_input_channels = self.aux_backbone.num_bev_features
        point_cloud_range = self.dataset.point_cloud_range
        aux_head_cfg = self._build_aux_head_cfg()

        self.aux_dense_head = PedestrianAuxAnchorHeadSingle(
            model_cfg=aux_head_cfg,
            input_channels=aux_head_input_channels,
            num_class=self.aux_num_class,   # 1
            class_names=self.aux_class_names,  # ['Pedestrian']
            grid_size=aux_grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=True,
        )

    def _build_aux_head_cfg(self):
        """Override: keep Pedestrian anchors only, no Cyclist."""
        head_cfg = copy.deepcopy(self.model_cfg.DENSE_HEAD)

        all_anchor_configs = head_cfg.ANCHOR_GENERATOR_CONFIG
        aux_anchor_configs = [
            acfg for acfg in all_anchor_configs
            if acfg['class_name'] in self.aux_class_names
        ]

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
    #  Post-processing: add ped_selective
    # ------------------------------------------------------------------

    def post_processing(self, batch_dict):
        if not self.aux_enabled:
            return super(PointPillarSmallAux, self).post_processing(batch_dict)

        mode = self.aux_inference_mode
        if mode == 'main_only':
            return super(PointPillarSmallAux, self).post_processing(batch_dict)
        elif mode == 'aux_only':
            return self._post_processing_aux_only(batch_dict)
        elif mode == 'ped_selective':
            return self._post_processing_ped_selective_strict(batch_dict)
        elif mode in ('merged', 'selective'):
            return self._post_processing_selective(batch_dict)
        else:
            raise ValueError(f"Unknown AUX_INFERENCE_MODE: '{mode}'")

    def _post_processing_ped_selective_strict(self, batch_dict):
        """Keep the main Car/Cyclist output exactly and replace only Pedestrian.

        The original selective path re-ran NMS independently for all three
        classes. That changed low-score Car/Cyclist predictions even though
        both classes were intended to come from the frozen main detector. In
        strict mode, the standard main post-processing is executed first and
        its Car/Cyclist predictions are copied unchanged. Only Pedestrian is
        produced by the main/auxiliary class-wise fusion path.
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        ped_score_thresh = float(
            self.aux_cfg.get('PED_SCORE_THRESH', post_process_cfg.SCORE_THRESH)
        )

        main_pred_dicts, _ = super(
            PointPillarSmallAux, self
        ).post_processing(batch_dict)

        recall_dict = {}
        pred_dicts = []
        for index, main_pred in enumerate(main_pred_dicts):
            main_cls = batch_dict['batch_cls_preds'][index]
            main_box = batch_dict['batch_box_preds'][index]
            if not batch_dict['cls_preds_normalized']:
                main_cls = torch.sigmoid(main_cls)

            aux_cls = batch_dict['aux_batch_cls_preds'][index]
            aux_box = batch_dict['aux_batch_box_preds'][index]
            if not batch_dict['aux_cls_preds_normalized']:
                aux_cls = torch.sigmoid(aux_cls)

            ped_boxes, ped_scores, ped_labels = self._merge_class(
                main_cls[:, 1], main_box,
                aux_cls[:, 0], aux_box,
                2, post_process_cfg.NMS_CONFIG, ped_score_thresh,
            )

            non_ped_mask = main_pred['pred_labels'] != 2
            final_boxes = torch.cat([
                main_pred['pred_boxes'][non_ped_mask], ped_boxes
            ], dim=0)
            final_scores = torch.cat([
                main_pred['pred_scores'][non_ped_mask], ped_scores
            ], dim=0)
            final_labels = torch.cat([
                main_pred['pred_labels'][non_ped_mask], ped_labels
            ], dim=0)

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict,
                batch_index=index,
                data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST,
            )
            pred_dicts.append({
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
            })

        return pred_dicts, recall_dict

    # ------------------------------------------------------------------
    #  aux_only — Pedestrian only
    # ------------------------------------------------------------------

    def _post_processing_aux_only(self, batch_dict):
        """Return only aux-head Pedestrian predictions (global label 2)."""
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []

        for index in range(batch_size):
            batch_mask = index
            aux_cls_preds = batch_dict['aux_batch_cls_preds'][batch_mask]  # (N, 1)
            aux_box_preds = batch_dict['aux_batch_box_preds'][batch_mask]
            if not batch_dict['aux_cls_preds_normalized']:
                aux_cls_preds = torch.sigmoid(aux_cls_preds)

            # aux channel 0 = Ped (global 2)
            cls_preds_max = aux_cls_preds[:, 0]
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds_max, box_preds=aux_box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )
            final_boxes = aux_box_preds[selected]
            final_scores = selected_scores
            final_labels = final_boxes.new_full(
                (final_boxes.shape[0],), 2, dtype=torch.long  # global Ped = 2
            )

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes, recall_dict=recall_dict, batch_index=index,
                data_dict=batch_dict, thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
            pred_dicts.append({
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
            })
        return pred_dicts, recall_dict
