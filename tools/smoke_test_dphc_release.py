#!/usr/bin/env python3
"""Dataset-free release checks for DPHC-PointPillars."""
import ast
import importlib.util
import os
import sys
import types
from pathlib import Path
import torch
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def load_lightweight_model_classes():
    packages = {
        "pcdet.models": REPO_ROOT / "pcdet/models",
        "pcdet.models.backbones_3d": REPO_ROOT / "pcdet/models/backbones_3d",
        "pcdet.models.backbones_3d.vfe": REPO_ROOT / "pcdet/models/backbones_3d/vfe",
        "pcdet.models.backbones_2d": REPO_ROOT / "pcdet/models/backbones_2d",
    }
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    load_source_module(
        "pcdet.models.backbones_3d.vfe.vfe_template",
        REPO_ROOT / "pcdet/models/backbones_3d/vfe/vfe_template.py",
    )
    pillar = load_source_module(
        "pcdet.models.backbones_3d.vfe.pillar_vfe",
        REPO_ROOT / "pcdet/models/backbones_3d/vfe/pillar_vfe.py",
    )
    aux = load_source_module(
        "pcdet.models.backbones_2d.aux_bev_backbone",
        REPO_ROOT / "pcdet/models/backbones_2d/aux_bev_backbone.py",
    )
    return pillar.PillarVFE, aux.AuxBEVBackboneFullResContext

PillarVFE, AuxBEVBackboneFullResContext = load_lightweight_model_classes()
TOOLS_DIR = REPO_ROOT / "tools"
CONFIGS = [
    "pointpillar.yaml", "pointpillar_dpe.yaml", "pointpillar_dphc.yaml",
    "pointpillar_dphc_no_density.yaml", "pointpillar_dphc_identity.yaml",
    "pointpillar_dphc_stride2.yaml", "pointpillar_dphc_single_dilation.yaml",
    "pointpillar_dphc_ped_cyc.yaml", "pointpillar_dpe_trainval.yaml",
    "pointpillar_dphc_trainval.yaml", "pointpillar_dphc_test.yaml",
]

def check_configs():
    old = Path.cwd()
    os.chdir(TOOLS_DIR)
    try:
        for name in CONFIGS:
            config = EasyDict()
            cfg_from_yaml_file(str(Path("cfgs/kitti_models") / name), config)
            assert "MODEL" in config and "OPTIMIZATION" in config, name
    finally:
        os.chdir(old)

def check_density_vfe():
    config = EasyDict({"USE_NORM": True, "WITH_DISTANCE": False,
        "USE_ABSLOTE_XYZ": True, "USE_DENSITY_FEATURE": True,
        "NUM_FILTERS": [64]})
    vfe = PillarVFE(config, 4, [0.16, 0.16, 4.0],
        [0, -39.68, -3, 69.12, 39.68, 1]).eval()
    assert vfe.pfn_layers[0].linear.in_features == 11
    voxels = torch.zeros(2, 32, 4)
    voxels[0, :4] = torch.rand(4, 4)
    voxels[1, :8] = torch.rand(8, 4)
    batch = {"voxels": voxels, "voxel_num_points": torch.tensor([4, 8]),
        "voxel_coords": torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]])}
    with torch.no_grad():
        output = vfe(batch)["pillar_features"]
    assert output.shape == (2, 64)

def check_pfcb_shape():
    config = EasyDict({"HIDDEN_CHANNELS": 32, "DILATIONS": [1, 2],
        "USE_BATCH_NORM": True, "USE_PLAIN_RESIDUAL": True})
    module = AuxBEVBackboneFullResContext(config, 64).eval()
    x = torch.randn(1, 64, 16, 16)
    with torch.no_grad():
        y = module({"spatial_features_fullres": x})
    assert y.shape == x.shape

def check_cpf_contract():
    path = REPO_ROOT / "pcdet/models/detectors/pointpillar_ped_aux.py"
    source = path.read_text()
    tree = ast.parse(source)
    assert any(isinstance(n, ast.FunctionDef) and
        n.name == "_post_processing_ped_selective_strict"
        for n in ast.walk(tree))
    assert "non_ped_mask = main_pred['pred_labels'] != 2" in source
    assert "PED_SCORE_THRESH" in source
    aux_source = (REPO_ROOT / "pcdet/models/detectors/pointpillar_aux.py").read_text()
    assert "param.requires_grad = False" in aux_source
    assert "with torch.no_grad():" in aux_source
    assert "module.eval()" in aux_source

if __name__ == "__main__":
    check_configs()
    check_density_vfe()
    check_pfcb_shape()
    check_cpf_contract()
    print("DPHC release smoke tests passed.")
