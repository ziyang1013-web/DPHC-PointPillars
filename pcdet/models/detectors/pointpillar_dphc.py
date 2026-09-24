"""Public entry point for DPHC-PointPillars."""

from .pointpillar_ped_aux import PointPillarPedestrianAux


class PointPillarDPHC(PointPillarPedestrianAux):
    """DPE main path with PFCB and class-selective pedestrian fusion.

    The legacy ``PointPillarPedestrianAux`` name remains checkpoint-compatible.
    """
    pass
