from models.baseline_nafnet import BaselineNAFNet
from models.archs.nafnet_arch import NAFBlockA


class VariantA(BaselineNAFNet):
    """Variant A: GELU gate (x1 * gelu(x2)) instead of SimpleGate (x1 * x2).

    Tests whether a nonlinear activation in the gating mechanism
    improves over the activation-free baseline.
    """

    def __init__(self, opt: dict):
        super().__init__(opt, block_cls=NAFBlockA)
