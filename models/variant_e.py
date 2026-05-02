from models.baseline_nafnet import BaselineNAFNet
from models.archs.nafnet_arch import NAFBlockE


class VariantE(BaselineNAFNet):
    """Variant E: Asymmetric SimpleGate with 2:1 channel split.

    Tests whether the symmetry of the gate halves matters. The depthwise
    and FFN gates each split (2c, c) and project the smaller half up to
    2c via a 1x1 conv before element-wise multiplication, so DW_Expand
    and FFN_Expand are 3 (vs. 2 in baseline).
    """

    def __init__(self, opt: dict):
        super().__init__(opt, block_cls=NAFBlockE)
