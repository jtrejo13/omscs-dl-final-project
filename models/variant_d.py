from models.baseline_nafnet import BaselineNAFNet
from models.archs.nafnet_arch import NAFBlockD


class VariantD(BaselineNAFNet):
    """Variant D: Asymmetric SimpleGate (2C/3 / C/3 split).

    Replaces the baseline 1:1 SimpleGate split with a 2:1 split, giving
    the value path 2x the channels of the gate.
    Implemented with DW_Expand=FFN_Expand=3 so 3c channels split exactly into [2c, c].
    """

    def __init__(self, opt: dict):
        super().__init__(opt, block_cls=NAFBlockD)
