from models.baseline_nafnet import BaselineNAFNet
from models.archs.nafnet_arch import NAFBlockC


class VariantC(BaselineNAFNet):
    """Variant C: BatchNorm instead of LayerNorm2d.

    Tests whether the normalization strategy (layer vs. batch) affects
    restoration quality on SIDD.
    """

    def __init__(self, opt: dict):
        super().__init__(opt, block_cls=NAFBlockC)
