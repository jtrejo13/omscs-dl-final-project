from models.baseline_nafnet import BaselineNAFNet
from models.archs.nafnet_arch import NAFBlockB


class VariantB(BaselineNAFNet):
    """Variant B: SCA removed.

    Tests how much the Simplified Channel Attention contributes to
    denoising quality by ablating it from the baseline NAFBlock.
    """

    def __init__(self, opt: dict):
        super().__init__(opt, block_cls=NAFBlockB)
