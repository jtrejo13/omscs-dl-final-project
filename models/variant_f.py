from models.baseline_nafnet import BaselineNAFNet


class VariantF(BaselineNAFNet):
    """Variant F: Gated Feature Fusion in U-Net decoder skip connections.

    Replaces additive skips (x + enc_skip) with a learned per-channel
    sigmoid gate: x + sigmoid(conv1x1([x, enc_skip])) * enc_skip. Block
    architecture is unchanged from baseline NAFBlock; only the decoder
    fusion changes. Wired up via opt['model']['skip_fusion'] = 'gated'.
    """

    def __init__(self, opt: dict):
        super().__init__(opt)
