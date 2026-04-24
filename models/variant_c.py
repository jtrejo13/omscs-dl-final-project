# ------------------------------------------------------------------------
# Variant C: BatchNorm
#
# Spec:
#   - Replace LayerNorm2d with nn.BatchNorm2d for norm1 and norm2 inside NAFBlock.
#   - Test whether the normalization strategy (layer vs. batch) affects restoration quality.
# ------------------------------------------------------------------------


class VariantC:
    """Variant C — BatchNorm"""

    def __init__(self, opt: dict):
        raise NotImplementedError()

    def feed_data(self, data):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def get_current_visuals(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
