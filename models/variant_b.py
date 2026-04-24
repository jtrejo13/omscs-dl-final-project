# ------------------------------------------------------------------------
# Variant B: No SCA (Simplified Channel Attention)
#
# Spec:
#   - Remove the SCA module from NAFBlock entirely.
#   - Delete self.sca in __init__ and remove `x = x * self.sca(x)` from forward to measure
#     the contribution of channel attention to denoising quality.
# ------------------------------------------------------------------------


class VariantB:
    """Variant B — No SCA"""

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
