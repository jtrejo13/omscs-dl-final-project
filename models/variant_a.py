# ------------------------------------------------------------------------
# Variant A: GELU Gate
#
# Spec:
#   - Replace SimpleGate's activation-free gate (x1 * x2) with a GELU-gated
#     variant (x1 * F.gelu(x2)) inside NAFBlock.
#   - To test the role of the nonlinear activation in the gating mechanism.
# ------------------------------------------------------------------------


class VariantA:
    """Variant A — GELU Gate"""

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
