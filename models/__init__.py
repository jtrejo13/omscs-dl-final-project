from .dummy import DummyRestorationModel


def build_model(opt):
    model_type = opt.get("model", {}).get("type", "dummy").lower()

    if model_type == "dummy":
        return DummyRestorationModel(opt)
    elif model_type == "baseline":
        from .baseline_nafnet import BaselineNAFNet
        return BaselineNAFNet(opt)
    elif model_type == "variant_a":
        from .variant_a import VariantA
        return VariantA(opt)
    elif model_type == "variant_b":
        from .variant_b import VariantB
        return VariantB(opt)
    elif model_type == "variant_c":
        from .variant_c import VariantC
        return VariantC(opt)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")
