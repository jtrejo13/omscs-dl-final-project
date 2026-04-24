from .dummy import DummyRestorationModel


def build_model(opt):
    model_type = opt.get("model", {}).get("type", "dummy").lower()

    if model_type == "dummy":
        return DummyRestorationModel(opt)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")
