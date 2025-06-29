from .build_arch import build_arch


def build_model(cfg):
    model = build_arch(cfg.MODEL)
    return model
