from inter_det.utils.REGISTRY import Registry
BACKBONE_REGISTRY=Registry()


def build_backbone(backbone_cfg, input_shape):
    backbone_module = BACKBONE_REGISTRY[backbone_cfg.NAME](backbone_cfg, input_shape)
    return backbone_module
