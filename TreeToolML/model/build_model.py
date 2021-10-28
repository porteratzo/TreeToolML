from .head.build import build_head
from .backbone.build import build_backbone
from tensorflow.keras.models import Sequential


def build_model(cfg, input_shape):
    model = Sequential()
    backbone_layers = build_backbone(cfg.MODEL.BACKBONE, input_shape)
    for n, i in enumerate(backbone_layers):
        model.add(i)
    head_layers = build_head(cfg.MODEL.HEAD)
    for n, i in enumerate(head_layers):
        model.add(i) 
    model.compile(
        loss=cfg.MODEL.SOLVER.LOSS.NAME,
        optimizer=cfg.MODEL.SOLVER.OPTIMIZER.NAME,
        metrics=cfg.MODEL.SOLVER.METRICS,
    )
    return model
