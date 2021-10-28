from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from .build import HEAD_REGISTRY


@HEAD_REGISTRY.register('simple_head_module')
def simple_head_module(cfg):
    layers = []
    layers.append(Dropout(cfg.SHM.DROPOUT))
    layers.append(Dense(units=cfg.SHM.OUTPUT_LABELS))
    layers.append(Activation(cfg.SHM.ACTIVATION))
    return layers
    