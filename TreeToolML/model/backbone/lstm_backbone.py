from tensorflow.keras.layers import (
    LSTM,
    TimeDistributed,
    Conv1D,
    MaxPooling1D,
    Dropout,
    Flatten,
    Reshape,
    ConvLSTM2D
)
from .build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register("simple_lstm")
def simple_lstm(cfg, input_shape):
    layers = []
    for n, units in enumerate(cfg.LSTM.UNITS):
        return_sequences = (
            cfg.LSTM.RETURN_SEQUENCES if n == len(cfg.LSTM.UNITS) - 1 else 1
        )
        if n == 0:
            layers.append(
                LSTM(units, return_sequences=return_sequences, input_shape=input_shape,)
            )
        else:
            layers.append(LSTM(units, return_sequences=return_sequences,))
    if cfg.LSTM.RETURN_SEQUENCES:
        layers.append(Flatten())
    return layers


@BACKBONE_REGISTRY.register("cnn_lstm")
def cnn_lstm(cfg, input_shape):
    # define model
    layers = []
    subsequences = cfg.CONV_LSTM.SUBSEQUENCES
    n_length, n_features = input_shape
    assert (
        n_length % subsequences == 0
    ), "LOADER.N_FRAMES must be divisable in MODEL.BACKBONE.CONV_LSTM.SUBSEQUENCES"

    layers.append(
        Reshape(
            (subsequences, int(n_length / subsequences), n_features),
            input_shape=input_shape,
        )
    )
    assert len(cfg.CONV_LSTM.CNN_KERNELS) == len(
        cfg.CONV_LSTM.CNN_FILTERS
    ), "MODEL.BACKBONE.CONV_LSTM.CNN_KERNELS and MODEL.BACKBONE.CONV_LSTM.CNN_FILTERS need to be the same size"
    for n in range(len(cfg.CONV_LSTM.CNN_FILTERS)):
        layers.append(
            TimeDistributed(
                Conv1D(
                    filters=cfg.CONV_LSTM.CNN_FILTERS[n],
                    kernel_size=cfg.CONV_LSTM.CNN_KERNELS[n],
                    activation="relu",
                ),
            )
        )
    layers.append(TimeDistributed(Dropout(cfg.CONV_LSTM.DROPOUT)))
    layers.append(TimeDistributed(MaxPooling1D(pool_size=cfg.CONV_LSTM.POOLSIZE)))
    layers.append(TimeDistributed(Flatten()))
    layers.append(LSTM(cfg.CONV_LSTM.LSTM_UNITS))
    return layers


@BACKBONE_REGISTRY.register("convlstm_1ddata")
def convlstm_1ddata(cfg, input_shape):
    # define model
    layers = []
    subsequences = cfg.CONVLSTM.SUBSEQUENCES
    assert len(input_shape) == 2, f'the input shape has an incorrect number of dimensions, should be [n_length, n_features] got {input_shape}'
    n_length, n_features = input_shape
    assert (
        n_length % subsequences == 0
    ), "LOADER.N_FRAMES must be divisable in MODEL.BACKBONE.CONV_LSTM.SUBSEQUENCES"

    layers.append(
        Reshape(
            (subsequences, 1, int(n_length / subsequences), n_features),
            input_shape=input_shape,
        )
    )
    layers.append(
        ConvLSTM2D(
                filters=cfg.CONVLSTM.CNN_FILTERS,
                kernel_size=cfg.CONVLSTM.CNN_KERNELS,
                activation="relu",
        )
    )
    layers.append(Flatten())
    return layers


@BACKBONE_REGISTRY.register("convlstm_2ddata")
def convlstm_2ddata(cfg, input_shape):
    # define model
    layers = []
    layers.append(
        ConvLSTM2D(
                filters=cfg.CONVLSTM.CNN_FILTERS,
                kernel_size=cfg.CONVLSTM.CNN_KERNELS,
                activation="relu", input_shape=input_shape
        )
    )
    layers.append(Flatten())
    return layers
