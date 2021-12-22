from fvcore.common.config import CfgNode

# YACS overwrite these settings using YAML
_C = CfgNode()

_C._BASE_ = ""

_C.VALIDATION = CfgNode()
_C.VALIDATION.PATH = 'datasets/custom_data/PDE/validating_data'
_C.VALIDATION.BATCH_SIZE = 12

_C.TRAIN = CfgNode()
_C.TRAIN.MODEL_NAME = 'result'
_C.TRAIN.N_POINTS = 4096
_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 12
_C.TRAIN.PATH = 'datasets/custom_data/PDE/training_data'

_C.TRAIN.HYPER_PARAMETERS = CfgNode()
_C.TRAIN.HYPER_PARAMETERS.LR = 0.001
_C.TRAIN.HYPER_PARAMETERS.DECAY_RATE = 0.95

_C.TRAIN.LOSS = CfgNode()
_C.TRAIN.LOSS.L2 = 0.095

_C.MODEL = CfgNode()
_C.MODEL.MODEL_NAME = 'RRFSegNet'

# data augmentation parameters with albumentations library
_C.DATA_PREPROCESSING = CfgNode()
_C.DATA_PREPROCESSING.DATA_PATH = "datasets/custom_data"

_C.DATA_CREATION = CfgNode()
_C.DATA_CREATION.CENTER_METHOD = 0
_C.DATA_CREATION.TRAIN_AMOUNT = 10000
_C.DATA_CREATION.TEST_AMOUNT = 2000
_C.DATA_CREATION.MIN_SIZE = 4096
_C.DATA_CREATION.AUGMENTATION = CfgNode()
_C.DATA_CREATION.AUGMENTATION.MAX_TREES = 4
_C.DATA_CREATION.AUGMENTATION.TRANSLATION_XY = 4.0
_C.DATA_CREATION.AUGMENTATION.TRANSLATION_Z = 0.2
_C.DATA_CREATION.AUGMENTATION.SCALE = 0.2
_C.DATA_CREATION.AUGMENTATION.XY_ROTATION = 0.0
_C.DATA_CREATION.AUGMENTATION.MIN_DIST_BETWEEN = 3.0
_C.DATA_CREATION.AUGMENTATION.DO_NORMALIZE = 0
_C.DATA_CREATION.AUGMENTATION.ZERO_FLOOR = 1