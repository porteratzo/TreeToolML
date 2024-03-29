from fvcore.common.config import CfgNode

# YACS overwrite these settings using YAML
_C = CfgNode()

_C._BASE_ = ""

_C.FILES = CfgNode()
_C.FILES.DATA_SET = "datasets/custom_data"
_C.FILES.DATA_WORK_FOLDER = ""
_C.FILES.RESULT_FOLDER = "FILE"

_C.VALIDATION = CfgNode()
_C.VALIDATION.PATH = 'datasets/custom_data/PDE/validating_data'
_C.VALIDATION.BATCH_SIZE = 12

_C.TRAIN = CfgNode()
_C.TRAIN.MODEL_NAME = 'FILE'
_C.TRAIN.N_POINTS = 4096
_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 12
_C.TRAIN.PATH = 'datasets/custom_data/PDE/training_data'
_C.TRAIN.FILES = ['open','paris','tropical']
_C.TRAIN.DISTANCE = 0
_C.TRAIN.DISTANCE_LOSS = 0
_C.TRAIN.SCALED_DIST = 0

_C.TRAIN.HYPER_PARAMETERS = CfgNode()
_C.TRAIN.HYPER_PARAMETERS.LR = 0.001
_C.TRAIN.HYPER_PARAMETERS.DECAY_RATE = 0.95

_C.TRAIN.LOSS = CfgNode()
_C.TRAIN.LOSS.L2 = 0.095


_C.MODEL = CfgNode()
_C.MODEL.MODEL_NAME = 'RRFSegNet'
_C.MODEL.OUTPUT_NODS = 3
_C.MODEL.LAST_BATCHNORM = 1
_C.MODEL.SIGMOID = 0
_C.MODEL.CLASS_SIGMOID = 0

# data augmentation parameters with albumentations library
_C.DATA_PREPROCESSING = CfgNode()
_C.DATA_PREPROCESSING.DATA_PATH = "datasets/custom_data"
_C.DATA_PREPROCESSING.PC_FILTER = 0
_C.DATA_PREPROCESSING.DISTANCE_FILTER = 0.0


_C.DATA_CREATION = CfgNode()
_C.DATA_CREATION.USE_CENTER_FILTERED = 0
_C.DATA_CREATION.CENTER_METHOD = 0
_C.DATA_CREATION.TRAIN_AMOUNT = 10000
_C.DATA_CREATION.VAL_AMOUNT = 2000
_C.DATA_CREATION.TEST_AMOUNT = 2000
_C.DATA_CREATION.MIN_SIZE = 4096
_C.DATA_CREATION.AUGMENTATION = CfgNode()
_C.DATA_CREATION.AUGMENTATION.MAX_TREES = 4
_C.DATA_CREATION.AUGMENTATION.TRANSLATION_XY = 4.0
_C.DATA_CREATION.AUGMENTATION.TRANSLATION_Z = 0.2
_C.DATA_CREATION.AUGMENTATION.MIN_HEIGHT = 2
_C.DATA_CREATION.AUGMENTATION.MAX_HEIGHT = 6
_C.DATA_CREATION.AUGMENTATION.XY_ROTATION = 0.0
_C.DATA_CREATION.AUGMENTATION.MIN_DIST_BETWEEN = 3.0
_C.DATA_CREATION.AUGMENTATION.DO_NORMALIZE = 0
_C.DATA_CREATION.AUGMENTATION.ZERO_FLOOR = 1
_C.DATA_CREATION.STICK = 0
_C.DATA_CREATION.NOISE = 0.0

_C.BENCHMARKING = CfgNode()
_C.BENCHMARKING.WINDOW_STRIDE = 8
_C.BENCHMARKING.OVERLAP = 0.2
_C.BENCHMARKING.COMBINE_IOU = 1
_C.BENCHMARKING.COMBINE_STEMS = 0
_C.BENCHMARKING.XY_THRESHOLD = 80
_C.BENCHMARKING.ANGLE_THRESHOLD = 20
_C.BENCHMARKING.GROUP_STEMS = 0
_C.BENCHMARKING.VOXEL_SIZE = 0.08
_C.BENCHMARKING.CENTER_DETECTION_ENABLE = 1
_C.BENCHMARKING.VERTICALITY = 0.08
_C.BENCHMARKING.CURVATURE = 0.12

