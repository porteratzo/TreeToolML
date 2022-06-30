from fvcore.common.config import CfgNode
from os.path import exists
import os

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()


def combine_cfgs(cfg_path, opt=[]):
    # Priority 3: get default configs
    cfg_base = get_cfg()

    # Priority 2: merge from yaml config
    if cfg_path is not None and exists(cfg_path):
        cfg_base.merge_from_file(cfg_path)
    else:
        raise Exception('cfg not found')

    if opt is not []:
        cfg_base.merge_from_list(opt)

    # Load variables
    path_overwrite_keys = []

    if path_overwrite_keys is not []:
        cfg_base.merge_from_list(path_overwrite_keys)

    if cfg_base.TRAIN.MODEL_NAME == 'FILE':
        cfg_base.TRAIN.MODEL_NAME = os.path.splitext(os.path.basename(cfg_path))[0]

    if cfg_base.FILES.RESULT_FOLDER == 'FILE':
        cfg_base.FILES.RESULT_FOLDER = os.path.splitext(os.path.basename(cfg_path))[0]


    cfg_base.freeze()

    return cfg_base
