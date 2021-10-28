from yacs.config import CfgNode
from os.path import exists

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

    if opt is not []:
        cfg_base.merge_from_list(opt)

    # Load variables
    path_overwrite_keys = []

    if path_overwrite_keys is not []:
        cfg_base.merge_from_list(path_overwrite_keys)

    cfg_base.freeze()

    return cfg_base
