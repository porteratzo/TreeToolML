from inter_det.utils.REGISTRY import Registry
HEAD_REGISTRY=Registry()

def build_head(head_cfg):
    head_module = HEAD_REGISTRY[head_cfg.NAME](head_cfg)
    return head_module
