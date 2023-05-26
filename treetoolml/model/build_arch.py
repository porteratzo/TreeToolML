from treetoolml.utils.REGISTRY import Registry
ARCH_REGISTRY=Registry()


def build_arch(arch_cfg):
    arch_module = ARCH_REGISTRY[arch_cfg.MODEL_NAME](arch_cfg)
    return arch_module
