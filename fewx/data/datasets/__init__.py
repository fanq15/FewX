from . import builtin  # ensure the builtin datasets are registered
from .register_coco import register_coco_instances

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
