import os

from .register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2017_train_nonvoc": ("coco/train2017", "coco/new_annotations/final_split_non_voc_instances_train2017.json"),
    "coco_2017_train_voc_10_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_10_shot_instances_train2017.json"),
}

def register_all_coco(root):
    """
    Register all coco images.

    Args:
        root: (todo): write your description
    """
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
