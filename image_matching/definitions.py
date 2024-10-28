from typing import Final
from dataclasses import dataclass


# Dataset Format
@dataclass(frozen=True)
class DatasetFormat:
    # Custom dataset
    # SELF_GEN: Final = "self_gen"
    # SELF_GEN_4PTS: Final = "self_gen_4pts"
    # UNLABELED_PAIR: Final = "unlabeled_pair"
    # LABELED_PAIR: Final = "labeled_pair"
    # ALIGNED_PAIR: Final = "aligned_pair"

    # Pre-defined dataset
    GOOGLE_MULTIMODAL: Final = "google_multimodal"
    MSCOCO_SELF_GEN: Final = "mscoco_self_gen"


# DataField
@dataclass(frozen=True)
class DataKey:
    IMAGE_PATH: Final = "image_path"
    IMAGE_PATH_MOVING: Final = "image_path_moving"
    IMAGE_PATH_FIXED: Final = "image_path_fixed"
    LABEL_PATH: Final = "label_path"
    LABEL_H: Final = "label_h"
    LABEL_H_PATCH: Final = "label_h_patch"

    IMAGE_MOVING: Final = "image_moving"
    IMAGE_FIXED: Final = "image_fixed"
    IMAGE_PATCH_MOVING: Final = "image_patch_moving"
    IMAGE_PATCH_FIXED: Final = "image_patch_fixed"
    H: Final = "h"
    H_PATCH: Final = "h_patch"
    RECT_PATCH: Final = "rect_patch"
    PERTURBS: Final = "perturbs"
    LABEL_MASK_MOVING: Final = "label_mask_moving"
    LABEL_MASK_FIXED: Final = "label_mask_fixed"
    LABEL_MASK_PATCH_MOVING: Final = "label_mask_patch_moving"
    LABEL_MASK_PATCH_FIXED: Final = "label_mask_patch_fixed"
    ROI_MASK_MOVING: Final = "roi_mask_moving"
    ROI_MASK_FIXED: Final = "roi_mask_fixed"
    ROI_MASK_PATCH_MOVING: Final = "roi_mask_patch_moving"
    ROI_MASK_PATCH_FIXED: Final = "roi_mask_patch_fixed"
    FLOW_FIELD: Final = "flow_field"  # CPU에서 만들면 느려서, elastic transform 일때만 미리 만들어 쓴다. 선형변환은 batch 꺼낸 후에 GPU에서 만든다.


# 2D Geometric Trans Type
@dataclass(frozen=True)
class GeoTransType:  # 자유도
    RIGID: Final = 3
    SIMILARITY: Final = 4
    AFFINE: Final = 6
    PERSPECTIVE: Final = 8


# Geo Trans Mat Gen Info Field
@dataclass(frozen=True)
class HomographyParam:
    GEO_TRANS_TYPE: Final = "geo_trans_type"
    ANGLE_RANGE: Final = "angle_range"  # degree
    SCALE_RANGE: Final = "scale_range"
    SHIFT_RANGE: Final = "shift_range"
    MAX_PERTURB: Final = "max_perturb"
    H11_RANGE: Final = "h11_range"
    H12_RANGE: Final = "h12_range"
    H13_RANGE: Final = "h13_range"
    H21_RANGE: Final = "h21_range"
    H22_RANGE: Final = "h22_range"
    H23_RANGE: Final = "h23_range"
    H31_RANGE: Final = "h31_range"
    H32_RANGE: Final = "h32_range"
