from typing import Final
from dataclasses import dataclass


NO_LABEL: Final = "(No_Label)"


# Task names
@dataclass(frozen=True)
class TaskName:
    CLASSIFICATION: Final = "classification"
    SEGMENTATION: Final = "segmentation"
    IMAGE_MATCHING: Final = "image_matching"


# UI Type
@dataclass(frozen=True)
class UiType:
    PYQT: Final = "pyqt"
    HARD_CODING: Final = "hard_coding"


# DataField (Commonly Used)
@dataclass(frozen=True)
class BasicDataKey:
    PURPOSE: Final = "purpose"
    ID: Final = "id"


# Purpose
@dataclass(frozen=True)
class PurposeType:
    NOT_SET: Final = "not_set"
    TRAIN: Final = "train"
    VALID: Final = "valid"
    TEST: Final = "test"


# Learning Rate Params
@dataclass(frozen=True)
class LrParams:
    LR_POLICY: Final = "lr_policy"
    DECAY_PERIOD: Final = "epoch_counts"
    DECAY_RATE: Final = "decay_rate"
    MAX_STEPS: Final = "max_steps"
    INIT_PERIOD: Final = "init_period"
    PERIOD_MULT_FACTOR: Final = "period_multiply_factor"


# Learning Rate Policy
@dataclass(frozen=True)
class LrPolicy:
    STEP_DECAY: Final = "step_decay"
    COS_ANNEAL: Final = "cos_annealing"
    COS_RESTART: Final = "cos_restart"
    ONE_CYCLE: Final = "one_cycle"


@dataclass(frozen=True)
class DatasetParam:
    LENGTH: Final = "length"
    WIDTH: Final = "width"
    HEIGHT: Final = "height"
    CHANNEL: Final = "channel"
    DATASET_FORMAT: Final = "dataset_format"

# Optimizer
@dataclass(frozen=True)
class OptimizerName:
    SGD: Final = "SGD"
    NAG: Final = "NAG"
    ADAM: Final = "ADAM"
    ADAMW: Final = "ADAMW"
    RADAM: Final = "RADAM"


# Backbone
@dataclass(frozen=True)
class BackboneType:
    RESNET_18: Final = "resnet_18"
    RESNET_H_18: Final = "resnet_H_18"
    RESNET_34: Final = "resnet_34"
    RESNET_50: Final = "resnet_50"


# Preprocess Name
@dataclass(frozen=True)
class PreprocessName:
    SCALE_1_OF_255: Final = "scale_1_of_255"
    AFFINE_MINUS1_TO_PLUS1: Final = "affine_minus1_to_plus1"
    MIN_MAX_STRETCHING: Final = "min_max_stretching"
    DATA_WISE_STANDARDIZATION: Final = "data_wise_standardization"
    STATIC_RESIZE: Final = "static_resize"
    STATIC_CROP_RESIZE: Final = "static_crop_resize"
    ROTATION_90: Final = "rotation_90"
    ROTATION: Final = "rotation"
    FLIP_LEFT_RIGHT: Final = "flip_left_right"
    FLIP_UP_DOWN: Final = "flip_up_down"
    MIX_UP: Final = "mix_up"
    CUT_MIX: Final = "cut_mix"
    JPEG_LOSSY: Final = "jpeg_lossy"
    ZOOM: Final = "zoom"
    BRIGHT: Final = "bright"
    RANDOM_CROP_RESIZE: Final = "random_crop_resize"
    PATCH_BRIGHT: Final = "patch_bright"


# Activation Function
@dataclass(frozen=True)
class ActFunc:
    IDENTITY: Final = "identity"
    RELU: Final = "relu"
    LEAKY_RELU: Final = "leaky_relu"
    SIGMOID: Final = "sigmoid"
    TANH: Final = "tanh"
    MISH: Final = "mish"


# Normalization Layer
@dataclass(frozen=True)
class NormLayer:
    NONE: Final = "none"
    BATCH_NORM: Final = "batch_norm"
    # LAYER_NORM: Final = "layer_norm"  # input param 으로 h, w가 더 필요.
    INSTANCE_NORM: Final = "instance_norm"
    INSTANCE_NORM_F: Final = "instance_norm_fixed"
