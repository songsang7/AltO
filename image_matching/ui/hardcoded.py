import os

import albumentations as alb

from common.definitions import LrPolicy, OptimizerName
from image_matching.dataset.selfgen_solo import SelfGenSolo
from image_matching.dataset.selfgen_aligned_pair import SelfGenAlignedPair


class HardCodedUI:
    def __init__(self):
        pass

    def _prepare_mscoco_data(self, is_train, is_pds_coco=False, is_elastic=False):  # pds : Photometric Distorted Synthetic
        if is_elastic:
            trans_param = {"alpha": 128*2,
                           "sigma": 128*0.08}
        else:
            trans_param = {"max_perturb": 32}

        dataset_path = os.path.join(os.getcwd(), "..", "..", "..", "Dataset", "MSCOCO")

        augmentations = []
        if is_pds_coco:
            augmentations.append(alb.ColorJitter(always_apply=True))
            augmentations.append(alb.ChannelShuffle(p=0.5))
        augmentations = alb.Compose(augmentations)

        dataset_train = dataset_valid = dataset_test = None
        if is_train:
            train_dataset_path = os.path.join(dataset_path, "train2014")
            dataset_train = SelfGenSolo("coco", "", train_dataset_path, ch=3, width_ori=320, height_ori=240, width_patch=128, height_patch=128, is_elastic_trans=is_elastic, trans_param=trans_param, augmentations=augmentations)

            valid_dataset_path = os.path.join(dataset_path, "val2014")
            dataset_valid = SelfGenSolo("coco", "", valid_dataset_path, ch=3, width_ori=320, height_ori=240, width_patch=128, height_patch=128, is_elastic_trans=is_elastic, trans_param=trans_param, augmentations=augmentations)
        else:
            test_dataset_path = os.path.join(dataset_path, "test2014")
            dataset_test = SelfGenSolo("coco", "", test_dataset_path, ch=3, width_ori=320, height_ori=240, width_patch=128, height_patch=128, is_elastic_trans=is_elastic, trans_param=trans_param, augmentations=augmentations)

        return dataset_train, dataset_valid, dataset_test

    def _prepare_gg_mm_data(self, is_train, is_gg_map):
        from image_matching.dataset.gg_mm import GoogleMultimodal

        dataset_path = os.path.join(os.getcwd(), "..", "..", "..", "Dataset", "DLKFM")
        middle_dir = "GoogleMap" if is_gg_map else "GoogleEarth"

        dataset_train = dataset_valid = dataset_test = None
        if is_train:
            moving_path = os.path.join(dataset_path, middle_dir, "train2014_input")
            fixed_path = os.path.join(dataset_path, middle_dir, "train2014_template")
            label_path = os.path.join(dataset_path, middle_dir, "train2014_label")
            dataset_train = GoogleMultimodal("", "", [moving_path, fixed_path, label_path])

            moving_path = os.path.join(dataset_path, middle_dir, "val2014_input")
            fixed_path = os.path.join(dataset_path, middle_dir, "val2014_template")
            label_path = os.path.join(dataset_path, middle_dir, "val2014_label")
            dataset_valid = GoogleMultimodal("", "", [moving_path, fixed_path, label_path])
        else:
            moving_path = os.path.join(dataset_path, middle_dir, "val2014_input")
            fixed_path = os.path.join(dataset_path, middle_dir, "val2014_template")
            label_path = os.path.join(dataset_path, middle_dir, "val2014_label")
            dataset_test = GoogleMultimodal("", "", [moving_path, fixed_path, label_path])

        return dataset_train, dataset_valid, dataset_test

    def _prepare_gg_mm_data2(self, is_train):
        from image_matching.dataset.selfgen_aligned_pair import SelfGenAlignedPair

        trans_param = {"max_perturb": 32}
        dataset_path = os.path.join(os.getcwd(), "..", "..", "..", "Dataset", "DLKFM")
        middle_dir = "GoogleMap"  # google earth는 template_original이 없다.

        dataset_train = dataset_valid = dataset_test = None
        if is_train:
            moving_path = os.path.join(dataset_path, middle_dir, "train2014_input")
            fixed_path = os.path.join(dataset_path, middle_dir, "train2014_template_original")
            dataset_train = SelfGenAlignedPair("", "", [moving_path, fixed_path], 3, 192, 192, 128, 128, False, trans_param)

            moving_path = os.path.join(dataset_path, middle_dir, "val2014_input")
            fixed_path = os.path.join(dataset_path, middle_dir, "val2014_template_original")
            dataset_valid = SelfGenAlignedPair("", "", [moving_path, fixed_path], 3, 192, 192, 128, 128, False, trans_param)
        else:
            moving_path = os.path.join(dataset_path, middle_dir, "val2014_input")
            fixed_path = os.path.join(dataset_path, middle_dir, "val2014_template_original")
            dataset_test = SelfGenAlignedPair("", "", [moving_path, fixed_path], 3, 192, 192, 128, 128, False, trans_param)

        return dataset_train, dataset_valid, dataset_test

    def _prepare_deep_nir_x100_dataset(self, is_train):
        trans_param = {"max_perturb": 32}
        dataset_path = os.path.join(os.getcwd(), "..", "..", "..", "Dataset", "nirscene_img_aug_100_oversample")

        dataset_train = dataset_valid = dataset_test = None
        if is_train:
            moving_path = os.path.join(dataset_path, "train_A")
            fixed_path = os.path.join(dataset_path, "train_B")
            dataset_train = SelfGenAlignedPair("", "", [moving_path, fixed_path], 3, 192, 192, 128, 128, False, trans_param)

            moving_path = os.path.join(dataset_path, "val_A")
            fixed_path = os.path.join(dataset_path, "val_B")
            dataset_valid = SelfGenAlignedPair("", "", [moving_path, fixed_path], 3, 192, 192, 128, 128, False, trans_param)
        else:
            moving_path = os.path.join(dataset_path, "test_A")
            fixed_path = os.path.join(dataset_path, "test_B")
            dataset_test = SelfGenAlignedPair("", "", [moving_path, fixed_path], 3, 192, 192, 128, 128, False, trans_param)

        return dataset_train, dataset_valid, dataset_test

    def train_alto(self):
        from image_matching.controller.alto import AltoDhn, AltoRaft, AltoIhn, AltoRhwf

        dataset_train, dataset_valid, dataset_test = self._prepare_gg_mm_data2(is_train=True)  # GG map
        # dataset_train, dataset_valid, dataset_test = self._prepare_gg_mm_data(is_train=True, is_gg_map=False)  # GG earth
        # dataset_train, dataset_valid, dataset_test = self._prepare_deep_nir_x100_dataset(is_train=True)  # deepNIR

        learning_rate = {
            "init_lr": 0.0003,
            "lr_policy": LrPolicy.ONE_CYCLE
            }

        print_param = {"result_path": os.path.join(os.getcwd(), "results", "alto_ggmap_ihn")
                       }

        train_param = {"gpu_idx": 0,
                       "max_epoch": 200,
                       "optimizer": OptimizerName.ADAMW,
                       "l2_regular": 0.00001,
                       "lr": learning_rate,
                       "batch_size": 16,
                       "print_param": print_param
                       }

        AltoIhn().do_train(dataset_train, dataset_valid, train_param)

    def test_alto(self):
        from image_matching.controller.alto import AltoDhn, AltoRaft, AltoIhn, AltoRhwf

        dataset_train, dataset_valid, dataset_test = self._prepare_gg_mm_data(is_train=False, is_gg_map=True)  # GG map
        # dataset_train, dataset_valid, dataset_test = self._prepare_gg_mm_data(is_train=False, is_gg_map=False)  # GG earth
        # dataset_train, dataset_valid, dataset_test = self._prepare_deep_nir_x100_dataset(is_train=False)  # deepNIR

        print_param = {"result_path": os.path.join(os.getcwd(), "results", "alto_ggmap_ihn")
                       }

        test_param = {"gpu_idx": 0,
                      "batch_size": 1,
                      "print_param": print_param
                      }

        AltoIhn().do_test(dataset_test, test_param)
