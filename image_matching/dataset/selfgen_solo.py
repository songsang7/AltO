import albumentations as alb

from common.definitions import BasicDataKey
from image_matching.definitions import DataKey, DataKey
from image_matching.dataset.selfgen_aligned_pair import SelfGenAlignedPair
from image_matching.dataset.spatial_transforms import *


class SelfGenSolo(SelfGenAlignedPair):
    def __init__(self, name, memo, path, ch, width_ori, height_ori, width_patch, height_patch, is_elastic_trans, trans_param, augmentations=alb.Compose([])):
        super().__init__(name, memo, path, ch, width_ori, height_ori, width_patch, height_patch, is_elastic_trans, trans_param, augmentations)

    def _load_data_list(self, path):
        item_list = os.listdir(path)
        item_list.sort(key=GeneralUtils.natural_keys)

        result = []
        i = 0  # enumerate()로 쓰면 continue 할때도 ++ 되어서 안맞다.
        for item_name in item_list:
            full_path = os.path.join(path, item_name)

            if os.path.isdir(full_path):
                continue
            if not ImageUtils.is_image_file(full_path):
                continue

            single_data = {DataKey.IMAGE_PATH_MOVING: full_path,
                           BasicDataKey.ID: full_path
                           }
            result.append(single_data)
            i += 1

        return result

    def __getitem__(self, index):
        data = self._data_list[index]

        image_moving_ori = self._load_image_resized(data[DataKey.IMAGE_PATH_MOVING])
        mask_moving_ori = self._get_original_mask(data[DataKey.LABEL_PATH]) if DataKey.LABEL_PATH in data else None

        result = self._data_generator.generate_data(image_moving_ori, image_moving_ori, mask_moving_ori, mask_moving_ori, self._augmentations)
        result.update(data)
        return result
