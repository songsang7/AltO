import os

from common.dataset.dataset_info import BasicImageDatasetInfo
from common.utils import GeneralUtils


class LabeledPair(BasicImageDatasetInfo):
    def __init__(self, name, memo, paths, width, height, ch):
        super().__init__(name, memo, paths, width, height, ch)

    def _check_tuple(self, item_moving, item_fixed, item_label) -> bool:
        pass

    def _make_single_data(self, idx, full_path_moving, full_path_fixed, full_path_label) -> dict:
        pass

    def _load_data_list(self, paths):
        dir_moving, dir_fixed, dir_label = paths

        item_list_moving = os.listdir(dir_moving)
        item_list_moving.sort(key=GeneralUtils.natural_keys)

        item_list_fixed = os.listdir(dir_fixed)
        item_list_fixed.sort(key=GeneralUtils.natural_keys)

        item_list_label = os.listdir(dir_label)
        item_list_label.sort(key=GeneralUtils.natural_keys)

        assert len(item_list_moving) == len(item_list_fixed) and len(item_list_fixed) == len(item_list_label)

        result = []
        i = 0  # enumerate()로 쓰면 continue 할때도 ++ 되어서 안맞다.
        for item_moving, item_fixed, item_label in zip(item_list_moving, item_list_fixed, item_list_label):
            assert self._check_tuple(item_moving, item_fixed, item_label)

            full_path_moving = os.path.join(dir_moving, item_moving)
            full_path_fixed = os.path.join(dir_fixed, item_fixed)
            full_path_label = os.path.join(dir_label, item_label)

            if os.path.isdir(full_path_moving) or os.path.isdir(full_path_fixed) or os.path.isdir(full_path_label):
                continue

            single_data = self._make_single_data(i, full_path_moving, full_path_fixed, full_path_label)
            result.append(single_data)
            i += 1

        return result
