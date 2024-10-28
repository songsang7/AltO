import albumentations as alb

from common.definitions import BasicDataKey
from common.utils import *
from common.dataset.dataset_info import BasicImageDatasetInfo
from image_matching.definitions import DataKey, DataKey
from image_matching.dataset.spatial_transforms import *


class SelfGenAlignedPair(BasicImageDatasetInfo):
    def __init__(self, name, memo, path, ch, width_ori, height_ori, width_patch, height_patch, is_elastic_trans, trans_param, augmentations=alb.Compose([])):
        super().__init__(name, memo, path, width_patch, height_patch, ch)

        self._source_width = width_ori
        self._source_height = height_ori
        self._is_elastic_trans = is_elastic_trans
        self._trans_param = trans_param
        self._corners = GeometryUtils.gen_corners_numpy(self._height, self._width).astype(np.int32)
        self._augmentations = augmentations

        if is_elastic_trans:
            self._data_generator = ElasticTransforms(height_ori, width_ori, height_patch, width_patch, trans_param["alpha"], trans_param["sigma"])
        else:
            self._data_generator = FourPtsTransforms(height_ori, width_ori, height_patch, width_patch, trans_param["max_perturb"])

    def _load_data_list(self, paths):
        dir_moving, dir_fixed = paths

        item_list_moving = os.listdir(dir_moving)
        item_list_moving.sort(key=GeneralUtils.natural_keys)

        item_list_fixed = os.listdir(dir_fixed)
        item_list_fixed.sort(key=GeneralUtils.natural_keys)

        assert len(item_list_moving) == len(item_list_fixed)

        result = []
        i = 0  # enumerate()로 쓰면 continue 할때도 ++ 되어서 안맞다.
        for item_moving, item_fixed in zip(item_list_moving, item_list_fixed):
            assert self._check_tuple(item_moving, item_fixed)

            full_path_moving = os.path.join(dir_moving, item_moving)
            full_path_fixed = os.path.join(dir_fixed, item_fixed)

            if os.path.isdir(full_path_moving) or os.path.isdir(full_path_fixed):
                continue

            single_data = self._make_single_data(i, full_path_moving, full_path_fixed)
            result.append(single_data)
            i += 1

        return result

    def _check_tuple(self, item_moving, item_fixed) -> bool:
        return os.path.splitext(item_moving)[0] == os.path.splitext(item_fixed)[0]

    def _make_single_data(self, idx, full_path_moving, full_path_fixed) -> dict:
        single_data = {BasicDataKey.ID: idx,
                       DataKey.IMAGE_PATH_MOVING: full_path_moving,
                       DataKey.IMAGE_PATH_FIXED: full_path_fixed,
                       }

        return single_data

    def _load_image_resized(self, path):  # resize 할 width, height가 부모 class와 다르다.
        flag = cv2.IMREAD_GRAYSCALE if self._ch == 1 else cv2.IMREAD_COLOR

        image_ori = ImageUtils.imread(path, flag)
        image_ori = cv2.resize(image_ori, dsize=(self._source_width, self._source_height))

        if flag == cv2.IMREAD_COLOR:
            image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)

        return image_ori

    def __getitem__(self, index):
        data = self._data_list[index]

        image_moving_ori = self._load_image_resized(data[DataKey.IMAGE_PATH_MOVING])
        image_fixed_ori = self._load_image_resized(data[DataKey.IMAGE_PATH_FIXED])

        mask_moving_ori = self._get_original_mask(data[DataKey.LABEL_PATH]) if DataKey.LABEL_PATH in data else None  # 이미지는 pair지만 정답 seg-mask는 1개다.

        result = self._data_generator.generate_data(image_moving_ori, image_fixed_ori, mask_moving_ori, mask_moving_ori, self._augmentations)
        result.update(data)
        return result
