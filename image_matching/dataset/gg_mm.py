import os
import json

import cv2
import numpy as np

from common.definitions import BasicDataKey
from common.utils import ImageUtils, GeometryUtils
from image_matching.definitions import DataKey, DataKey
from image_matching.dataset.labeled_pair import LabeledPair


class GoogleMultimodal(LabeledPair):
    def __init__(self, name, memo, paths):
        super().__init__(name, memo, paths, 128, 128, 3)

        self._patch_left_top = [192 // 6, 192 // 6]
        self._corners = np.array([[0, 0], [127, 0], [0, 127], [127, 127]], np.float32)

    def _check_tuple(self, item_moving, item_fixed, item_label) -> bool:
        return os.path.splitext(item_moving)[0] == os.path.splitext(item_fixed)[0] and os.path.splitext(item_fixed)[0] == os.path.splitext(item_label)[0][0:-6]

    def _make_single_data(self, idx, full_path_moving, full_path_fixed, full_path_label) -> dict:
        h, h_patch = self._parse_label(full_path_label)

        single_data = {BasicDataKey.ID: idx,
                       DataKey.IMAGE_PATH_MOVING: full_path_moving,
                       DataKey.IMAGE_PATH_FIXED: full_path_fixed,
                       DataKey.LABEL_PATH: full_path_label,
                       DataKey.LABEL_H: h,
                       DataKey.LABEL_H_PATCH: h_patch
                       }

        return single_data

    def _parse_label(self, label_path):
        with open(label_path, 'r') as outfile:
            data = json.load(outfile)
        x_list = [data['location'][0]['top_left_u'], data['location'][1]['top_right_u'], data['location'][2]['bottom_left_u'], data['location'][3]['bottom_right_u']]
        x_list = np.array(x_list, np.float32)
        y_list = [data['location'][0]['top_left_v'], data['location'][1]['top_right_v'], data['location'][2]['bottom_left_v'], data['location'][3]['bottom_right_v']]
        y_list = np.array(y_list, np.float32)
        dst_pts = np.stack([x_list, y_list], axis=-1)

        width_ori = 192
        height_ori = 192
        width_patch = 128
        height_patch = 128
        left_top = np.array([width_ori // 6, height_ori // 6], np.float32)
        src_pts = left_top + np.array([[0, 0], [width_patch - 1, 0], [0, height_patch - 1], [width_patch - 1, height_patch - 1]], np.float32)

        h = cv2.getPerspectiveTransform(src_pts, dst_pts)
        h_patch = cv2.getPerspectiveTransform(src_pts - left_top, dst_pts - left_top)

        return h.tolist(), h_patch.tolist()

    def _get_moving_image(self, path):
        flag = cv2.IMREAD_GRAYSCALE if self._ch == 1 else cv2.IMREAD_COLOR
        image_moving = ImageUtils.imread(path, flag)
        if flag == cv2.IMREAD_COLOR:
            image_moving = cv2.cvtColor(image_moving, cv2.COLOR_BGR2RGB)
        left, top = self._patch_left_top
        image_patch_moving = image_moving[left:left + 128, top:top + 128, ...]
        return image_moving, image_patch_moving

    def _get_fixed_image_patch(self, path):
        flag = cv2.IMREAD_GRAYSCALE if self._ch == 1 else cv2.IMREAD_COLOR
        image_patch_fixed = ImageUtils.imread(path, flag)
        if flag == cv2.IMREAD_COLOR:
            image_patch_fixed = cv2.cvtColor(image_patch_fixed, cv2.COLOR_BGR2RGB)
        return image_patch_fixed

    def __getitem__(self, index):
        data = self._data_list[index]

        image_moving, image_patch_moving = self._get_moving_image(data[DataKey.IMAGE_PATH_MOVING])
        image_patch_fixed = self._get_fixed_image_patch(data[DataKey.IMAGE_PATH_FIXED])

        rect_patch = self._corners + self._patch_left_top

        h = data[DataKey.LABEL_H]
        h_patch = data[DataKey.LABEL_H_PATCH]

        perturbs = GeometryUtils.get_pts_offset_from_homography(self._corners, h_patch)

        # h_test = GeometryUtils.get_homography_from_pts_offset(self._corners, perturbs)
        # b = h_test == h_patch

        result = {DataKey.IMAGE_MOVING: self._to_tensor(image_moving),
                  DataKey.IMAGE_PATCH_MOVING: self._to_tensor(image_patch_moving),
                  DataKey.IMAGE_PATCH_FIXED: self._to_tensor(image_patch_fixed),
                  DataKey.RECT_PATCH: rect_patch,
                  DataKey.H: np.array(h, np.float32),
                  DataKey.H_PATCH: np.array(h_patch, np.float32),
                  DataKey.PERTURBS: perturbs
                  }
        result.update(data)

        return result
