import albumentations as alb
import numpy as np
import torchvision
import torchvision.transforms.functional as ttf

from common.utils import *
from image_matching.definitions import DataKey


class SpatialTransforms:
    def __init__(self, height_ori, width_ori, height_patch, width_patch):
        self._height_ori = height_ori
        self._width_ori = width_ori
        self._height_patch = height_patch
        self._width_patch = width_patch

        self._corners = GeometryUtils.gen_corners_numpy(height_patch, width_patch).astype(np.int32)

        self._to_tensor = torchvision.transforms.ToTensor()

    def generate_data(self, image_moving_ori, image_fixed_ori, mask_moving_ori, mask_fixed_ori, augmentations=alb.Compose([])) -> dict:
        pass


class FourPtsTransforms(SpatialTransforms):
    def __init__(self, height_ori, width_ori, height_patch, width_patch, max_perturb):
        super().__init__(height_ori, width_ori, height_patch, width_patch)
        self._max_perturb = max_perturb

    def _get_rect_patch(self):
        patch_left = np.random.randint(self._max_perturb, self._width_ori - self._width_patch - self._max_perturb + 1)
        patch_top = np.random.randint(self._max_perturb, self._height_ori - self._height_patch - self._max_perturb + 1)

        left_top = (patch_left, patch_top)
        rect_patch = self._corners + left_top
        return rect_patch

    def _generate_perturbation(self, rect_patch):
        while True:
            perturbs = np.random.randint(-self._max_perturb, self._max_perturb + 1, (8,))
            # perturbs = np.array([0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
            perturbs = perturbs.reshape((4, 2)).astype(np.float32)
            h = GeometryUtils.get_homography_from_pts_offset(rect_patch, perturbs)
            left_top = rect_patch[0, :]
            h_patch = GeometryUtils.get_homography_from_pts_offset(rect_patch - left_top, perturbs)

            if np.linalg.det(h.astype(np.float64)) != 0.0:
                break
        return perturbs, h, h_patch

    def generate_data(self, image_moving_ori, image_fixed_ori, mask_moving_ori, mask_fixed_ori, augmentations=alb.Compose([])):
        rect_patch = self._get_rect_patch()
        perturbs, h, h_patch = self._generate_perturbation(rect_patch)

        top = rect_patch[0][1]
        left = rect_patch[0][0]
        bot = rect_patch[3][1] + 1
        right = rect_patch[3][0] + 1

        image_moving = augmentations(image=image_moving_ori)["image"]
        image_patch_moving = image_moving[top:bot, left:right, ...]

        image_fixed_aug = augmentations(image=image_fixed_ori)["image"]
        image_fixed = GeometryUtils.get_warped_image_from_homography(image_fixed_aug, h)
        image_patch_fixed = image_fixed[top:bot, left:right, ...]

        result = {DataKey.IMAGE_MOVING: self._to_tensor(image_moving),
                  DataKey.IMAGE_FIXED: self._to_tensor(image_fixed),
                  DataKey.IMAGE_PATCH_MOVING: self._to_tensor(image_patch_moving),
                  DataKey.IMAGE_PATCH_FIXED: self._to_tensor(image_patch_fixed),
                  DataKey.H: h,
                  DataKey.H_PATCH: h_patch,
                  DataKey.RECT_PATCH: rect_patch,
                  DataKey.PERTURBS: perturbs
                  }
        return result


class ElasticTransforms(SpatialTransforms):
    def __init__(self, height_ori, width_ori, height_patch, width_patch, alpha=1.0, sigma=50.0):
        super().__init__(height_ori, width_ori, height_patch, width_patch)

        self._transform = alb.ElasticTransform(alpha, sigma, alpha_affine=0, always_apply=True)
        # self._transform = torchvision.transforms.ElasticTransform  # flow field도 같이 얻으려고 써봤지만, 너무너무 느리다.
        self._alpha = [alpha, alpha]
        self._sigma = [sigma, sigma]

    def _get_rect_patch(self):
        patch_left = np.random.randint(0, self._width_ori - self._width_patch + 1)
        patch_top = np.random.randint(0, self._height_ori - self._height_patch + 1)

        left_top = (patch_left, patch_top)
        rect_patch = self._corners + left_top
        return rect_patch

    def generate_data(self, image_moving_ori, image_fixed_ori, mask_moving_ori, mask_fixed_ori, augmentations=alb.Compose([])):
        rect_patch = self._get_rect_patch()

        top = rect_patch[0][1]
        left = rect_patch[0][0]
        bot = rect_patch[3][1] + 1
        right = rect_patch[3][0] + 1

        # Moving
        if mask_moving_ori is None:
            aug_moving = augmentations(image=image_moving_ori)
            image_moving = aug_moving["image"]
            image_patch_moving = image_moving[top:bot, left:right, ...]
            mask_moving = mask_patch_moving = None
        else:
            aug_moving = augmentations(image=image_moving_ori, mask=mask_moving_ori)
            image_moving = aug_moving["image"]
            image_patch_moving = image_moving[top:bot, left:right, ...]

            mask_moving = aug_moving["mask"]
            mask_patch_moving = mask_moving[top:bot, left:right, ...]

        # Fixed
        if mask_moving_ori is None:
            elastic_transed = self._transform(image=image_fixed_ori)
            image_fixed_ori = elastic_transed["image"]
            aug_fixed = augmentations(image=image_fixed_ori)
            image_fixed = aug_fixed["image"]
            image_patch_fixed = image_fixed[top:bot, left:right, ...]
            mask_fixed = mask_patch_fixed = None
        else:
            elastic_transed = self._transform(image=image_fixed_ori, mask=mask_fixed_ori)
            image_fixed_ori = elastic_transed["image"]
            mask_fixed_ori = elastic_transed["mask"]
            aug_fixed = augmentations(image=image_fixed_ori, mask=mask_fixed_ori)
            image_fixed = aug_fixed["image"]
            image_patch_fixed = image_fixed[top:bot, left:right, ...]

            mask_fixed = aug_fixed["mask"]
            mask_patch_fixed = mask_fixed[top:bot, left:right, ...]

        result = {DataKey.IMAGE_MOVING: self._to_tensor(image_moving),
                  DataKey.IMAGE_FIXED: self._to_tensor(image_fixed),
                  DataKey.IMAGE_PATCH_MOVING: self._to_tensor(image_patch_moving),
                  DataKey.IMAGE_PATCH_FIXED: self._to_tensor(image_patch_fixed)
                  }

        if mask_moving is not None:
            result.update({DataKey.LABEL_MASK_MOVING: self._to_tensor(mask_moving)})
            result.update({DataKey.LABEL_MASK_PATCH_MOVING: self._to_tensor(mask_patch_moving)})
        if mask_fixed is not None:
            result.update({DataKey.LABEL_MASK_FIXED: self._to_tensor(mask_fixed)})
            result.update({DataKey.LABEL_MASK_PATCH_FIXED: self._to_tensor(mask_patch_fixed)})

        return result
