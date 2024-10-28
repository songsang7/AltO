import os

import numpy as np
import kornia
import torch
import torchvision

from common.utils import GeneralUtils, GeometryUtils


class ImageMatchingUtils:
    def __init__(self):
        pass

    # @staticmethod
    # def smoothness_loss(field, img=None, p_norm=1, alpha=0.0):
    #     """Calculate the smoothness loss of the given defromation field
    #
    #     :param field: the input deformation  (B, 2, H, W)
    #     :param img: the image that the deformation is applied on (will be used for the bilateral filtering).
    #     :param alpha: the alpha coefficient used in the bilateral filtering.
    #     :return:
    #     """
    #     if p_norm == 1:
    #         diff_1 = torch.abs(field[:, :, 1::, :] - field[:, :, 0:-1, :])
    #         diff_2 = torch.abs((field[:, :, :, 1::] - field[:, :, :, 0:-1]))
    #         diff_3 = torch.abs(field[:, :, 0:-1, 0:-1] - field[:, :, 1::, 1::])
    #         diff_4 = torch.abs(field[:, :, 0:-1, 1::] - field[:, :, 1::, 0:-1])
    #     elif p_norm == 2:
    #         diff_1 = torch.square(field[:, :, 1::, :] - field[:, :, 0:-1, :])
    #         diff_2 = torch.square((field[:, :, :, 1::] - field[:, :, :, 0:-1]))
    #         diff_3 = torch.square(field[:, :, 0:-1, 0:-1] - field[:, :, 1::, 1::])
    #         diff_4 = torch.square(field[:, :, 0:-1, 1::] - field[:, :, 1::, 0:-1])
    #     else:
    #         assert False
    #
    #     if img is not None and alpha > 0.0:  # Bilateral Filter
    #         mask = img
    #         weight_1 = torch.exp(-alpha * torch.abs(mask[:, :, 1::, :] - mask[:, :, 0:-1, :]))
    #         weight_1 = torch.mean(weight_1, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    #         weight_2 = torch.exp(- alpha * torch.abs(mask[:, :, :, 1::] - mask[:, :, :, 0:-1]))
    #         weight_2 = torch.mean(weight_2, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    #         weight_3 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 0:-1] - mask[:, :, 1::, 1::]))
    #         weight_3 = torch.mean(weight_3, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    #         weight_4 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 1::] - mask[:, :, 1::, 0:-1]))
    #         weight_4 = torch.mean(weight_4, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    #     else:
    #         weight_1 = weight_2 = weight_3 = weight_4 = 1.0
    #
    #     numer = (weight_1 * diff_1).mean(dim=(1, 2, 3)) +\
    #             (weight_2 * diff_2).mean(dim=(1, 2, 3)) +\
    #             (weight_3 * diff_3).mean(dim=(1, 2, 3)) +\
    #             (weight_4 * diff_4).mean(dim=(1, 2, 3))
    #
    #     denomi = weight_1 + weight_2 + weight_3 + weight_4
    #
    #     loss = numer / denomi
    #     return loss
    #
    # @staticmethod
    # def smoothness_loss2(field, p_norm, mask=None):
    #     class GradLoss(torch.nn.Module):
    #         def __init__(self, p_norm_):
    #             super().__init__()
    #             self._p_norm = p_norm_
    #
    #         def _grad2d(self, prediction):
    #             dy = torch.abs(prediction[:, :, 1:] - prediction[:, :, :-1])
    #             dx = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
    #
    #             if self._p_norm == 2:
    #                 dy = dy * dy
    #                 dx = dx * dx
    #             elif self._p_norm == 1:
    #                 dy = dy.abs()
    #                 dx = dx.abs()
    #             else:
    #                 assert False
    #
    #             d = torch.mean(dx, dim=(1, 2, 3)) + torch.mean(dy, dim=(1, 2, 3))
    #
    #             return d / 2.0
    #
    #         def forward(self, prediction, mask_=None):
    #             if mask_ is not None:
    #                 prediction = prediction * mask_
    #
    #             loss = self._grad2d(prediction)
    #             return loss
    #     return GradLoss(p_norm)(field, mask)

    @staticmethod
    def calc_smooth_loss(flow_field, p_norm, apply_root, mask=None, num_directions=2, img_bilateral=None, alpha_bilateral=1.0):
        """
        :param flow_field: (B, H, W, 2)
        :param p_norm: int 1 or 2
        :param apply_root: bool
        :param mask: (B, 1, H, W)
        :param num_directions: 2 or 4
        :param img_bilateral: (B, C, H, W)
        :param alpha_bilateral: float
        :return: (B)
        """
        eps = 0.00001

        flow_field = flow_field.permute(0, 3, 1, 2)  # (B, H, W, 2) --> (B, 2, H, W)

        # Apply Mask
        if mask is not None:
            flow_field = flow_field * mask

        if (img_bilateral is not None) and (mask is not None):
            img_bilateral = img_bilateral * mask

        # Prepare slices
        slice_dy1 = (slice(None), slice(None), slice(1, None), slice(None))
        slice_dy2 = (slice(None), slice(None), slice(None, -1), slice(None))
        slice_dx1 = (slice(None), slice(None), slice(None), slice(1, None))
        slice_dx2 = (slice(None), slice(None), slice(None), slice(None, -1))
        slices_1 = [slice_dy1, slice_dx1]
        slices_2 = [slice_dy2, slice_dx2]
        if num_directions == 4:
            slice_dyx1 = (slice(None), slice(None), slice(1, None), slice(1, None))
            slice_dyx2 = (slice(None), slice(None), slice(None, -1), slice(None, -1))
            slice_dxy1 = (slice(None), slice(None), slice(None, -1), slice(1, None))
            slice_dxy2 = (slice(None), slice(None), slice(1, None), slice(None, -1))
            slices_1.extend([slice_dyx1, slice_dxy1])
            slices_2.extend([slice_dyx2, slice_dxy2])

        # Calc
        numer = 10 * eps
        denomi = eps
        for slice_1, slice_2 in zip(slices_1, slices_2):
            difference = flow_field[slice_1] - flow_field[slice_2]

            weight = torch.ones_like(flow_field[slice_1])
            if img_bilateral is not None:
                weight = torch.exp(-alpha_bilateral * torch.abs(img_bilateral[slice_1] - img_bilateral[slice_2]))
                weight = weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)

            denomi += weight.sum(dim=(1, 2, 3))

            if p_norm == 1:
                numer += (difference.abs() * weight).sum(dim=(1, 2, 3))
            elif p_norm == 2:
                numer += (difference.square() * weight).sum(dim=(1, 2, 3))
            else:
                assert False

        result = numer / denomi
        if p_norm == 2 and apply_root:
            result = result.sqrt()

        return result


    @staticmethod
    def masked_diff_2d_loss(a, b, p, mask=None):
        """
        :param a: (B, C, H, W)
        :param b: (B, C, H, W)
        :param p: int
        :param mask: (B, 1, H, W)
        :return: (B)
        """
        eps = 0.00001

        if mask is None:
            mask = torch.ones_like(a)
        else:
            ch = a.shape[1]
            mask = mask.repeat(1, ch, 1, 1)

        if p == 1:
            numer = (mask * (a - b).abs()).sum(dim=(1, 2, 3)) + 10 * eps
        elif p == 2:
            numer = (mask * (a - b).square()).sum(dim=(1, 2, 3)) + 10 * eps
        else:
            numer = (mask * (a - b).pow(p)).sum(dim=(1, 2, 3)) + 10 * eps

        denom = mask.sum(dim=(1, 2, 3)) + eps
        result = numer / denom
        return result

    @staticmethod
    def calc_mace_0(corners_pred, corners_gt):
        """
        :param corners_pred: (B, 4, 2)
        :param corners_gt: (B, 4, 2)
        :return: float
        """
        result = torch.mean(torch.linalg.norm(corners_gt.reshape(-1, 2) - corners_pred.reshape(-1, 2), dim=-1))
        return result

    @staticmethod
    def calc_mace_1(corners_gt, corners_src, h_pred):
        result = torch.mean(kornia.geometry.oneway_transfer_error(corners_src, corners_gt, h_pred, squared=False))
        return result

    @staticmethod
    def calc_mace_2(corners_src, h_pred, h_gt):  # mace_1과 결과 동일하다.
        corners_gt = kornia.geometry.transform_points(h_gt, corners_src)
        corners_pred = kornia.geometry.transform_points(h_pred, corners_src)
        result = ImageMatchingUtils.calc_mace_0(corners_pred, corners_gt)
        return result

    @staticmethod
    def calc_mace_3(height, width, h_pred, h_gt):
        corners_src = GeometryUtils.gen_corners_torch(height, width).to(h_pred.device)
        batch_size = h_pred.shape[0]
        corners_src = corners_src.repeat(batch_size, 1, 1)
        return ImageMatchingUtils.calc_mace_2(corners_src, h_pred, h_gt)

    @staticmethod
    def print_inputs(image_patch_moving, image_patch_fixed, flow_field, print_dir):
        warped_moving_patch = GeometryUtils.get_batch_warpd_image_from_flow_field(image_patch_moving, flow_field)

        os.makedirs(print_dir, exist_ok=True)
        torchvision.utils.save_image(image_patch_moving, os.path.join(print_dir, "image_patch_moving.png"))
        torchvision.utils.save_image(image_patch_fixed, os.path.join(print_dir, "image_patch_fixed.png"))
        torchvision.utils.save_image(warped_moving_patch, os.path.join(print_dir, "warped_moving_patch.png"))

    @staticmethod
    def calc_batch_dsc(seg_mask_moving: torch.tensor, seg_mask_fixed: torch.tensor, seg_mask_warped: torch.tensor, num_labels: int):
        """
        :param seg_mask_moving: (B, 1, H, W)
        :param seg_mask_fixed: (B, 1, H, W)
        :param seg_mask_warped: (B, 1, H, W)
        :param num_labels: int (include background(0))
        :return: (B)
        """
        eps = 1e-8
        batch_size = seg_mask_moving.shape[0]
        seg_mask_moving = seg_mask_moving.view(batch_size, -1).to(torch.int32)
        seg_mask_fixed = seg_mask_fixed.view(batch_size, -1).to(torch.int32)
        seg_mask_warped = seg_mask_warped.view(batch_size, -1).to(torch.int32)

        dice = torch.zeros((batch_size, num_labels - 1), device=seg_mask_moving.device, dtype=torch.float32)
        for i in range(1, num_labels):  # 0 is background
            src = (seg_mask_moving == i) * 1
            gt = (seg_mask_fixed == i) * 1
            pred = (seg_mask_warped == i) * 1
            cond_exception = torch.logical_or((src.sum(dim=1) == 0), (gt.sum(dim=1) == 0))  # (B)

            gt = gt.float()
            pred = pred.float()
            numer = 2 * (pred * gt).sum(dim=1)
            denomi = (pred + gt).sum(dim=1)

            dice[:, i - 1] = torch.where(cond_exception, torch.nan, numer / denomi)

        result = dice.nanmean(dim=1)
        return result

    @staticmethod
    def calc_dsc(seg_mask_moving: np.ndarray, seg_mask_fixed: np.ndarray, seg_mask_warped: np.ndarray, num_labels: int):
        def compute_dice_coefficient(mask_gt, mask_pred):
            """Computes soerensen-dice coefficient.

            compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
            and the predicted mask `mask_pred`.

            Args:
              mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
              mask_pred: 3-dim Numpy array of type bool. The predicted mask.

            Returns:
              the dice coeffcient as float. If both masks are empty, the result is NaN.
            """
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return 0
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        dice = []
        for i in range(1, num_labels):
            if ((seg_mask_fixed == i).sum() == 0) or ((seg_mask_moving == i).sum() == 0):
                dice.append(np.NAN)
            else:
                dice.append(compute_dice_coefficient((seg_mask_fixed == i), (seg_mask_warped == i)))
        mean_dice = np.nanmean(dice)
        return mean_dice, dice

    @staticmethod
    def create_grid_image(cell_height, cell_width, image_height, image_width, background_gv=1.0, line_gv=0.0):
        """
        :param cell_height: int
        :param cell_width: int
        :param image_height: int
        :param image_width: int
        :param background_gv: [0, 1]
        :param line_gv: [0, 1]
        :return: (1, 1, image_height, image_width), tensor
        """
        image = torch.full((1, 1, image_height, image_width), background_gv)
        image[:, :, 0:image_height:cell_height, :] = line_gv
        image[:, :, :, 0:image_width:cell_width] = line_gv
        return image

    @staticmethod
    def calc_local_ncc(prediction, ground_truth, mask=None, kernel_type='mean', kernel_var=None):
        class LocalNCC(torch.nn.Module):
            def __init__(self, device, kernel_var=None, name=None, kernel_type='mean', eps=1e-5):
                if name is None:
                    name = 'ncc'
                super().__init__()
                self.device = device
                self.kernel_var = kernel_var
                self.kernel_type = kernel_type
                self.eps = eps

                assert kernel_type in ['mean', 'gaussian', 'linear']

            def _get_kernel(self, kernel_type, kernel_sigma):

                if kernel_type == 'mean':
                    kernel = torch.ones([1, 1, *kernel_sigma]).to(self.device)

                elif kernel_type == 'linear':
                    raise NotImplementedError("Linear kernel for NCC still not implemented")

                elif kernel_type == 'gaussian':
                    kernel_size = kernel_sigma[0] * 3
                    kernel_size += np.mod(kernel_size + 1, 2)

                    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
                    x_cord = torch.arange(kernel_size)
                    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
                    y_grid = x_grid.t()
                    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

                    mean = (kernel_size - 1) / 2.
                    variance = kernel_sigma[0] ** 2.

                    # Calculate the 2-dimensional gaussian kernel which is
                    # the product of two gaussian distributions for two different
                    # variables (in this case called x and y)
                    # 2.506628274631 = sqrt(2 * pi)

                    kernel = (1. / (2.506628274631 * kernel_sigma[0])) * \
                             torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

                    # Make sure sum of values in gaussian kernel equals 1.
                    # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

                    # Reshape to 2d depthwise convolutional weight
                    kernel = kernel.view(1, 1, kernel_size, kernel_size)
                    kernel = kernel.to(self.device)

                return kernel

            def _compute_local_sums(self, I, J, filt, stride, padding):

                ndims = len(list(I.size())) - 2

                I2 = I * I
                J2 = J * J
                IJ = I * J

                conv_fn = getattr(torch.nn.functional, 'conv%dd' % ndims)

                I_sum = conv_fn(I, filt, stride=stride, padding=padding)
                J_sum = conv_fn(J, filt, stride=stride, padding=padding)
                I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
                J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
                IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

                win_size = torch.sum(filt)
                u_I = I_sum / win_size
                u_J = J_sum / win_size

                cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
                I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
                J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

                return I_var, J_var, cross

            def ncc(self, prediction, target):
                """
                    calculate the normalize cross correlation between I and J
                    assumes I, J are sized [batch_size, nb_feats, *vol_shape]
                    """

                ndims = len(list(prediction.size())) - 2

                assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

                if self.kernel_var is None:
                    if self.kernel_type == 'gaussian':
                        kernel_var = [3] * ndims  # sigma=3, radius = 9
                    else:
                        kernel_var = [9] * ndims  # sigma=radius=9 for mean and linear filter

                else:
                    kernel_var = self.kernel_var

                sum_filt = self._get_kernel(self.kernel_type, kernel_var)
                radius = sum_filt.shape[-1]
                pad_no = int(np.floor(radius / 2))

                if ndims == 1:
                    stride = (1)
                    padding = (pad_no)
                elif ndims == 2:
                    stride = (1, 1)
                    padding = (pad_no, pad_no)
                else:
                    stride = (1, 1, 1)
                    padding = (pad_no, pad_no, pad_no)

                # Eugenio: bug fixed where cross was not squared when computing cc
                I_var, J_var, cross = self._compute_local_sums(prediction, target, sum_filt, stride, padding)
                cc = cross * cross / (I_var * J_var + self.eps)

                return cc

            def forward(self, prediction, target, mask=None):

                # if mask is not None:
                #     prediction = prediction * mask
                #     target = target * mask

                cc = self.ncc(prediction, target)
                if mask is None:
                    return -1.0 * torch.sqrt(torch.mean(cc))
                elif torch.sum(mask) == 0:
                    return torch.tensor(0)
                else:
                    norm_factor = 1 / (torch.sum(mask))
                    return -1.0 * torch.sqrt(norm_factor * torch.sum(cc * mask))

        criterion = LocalNCC(prediction.device, kernel_var, None, kernel_type)
        return criterion(prediction, ground_truth, mask)

    @staticmethod
    def calc_corr_pyramid(f1, f2, num_levels, do_relu=True):
        """
        :param f1: (B, C, H, W)
        :param f2: (B, C, H, W)
        :param num_levels: int
        :param do_relu: bool
        :return: List[(BHW, 1, H_i, W_i)] size of num_levels
        """
        b, c, h, w = f1.shape
        f1 = f1.view(b, c, h*w)  # (B, C, L), L=HW
        f2 = f2.view(b, c, h*w)  # (B, C, L)

        corr = (f1.transpose(1, 2) @ f2)  # (B, L, L)
        if do_relu:
            corr = torch.relu(corr)
        corr = corr.view(-1, 1, h, w)  # (BL, 1, H, W)

        pyramid = [corr]
        for i in range(num_levels - 1):
            corr = torch.nn.functional.avg_pool2d(corr, kernel_size=2, stride=2)
            pyramid.append(corr)

        return pyramid

    @staticmethod
    def get_corr_block(loc_field, corr_pyramid, radius):
        """
        :param loc_field: (B, H, W, 2)
        :param corr_pyramid: List[(BHW, 1, H_i, W_i)] size of num_levels
        :param radius: int
        :return: (B, fov*fov*num_levels, H, W)
        """
        b, h, w, _ = loc_field.shape
        loc_field = loc_field.view(b * h * w, 1, 1, 2)  # (B, H, W, 2) -> (BHW, 1, 1, 2)
        num_levels = len(corr_pyramid)

        fov = 2 * radius + 1
        fov_grid = radius * GeometryUtils.gen_2d_grid_torch(fov, fov, True).to(loc_field.device)  # (1, fov, fov, 2), [-r, +r]

        sampled_blocks = []
        for i in range(num_levels):
            corr_mat = corr_pyramid[i]  # (BHW, 1, H_i, W_i)
            h_i, w_i = corr_mat.shape[-2:]

            loc_field_i = loc_field / (2 ** i)
            fov_loc_field_i = fov_grid + loc_field_i  # (1, fov, fov, 2) + (BHW, 1, 1, 2) -> (BHW, fov, fov, 2) by broad-casting
            fov_loc_field_i[..., 0] = 2 * fov_loc_field_i[..., 0] / (w_i - 1) - 1
            fov_loc_field_i[..., 1] = 2 * fov_loc_field_i[..., 1] / (h_i - 1) - 1

            corr_block = torch.nn.functional.grid_sample(corr_mat, fov_loc_field_i, align_corners=True, mode='bilinear', padding_mode='zeros')  # (BHW, 1, fov, fov)
            corr_block = corr_block.view(b, h, w, -1)  # (B, H, W, fov*fov)
            sampled_blocks.append(corr_block)
        result = torch.cat(sampled_blocks, dim=-1).permute(0, 3, 1, 2).contiguous()  # (B, fov*fov*num_levels, H, W)
        return result
