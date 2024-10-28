import logging
import os
import random
import re
import time

import cv2
import numpy as np
import torch
import kornia


class GeneralUtils:
    def __init__(self):
        pass

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def get_time_stamp(time_to_convert=None):
        if time_to_convert is None:
            time_to_convert = time.time()
        time_stamp = time.localtime(time_to_convert)
        time_stamp = f"{time_stamp.tm_year}-{time_stamp.tm_mon:02d}-{time_stamp.tm_mday:02d} {time_stamp.tm_hour:02d}:{time_stamp.tm_min:02d}:{time_stamp.tm_sec:02d}"
        return time_stamp

    @staticmethod
    def get_batch_categorical_accuracy(pred: torch.Tensor, target: torch.Tensor):
        """
        :param pred: probs of each cat. (B, Num_Classes)
        :param target: GT. (B, Num_Classes)
        :return: Acc
        """
        top_pred = pred.argmax(dim=1)
        top_gt = target.argmax(dim=1)
        incorrect = (top_pred - top_gt).count_nonzero()
        acc = 1.0 - (incorrect.float() / target.shape[0])
        return acc

    @staticmethod
    def standardize(x, dim, eps=0.00001):
        std, mean = torch.std_mean(x, dim=dim, keepdim=True)
        return (x - mean) / (eps + std)

    @staticmethod
    def calc_batch_ncc_matrix(a, b):
        """
        shape of a and b = (B, d1, d2)
        return (B, d2, d2) ncc matrix
        """
        d1 = a.shape[1]

        z_a = GeneralUtils.standardize(a, dim=1)
        z_b = GeneralUtils.standardize(b, dim=1)

        ncc_mat = torch.bmm(z_a.transpose(1, 2), z_b) / (d1 - 1)  # (B, d2, d2)
        return ncc_mat

    @staticmethod
    def off_diagonal(x):  # (B, L, L)
        batch_size = x.shape[0]
        l = x.shape[1]
        x = x.view(batch_size, -1)
        x = x[:, :-1].view(batch_size, l - 1, l + 1)  # 각각의 행이 [diag, ...]와 같은 형태로 된다.
        return x[:, :, 1:].reshape(batch_size, -1)  # (B, LL - L)

    @staticmethod
    def sampling_patch_nce_loss(prj_feats_a, prj_feats_b, max_samples_cnt=-1, mask_ori=None, normalize_lv=2, temperature=0.1):
        """
        :param prj_feats_a: list[(B, C_i, H_i, W_i)]
        :param prj_feats_b: list[(B, C_i, H_i, W_i)]
        :param max_samples_cnt: int (negatives mean all samples)
        :param mask_ori: (B, 1, H, W) or None (None means ones_like())
        :param normalize_lv: int, 0: no norm(=inner-product), 1: cosine-sim, 2: ncc
        :param temperature: float, default: 1.0
        :param swap_dim: bool. If True, (B, C, C) else (B, L, L)
        :return: (B)
        """
        if mask_ori is not None:
            mask_ori = mask_ori.type(torch.float32)

        result = 0
        for prj_feat_a, prj_feat_b in zip(prj_feats_a, prj_feats_b):
            # gen_mask
            b, c, h, w = prj_feat_a.shape
            if mask_ori is None:
                mask = torch.ones((b, 1, h, w), device=prj_feat_a.device)
            else:
                mask = torch.nn.functional.interpolate(mask_ori, size=(h, w), mode="nearest")

            # sample idx
            min_mask_area = int(mask.sum(dim=(1, 2, 3)).min())
            if min_mask_area <= 0:
                min_mask_area = h * w
                mask += 1.0

            num_samples = int(min(max_samples_cnt, min_mask_area)) if max_samples_cnt > 0 else min_mask_area
            sampled_idx = torch.multinomial(mask.view(b, -1), num_samples)  # (B, num_samples)
            sampled_idx = sampled_idx.unsqueeze(2).repeat(1, 1, c)

            # pick (sampling)
            prj_feat_a = prj_feat_a.view(b, c, -1).permute(0, 2, 1)
            prj_feat_a = torch.gather(prj_feat_a, 1, sampled_idx)

            prj_feat_b = prj_feat_b.view(b, c, -1).permute(0, 2, 1)
            prj_feat_b = torch.gather(prj_feat_b, 1, sampled_idx)  # (B, N, C)

            result += GeneralUtils.infonce_loss_sub(prj_feat_a, prj_feat_b, temperature, normalize_lv)
        return result / len(prj_feats_a)

    @staticmethod
    def infonce_loss_sub(z1, z2, temperature, normalize_lv):
        """
        :param z1: (B, N, C)
        :param z2: (B, N, C)
        :param temperature: float
        :param normalize_lv: int, 0: no norm(=inner-product), 1: cosine-sim, 2: ncc
        :return: torch.float
        """
        eps = 0.00001

        b, n, c = z1.shape

        # (B, N, N)
        if normalize_lv == 0:  # Corr
            sim_mat = z1 @ z2.transpose(1, 2)
        elif normalize_lv == 1:  # Normed-Corr (= Cosine Sim)
            z1_fnorm = z1.norm(dim=2, keepdim=True, p="fro")
            z2_fnorm = z2.norm(dim=2, keepdim=True, p="fro")
            sim_mat = (z1 @ z2.transpose(1, 2)) / (z1_fnorm * z2_fnorm)
        else:  # Normed Cross Coefficient
            sim_mat = GeneralUtils.calc_batch_ncc_matrix(z1.transpose(1, 2), z2.transpose(1, 2))

        softmax_mat = torch.softmax(sim_mat / temperature, dim=2)
        on_diag = softmax_mat.diagonal(dim1=1, dim2=2)  # (B, N)
        nll_loss = (-torch.log(eps + on_diag)).mean(dim=1)  # (B)

        return nll_loss

    @staticmethod
    def infonce_loss_2d(z1, z2, temperature=0.1, normalize_lv=2):
        """
        The other name is 'Patch NCE'
        :param z1: (B, C, H, W)
        :param z2: (B, C, H, W)
        :param temperature: float
        :param normalize_lv: int, 0: no norm(=inner-product), 1: cosine-sim, 2: ncc
        :return: torch.float
        """
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1).transpose(1, 2)  # (B, L, C)
        z2 = z2.view(b, c, -1).transpose(1, 2)  # (B, L, C)

        return GeneralUtils.infonce_loss_sub(z1, z2, temperature, normalize_lv).mean()

    @staticmethod
    def infonce_loss_1d(z1, z2, temperature=0.1, normalize_lv=2):
        """
        :param z1: (B, C)
        :param z2: (B, C)
        :param temperature: float
        :param normalize_lv: int, 0: no norm(=inner-product), 1: cosine-sim, 2: ncc
        :return: torch.float
        """
        z1 = z1.unsqueeze(0)  # (1, B, C)
        z2 = z2.unsqueeze(0)  # (1, B, C)

        return GeneralUtils.infonce_loss_sub(z1, z2, temperature, normalize_lv).mean()

    @staticmethod
    def bt_loss_sub(z1, z2, alpha):
        """
        Args:
            z1: (B, N, C)
            z2: (B, N, C)
            alpha: float
        Returns: torch.float

        """
        ncc_mat = GeneralUtils.calc_batch_ncc_matrix(z1, z2)  # (B, C, C)

        on_diag = ncc_mat.diagonal(dim1=1, dim2=2)  # (B, C)
        off_diag = GeneralUtils.off_diagonal(ncc_mat)  # (B, CC-C)

        loss = (on_diag - 1).square().sum(dim=1) + alpha * off_diag.square().sum(dim=1)  # (B,)
        return loss.mean()

    @staticmethod
    def bt_loss_1d(z1, z2, alpha=0.005):
        """
        Args:
            z1: (B, C)
            z2: (B, C)
            alpha: float
        Returns: tensor.float
        """
        z1 = z1.unsqueeze(0)  # (1, B, C)
        z2 = z2.unsqueeze(0)  # (1, B, C)

        return GeneralUtils.bt_loss_sub(z1, z2, alpha)

    @staticmethod
    def bt_loss_2d(z1, z2, alpha=0.005):
        """
        Args:
            z1: (B, C, H, W)
            z2: (B, C, H, W)
            alpha: float
        Returns: tensor.float
        """
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1).transpose(1, 2)  # (B, L, C)
        z2 = z2.view(b, c, -1).transpose(1, 2)  # (B, L, C)

        return GeneralUtils.bt_loss_sub(z1, z2, alpha)

    @staticmethod
    def vic_reg_loss_1d(z1, z2, lamb=25, mu=25, nu=1):
        """
        Args:
            z1: (B, C)
            z2: (B, C)
            lamb: float
            mu: float
            nu: float
        Returns: torch.float
        """
        z1 = z1.unsqueeze(0)  # (1, B, C)
        z2 = z2.unsqueeze(0)  # (1, B, C)

        return GeneralUtils.vic_reg_loss_sub(z1, z2, lamb, mu, nu)

    @staticmethod
    def vic_reg_loss_2d(z1, z2, lamb=25, mu=25, nu=1):
        """
        Args:
            z1: (B, C, H, W)
            z2: (B, C, H, W)
            lamb: float
            mu: float
            nu: float
        Returns: (B,) -> torch.float
        """
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1).transpose(1, 2)  # (B, L, C)
        z2 = z2.view(b, c, -1).transpose(1, 2)  # (B, L, C)

        return GeneralUtils.vic_reg_loss_sub(z1, z2, lamb, mu, nu)

    @staticmethod
    def vic_reg_loss_sub(z1, z2, lamb, mu, nu):
        """
        Args:
            z1: (B, N, C)
            z2: (B, N, C)
            lamb: float
            mu: float
            nu: float
        Returns: (B,) -> torch.float
        """
        b, n, c = z1.shape
        std1, mean1 = torch.std_mean(z1, dim=1, keepdim=True)
        z1_bar = z1 - mean1

        std2, mean2 = torch.std_mean(z2, dim=1, keepdim=True)
        z2_bar = z2 - mean2

        # Variance loss
        loss_var = torch.relu(1 - std1).mean(dim=(1, 2)) + torch.relu(1 - std2).mean(dim=(1, 2))  # (B,)

        # Invariance loss
        loss_inv = (z1 - z2).square().mean(dim=(1, 2))  # (B,)

        # Covariance loss
        cov_mat_1 = (z1_bar.transpose(1, 2) @ z1_bar) / (n - 1)  # (B, C, C)
        off_diag_1 = GeneralUtils.off_diagonal(cov_mat_1)  # (B, CC - C)

        cov_mat_2 = (z2_bar.transpose(1, 2) @ z2_bar) / (n - 1)
        off_diag_2 = GeneralUtils.off_diagonal(cov_mat_2)  # (B, CC - C)

        loss_cov = (off_diag_1.square().sum(dim=1) + off_diag_2.square().sum(dim=1)) / c  # (B,)

        total_loss = lamb * loss_var + mu * loss_inv + nu * loss_cov

        return total_loss.mean()  # mean over batch-axis

    @staticmethod
    def sim_siam_loss(f1, f2, p1, p2):
        """
        Args:
            f1: (B, C)
            f2: (B, C)
            p1: (B, C)
            p2: (B, C)
        Returns: torch.float
        """
        loss_12 = -torch.cosine_similarity(p1, f2.detach(), dim=1).mean()
        loss_21 = -torch.cosine_similarity(p2, f1.detach(), dim=1).mean()
        total_loss = 0.5 * (loss_12 + loss_21)
        return total_loss

    @staticmethod
    def natural_keys(text):  # 문자열을 숫자와 비숫자 부분으로 나누는 함수
        '''
        alist.sort(key=natural_keys) 를 사용하면,
        리스트가 '사람이 읽는' 순서대로 정렬됩니다.
        '''

        def atoi(t):  # 숫자 부분을 추출하기 위한 함수
            return int(t) if t.isdigit() else t

        return [atoi(c) for c in re.split(r'(\d+)', text)]

    @staticmethod
    def normalize_tensor(input_tensor, new_min=0.0, new_max=1.0):
        old_min = input_tensor.min()
        old_max = input_tensor.max()

        result = new_max + ((new_max - new_min) / (old_max - old_min)) * (input_tensor - old_max)
        return result

    @staticmethod
    def gradient_penalty(critics, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = critics(interpolates)
        fake = torch.ones(d_interpolates.size(), device=real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    @staticmethod
    def set_net_requires_grad(net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad

    @staticmethod
    def init_weights(m):  # he_normal only
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


class ImageUtils:
    def __init__(self):
        pass

    @staticmethod
    def imread(filename, flags=cv2.IMREAD_UNCHANGED, dtype=np.uint8):
        try:
            n = np.fromfile(filename, dtype)
            img = cv2.imdecode(n, flags)
            return img
        except Exception as e:
            logging.warning(e)
            return None

    @staticmethod
    def imwrite(filename, img, params=None):
        try:
            ext = os.path.splitext(filename)[1]
            result, n = cv2.imencode(ext, img, params)
            if result:
                with open(filename, mode='w+b') as f:
                    n.tofile(f)
                    return True
            else:
                return False
        except Exception as e:
            logging.warning(e)
            return False

    @staticmethod
    def is_image_file(path):
        path_lowercase = path.lower()

        if path_lowercase[-4:] == ".png":
            return True
        if path_lowercase[-4:] == ".jpg":
            return True
        if path_lowercase[-5:] == ".jpeg":
            return True
        if path_lowercase[-4:] == ".tif":
            return True
        if path_lowercase[-5:] == ".tiff":
            return True
        if path_lowercase[-4:] == ".bmp":
            return True

        return False

    @staticmethod
    def draw_mask_numpy(image, mask_generated, mask_color=(0, 1, 0), alpha=0.5):
        """
        :param image: ndarray
        :param mask_generated: ndarray
        :param mask_color:
        :param alpha:
        :return:
        """
        image_ori = image.copy()
        masked_image = image.copy()
        if len(image.shape) == 2:
            image_ori = np.expand_dims(image_ori, axis=-1).repeat(3, axis=-1)
            masked_image = np.expand_dims(masked_image, axis=-1)

        if len(mask_generated.shape) == 2:
            mask_generated = np.expand_dims(mask_generated, axis=-1)

        condition = 0 < mask_generated
        masked_image = np.where(condition, np.array(mask_color, dtype=masked_image.dtype), masked_image)
        # masked_image = masked_image.astype(np.uint8)

        return cv2.addWeighted(image_ori, alpha, masked_image, 1-alpha, 0)


class GeometryUtils:
    def __init__(self):
        pass

    @staticmethod
    def gen_corners_numpy(height: int, width: int):
        """
        :param height: int
        :param width: int
        :return: (4, 2) np.float32
        """
        corners = [[0, 0],  # width, height 순서. (xy-indexing)
                   [width - 1, 0],
                   [0, height - 1],
                   [width - 1, height - 1]
                   ]
        return np.array(corners, dtype=np.float32)

    @staticmethod
    def gen_corners_torch(height: int, width: int):
        """
        :param height: int
        :param width: int
        :return: (1, 4, 2) torch.float32
        """
        corners = [[0, 0],  # width, height 순서. (xy-indexing)
                   [width - 1, 0],
                   [0, height - 1],
                   [width - 1, height - 1]
                   ]
        return torch.tensor(corners, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def normalize_mesh_grid(grid, min_val, max_val):
        """
        :param grid: torch.tensor of shape (B, H, W, 2) or (H, W, 2)
        :param min_val: lower bound of result (float)
        :param max_val: upper bound of result (float)
        :return: same shape with grid
        """
        width, height = grid.shape[-2], grid.shape[-3]
        grid[..., 0] = min_val + (max_val - min_val) * grid[..., 0] / (width - 1)
        grid[..., 1] = min_val + (max_val - min_val) * grid[..., 1] / (height - 1)

        return grid

    @staticmethod
    def gen_2d_grid_numpy(height: int, width: int, normalized=False):
        """
        :param height: int
        :param width: int
        :param normalized: [-1, +1] normalize
        :return: (H, W, 2)
        """
        # create sampling grid
        vectors = [np.arange(0, s) for s in (width, height)]  # 순서는 meshgrid의 indexing에 따라 바뀐다.
        grids = np.meshgrid(vectors[0], vectors[1], indexing="xy")
        grids = np.stack(grids, axis=-1).astype(np.float32)

        # normalize
        if normalized:
            grids[..., 0] = -1 + 2 * grids[..., 0] / (width - 1)
            grids[..., 1] = -1 + 2 * grids[..., 1] / (height - 1)

        return grids

    @staticmethod
    def gen_2d_grid_torch(height: int, width: int, normalized=False):
        """
        :param height: int
        :param width: int
        :param normalized: [-1, +1] normalize
        :return: (1, H, W, 2) float-type
        """
        # create sampling grid
        vectors = [torch.arange(0, s) for s in (width, height)]
        grids = torch.meshgrid(vectors, indexing="xy")
        grids = torch.stack(grids, dim=-1).unsqueeze(0).type(torch.FloatTensor)

        # normalize
        if normalized:
            grids = GeometryUtils.normalize_mesh_grid(grids, -1, +1)

        return grids

    @staticmethod
    def get_batch_roi_from_image(image, left_top, height_roi, width_roi, grid=None):
        """
        :param image: (B, C, H, W)
        :param left_top: (B, 2)
        :param height_roi: int
        :param width_roi: int
        :param grid: 2d mesh grid. "xy" indexed (1, H, W, 2)
        :return:
        """
        batch_size = image.shape[0]
        height_ori = image.shape[2]
        width_ori = image.shape[3]

        if grid is None:
            grid = GeometryUtils.gen_2d_grid_torch(height_roi, width_roi).to(left_top.device)

        left_top = left_top.unsqueeze(1).unsqueeze(1)
        left_top = left_top.repeat(1, height_roi, width_roi, 1)
        grid_t = grid.repeat(batch_size, 1, 1, 1) + left_top

        # normalize
        grid_t = GeometryUtils.normalize_mesh_grid(grid_t, -1, +1)

        warped_patch_from_ori = torch.nn.functional.grid_sample(image, grid_t, align_corners=True, mode='bilinear', padding_mode='zeros')
        return warped_patch_from_ori

    @staticmethod
    def get_warped_image_from_homography(image, h):
        height, width = image.shape[0], image.shape[1]
        h_inv = np.linalg.inv(h)
        return cv2.warpPerspective(image, h_inv, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    @staticmethod
    def get_pts_offset_from_homography(src_pts, h):
        """
        :param src_pts: (N, 2), 'xy' indexed
        :param h: (3, 3)
        :return: (N, 2), np.float32
        """
        h = torch.Tensor(h).unsqueeze(0)
        src_pts = torch.Tensor(src_pts).unsqueeze(0)

        dst_pts = kornia.geometry.homography.transform_points(h, src_pts)
        result = dst_pts - src_pts
        return result.squeeze(0).numpy()

    @staticmethod
    def get_corners_offset_from_homography(height, width, h):
        """
        :param height: int
        :param width: int
        :param h: (3, 3)
        :return: (4, 2), np.float32
        """
        corners = GeometryUtils.gen_corners_numpy(height, width)
        return GeometryUtils.get_pts_offset_from_homography(corners, h)

    @staticmethod
    def get_flow_field_from_homography_sub(grid, h):
        """
        :param grid: (H, W, 2)
        :param h: (3, 3) float-type
        :return: (H, W, 2) np.float32
        """

        height, width = grid.shape[0], grid.shape[1]

        h = torch.Tensor(h).unsqueeze(0)
        grid = torch.Tensor(grid).unsqueeze(0)

        src_pts = grid.view(1, -1, 2)
        offsets = GeometryUtils.get_pts_offset_from_homography(src_pts, h)
        flow_field = offsets.reshape(height, width, 2)
        return flow_field

    @staticmethod
    def get_flow_field_from_homography(height, width, h):
        """
        :param height: int
        :param width: int
        :param h: (3, 3)
        :return: (H, W, 2) np.float32
        """

        grid = GeometryUtils.gen_2d_grid_numpy(height, width)
        return GeometryUtils.get_flow_field_from_homography_sub(grid, h)

    @staticmethod
    def get_homography_from_corners_offset(height, width, offsets, global_offset=None):
        """
        :param height: int
        :param width: int
        :param offsets: (4, 2)
        :param global_offset: (2) or None
        :return: (3, 3)
        """
        corners_src = GeometryUtils.gen_corners_numpy(height, width)
        return GeometryUtils.get_homography_from_pts_offset(corners_src, offsets, global_offset)

    @staticmethod
    def get_homography_from_pts_offset(src_pts, offsets, global_offset=None):
        """
        :param src_pts: (N, 2)
        :param offsets: (N, 2)
        :param global_offset: (2) or None
        :return: (3, 3)
        """
        if global_offset is not None:
            src_pts = src_pts + np.expand_dims(global_offset, axis=0)

        dst_pts = src_pts + offsets
        h = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
        return h.astype(np.float32)

    @staticmethod
    def get_homography_from_flow_field_sub(grid, flow_field, global_offset=None, method=cv2.USAC_MAGSAC):
        """
        :param grid: (H, W, 2), 'xy' indexed
        :param flow_field: same with 'grid'
        :param global_offset: (2) or None
        :param method: 0=(Least squares method), cv2.USAC_MAGSAC=(type of RANSAC)
        :return: (3, 3)
        """
        src_grid = np.array(grid, np.float32)

        if global_offset is not None:
            src_grid = src_grid + np.expand_dims(global_offset, axis=0)

        dst_grid = grid + flow_field
        h = cv2.findHomography(src_grid, dst_grid, method)[0]
        return h.astype(np.float32)

    @staticmethod
    def get_homography_from_flow_field(flow_field, global_offset=None, method=cv2.USAC_MAGSAC):
        """
        :param flow_field: (H, W, 2), 'xy' indexed
        :param global_offset: (2) or None
        :param method: 0=(Least squares method), cv2.USAC_MAGSAC=(type of RANSAC)
        :return: (3, 3)
        """
        height, width = flow_field.shape[0], flow_field.shape[1]
        grid = GeometryUtils.gen_2d_grid_numpy(height, width)
        return GeometryUtils.get_homography_from_flow_field_sub(grid, flow_field, global_offset, method)

    @staticmethod
    def get_batch_warpd_image_from_flow_field(image, flow, grid=None, interpolation="bilinear", padding_mode="zeros"):
        """
        Backward-warping
        :param image: (B, C, H, W)
        :param flow: (B, H, W, 2)
        :param grid: (1, H, W, 2) or None
        :param interpolation: "bilinear" or "nearest"
        :return: (B, C, H, W)
        """

        if grid is None:
            height, width = flow.shape[-3], flow.shape[-2]
            grid = GeometryUtils.gen_2d_grid_torch(height, width, False).to(flow.device)

        batch_size = flow.shape[0]
        grid = grid.repeat(batch_size, 1, 1, 1)
        new_locs = grid + flow
        new_locs = GeometryUtils.normalize_mesh_grid(new_locs, -1, +1)

        return torch.nn.functional.grid_sample(image, new_locs, align_corners=True, mode=interpolation, padding_mode=padding_mode)

    @staticmethod
    def get_batch_warped_image_from_homography(image, h):
        """
        Backward-warping
        :param image: (B, C, H, W)
        :param h: (B, 3, 3)
        :return: (B, C, H, W)
        """
        height, width = image.shape[-2], image.shape[-1]
        return kornia.geometry.warp_perspective(image, h.inverse(), (height, width))

    @staticmethod
    def get_batch_warped_mask_from_homography(height, width, h):
        """
        :param height: int
        :param width: int
        :param h: (B, 3, 3)
        :return: (B, 1, H, W)
        """
        batch_size = h.shape[0]
        mask = torch.ones(size=(batch_size, 1, height, width)).to(h.device)
        warped_mask = GeometryUtils.get_batch_warped_image_from_homography(mask, h)
        return warped_mask

    @staticmethod
    def get_batch_warped_mask_from_flow_field(flow_field):
        """
        :param flow_field: (B, H, W, 2)
        :return: (B, 1, H, W)
        """
        batch_size, height, width = flow_field.shape[0], flow_field.shape[1], flow_field.shape[2]
        mask = torch.ones(size=(batch_size, 1, height, width)).to(flow_field.device)
        warped_mask = GeometryUtils.get_batch_warpd_image_from_flow_field(mask, flow_field)
        return warped_mask

    @staticmethod
    def get_batch_pts_offset_from_homography(src_pts, h):
        """
        :param src_pts: (B, N, 2) or (1, N, 2)
        :param h: (B, 3, 3)
        :return: (B, N, 2), torch.float32
        """

        batch_size = h.shape[0]
        if src_pts.shape[0] < batch_size:
            q = batch_size // src_pts.shape[0]
            src_pts = src_pts.repeat(q, 1, 1)

        dst_pts = kornia.geometry.homography.transform_points(h, src_pts)
        result = dst_pts - src_pts
        return result

    @staticmethod
    def get_batch_corners_offset_from_homography(height, width, h):
        corners = GeometryUtils.gen_corners_torch(height, width).to(h.device)
        return GeometryUtils.get_batch_pts_offset_from_homography(corners, h)

    @staticmethod
    def get_batch_flow_field_from_homography_sub(grid, h):
        """
        :param grid: (B, H, W, 2) or (1, H, W, 2)
        :param h: (B, 3, 3)
        :return: (B, H, W, 2), torch.float32
        """

        batch_size = h.shape[0]
        height, width = grid.shape[1], grid.shape[2]
        src_pts = grid.view(grid.shape[0], -1, 2)
        offsets = GeometryUtils.get_batch_pts_offset_from_homography(src_pts, h)
        return offsets.view(batch_size, height, width, 2)

    @staticmethod
    def get_batch_flow_field_from_homography(height, width, h):
        """
        :param height: int
        :param width: int
        :param h: (B, 3, 3)
        :return: (B, H, W, 2)
        """
        grid = GeometryUtils.gen_2d_grid_torch(height, width).to(h.device)
        return GeometryUtils.get_batch_flow_field_from_homography_sub(grid, h)

    @staticmethod
    def get_batch_homography_from_pts_offsets(src_pts, offsets, global_offset=None):
        """
        :param src_pts: (B, N, 2) or (1, N, 2)
        :param offsets: (B, N, 2)
        :param global_offset: (B, 2) or None
        :return: (B, 3, 3)
        """
        batch_size = offsets.shape[0]
        if src_pts.shape[0] < batch_size:
            q = batch_size // src_pts.shape[0]
            src_pts = src_pts.repeat(q, 1, 1)

        if global_offset is not None:
            n = src_pts.shape[1]
            src_pts = src_pts + global_offset.unsqueeze(1).repeat(1, n, 1)

        dst_pts = src_pts + offsets

        h = kornia.geometry.get_perspective_transform(src_pts, dst_pts)
        return h

    @staticmethod
    def get_batch_homography_from_corners_offset(height: int, width: int, offsets, global_offset=None):
        """
        :param height: int
        :param width: int
        :param offsets: (B, 4, 2) or (B, 8)
        :param global_offset: (B, 2) or None
        :return: (B, 3, 3)
        """

        corners = GeometryUtils.gen_corners_torch(height, width).to(offsets.device)
        offsets = offsets.view(offsets.shape[0], 4, 2)
        return GeometryUtils.get_batch_homography_from_pts_offsets(corners, offsets, global_offset)

    @staticmethod
    def get_batch_homography_from_flow_field_sub(grid, flow_field, global_offset=None):
        """
        :param grid: (B, H, W, 2) or (1, H, W, 2), 'xy' indexed
        :param flow_field: (B, H, W, 2), 'xy' indexed
        :param global_offset: (B, 2) or None
        :return: (B, 3, 3)
        """

        src_pts = grid.view(grid.shape[0], -1, 2)
        batch_size = flow_field.shape[0]
        if src_pts.shape[0] < batch_size:
            q = batch_size // src_pts.shape[0]
            src_pts = src_pts.repeat(q, 1, 1)

        if global_offset is not None:
            n = src_pts.shape[1]
            src_pts = src_pts + global_offset.unsqueeze(1).repeat(1, n, 1)

        offsets = flow_field.view(batch_size, -1, 2)
        dst_pts = src_pts + offsets

        h = kornia.geometry.find_homography_dlt(src_pts, dst_pts)
        return h

    @staticmethod
    def get_batch_homography_from_flow_field(flow_field, global_offset=None):
        """
        :param flow_field: (B, H, W, 2), 'xy' indexed
        :param global_offset: (B, 2) or None
        :return: (B, 3, 3)
        """

        height, width = flow_field.shape[1], flow_field.shape[2]
        grid = GeometryUtils.gen_2d_grid_torch(height, width).to(flow_field.device)
        return GeometryUtils.get_batch_homography_from_flow_field_sub(grid, flow_field, global_offset)
