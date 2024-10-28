import torch

from common.definitions import ActFunc, NormLayer
from common.utils import GeometryUtils
from common.networks.basic_blocks import ConvBlock, ResidualBlock
from image_matching.utils import ImageMatchingUtils


class IHNNet(torch.nn.Module):
    def __init__(self, in_channels, height, width, use_lv2):
        super().__init__()

        self._num_iter_lv1 = 6
        self._num_iter_lv2 = 3

        self._num_pyramid_lv = 2
        self._radius = 4

        self._feature_extractor = FeatureExtractor(in_channels, 256)
        self._feat_h = height // 4
        self._feat_w = width // 4

        in_channels_gma = 2 + self._num_pyramid_lv * ((2 * self._radius + 1)**2)
        self._global_motion_aggregator = GlobalMotionAggregator(in_channels_gma, 128)

        self.register_buffer("_grid_q", GeometryUtils.gen_2d_grid_torch(self._feat_h, self._feat_w, False))
        self.register_buffer('_corners_q', GeometryUtils.gen_corners_torch(self._feat_h, self._feat_w))
        self.register_buffer('_corners', GeometryUtils.gen_corners_torch(height, width))

    def forward(self, image_moving, image_fixed):
        image_moving = 2 * image_moving - 1.0
        image_moving = image_moving.contiguous()

        image_fixed = 2 * image_fixed - 1.0
        image_fixed = image_fixed.contiguous()

        batch_size = image_moving.size(0)

        f_m_half, f_m_quad = self._feature_extractor(image_moving)
        f_f_half, f_f_quad = self._feature_extractor(image_fixed)

        corr_pyramid = ImageMatchingUtils.calc_corr_pyramid(f_f_quad, f_m_quad, 2)  # 순서 주의
        grid_q = self._grid_q.repeat(batch_size, 1, 1, 1)
        corners_q = self._corners_q.repeat(batch_size, 1, 1)
        displacements = torch.zeros((batch_size, 4, 2), device=image_moving.device)

        displacements_list = []
        for i in range(self._num_iter_lv1):
            h = GeometryUtils.get_batch_homography_from_pts_offsets(corners_q, displacements / 4)
            flow_field = GeometryUtils.get_batch_flow_field_from_homography_sub(grid_q, h)
            new_loc = grid_q + flow_field
            corr_block = ImageMatchingUtils.get_corr_block(new_loc, corr_pyramid, self._radius)  # (B, fov*fov*num_levels, H, W), fov=2r+1

            delta_disps = self._global_motion_aggregator(corr_block, flow_field)

            displacements = displacements + delta_disps  # += 쓰는 것과 다르다. += 쓰면 in-place operation
            displacements_list.append(displacements)

        h = GeometryUtils.get_batch_homography_from_pts_offsets(self._corners, displacements_list[-1])
        return displacements_list, h


class FeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self._stem = ConvBlock(in_channels, 64, kernel_size=7, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F, padding=3)

        self._stage_1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64, 64, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F),
            ResidualBlock(64, 64, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F)
            )

        self._head_1 = torch.nn.Conv2d(64, out_channels, kernel_size=1)

        self._stage_2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64, 96, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F),
            ResidualBlock(96, 96, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F)
            )

        self._head_2 = torch.nn.Conv2d(96, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d, torch.nn.GroupNorm)):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, image):
        x = self._stem(image)
        x = self._stage_1(x)
        out_1 = self._head_1(x)
        x = self._stage_2(x)
        out_2 = self._head_2(x)
        return out_1, out_2


class GlobalMotionAggregator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self._layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, 3, padding=1, stride=1),
            torch.nn.GroupNorm(num_groups=hidden_channels // 8, num_channels=hidden_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self._layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=1),
            torch.nn.GroupNorm(num_groups=hidden_channels // 8, num_channels=hidden_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self._layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=1),
            torch.nn.GroupNorm(num_groups=hidden_channels // 8, num_channels=hidden_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self._layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=1),
            torch.nn.GroupNorm(num_groups=hidden_channels // 8, num_channels=hidden_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

        ### global motion
        self._layer10 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=1),
            torch.nn.GroupNorm(num_groups=hidden_channels // 8, num_channels=hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channels, 2, 1)
            )

    def forward(self, corr_block, flow_field):
        """
        Args:
            corr_block: (B, fov*fov*num_levels, H, W), fov=2r+1
            flow_field: (B, H, W, 2)
        Returns: (B, 2*2, 2)
        """
        flow_field = flow_field.permute(0, 3, 1, 2)
        x = torch.concat([corr_block, flow_field], dim=1)

        x = self._layer1(x)
        x = self._layer2(x)
        x = self._layer3(x)
        x = self._layer4(x)
        x = self._layer10(x)

        x = x.view(x.shape[0], x.shape[1], -1)
        return x.permute(0, 2, 1)
