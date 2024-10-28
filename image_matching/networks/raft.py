import torch
import torchvision
from torchvision.models.optical_flow.raft import Conv2dNormActivation

from common.utils import GeometryUtils


class RaftLarge(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        backbone = torchvision.models.optical_flow.raft_large()
        backbone.feature_encoder.convnormrelu = Conv2dNormActivation(in_channels, 64, norm_layer=torch.nn.InstanceNorm2d, kernel_size=7, stride=2, bias=True)
        backbone.context_encoder.convnormrelu = Conv2dNormActivation(in_channels, 64, norm_layer=torch.nn.BatchNorm2d, kernel_size=7, stride=2, bias=True)
        self._model = backbone

        # self.register_buffer("_grid", GeometryUtils.gen_2d_grid_torch(height, width))

    def forward(self, image_moving, image_fixed, last_only):
        image_moving = 2 * image_moving - 1
        image_fixed = 2 * image_fixed - 1

        field_list = self._model(image_fixed, image_moving)  # 순서 유의

        permuted_field_list = []
        for field in field_list:
            permuted_field_list.append(field.permute(0, 2, 3, 1))

        if last_only:
            return permuted_field_list[-1]
        return permuted_field_list
