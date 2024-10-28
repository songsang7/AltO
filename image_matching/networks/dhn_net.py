import torch
import torchvision

from common.utils import GeometryUtils


class CnnRegistrator(torch.nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()

        self._model = torchvision.models.resnet34()
        self._model.conv1 = torch.nn.Conv2d(in_channels * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self._model.fc = torch.nn.Linear(512, 8, bias=True)

        self.register_buffer('_corners', GeometryUtils.gen_corners_torch(height, width))

    def forward(self, image_moving, image_fixed):
        image_moving = 2 * image_moving - 1
        image_fixed = 2 * image_fixed - 1

        x = torch.cat([image_moving, image_fixed], dim=1)
        x = self._model(x)

        h = GeometryUtils.get_batch_homography_from_pts_offsets(self._corners, x.view(-1, 4, 2))

        return x, h
