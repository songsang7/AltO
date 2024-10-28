import torch
import torchvision

from common.definitions import ActFunc, NormLayer
from common.networks.basic_blocks import ConvBlock, ResidualBlock


class IhnEncoder(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self._stem = ConvBlock(in_channels, 64, kernel_size=7, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F, padding=3, activation=ActFunc.MISH)

        self._stage_1 = torch.nn.Sequential(
            ResidualBlock(64, 64, stride=2, norm_layer=NormLayer.INSTANCE_NORM_F, activation=ActFunc.MISH),
            ResidualBlock(64, 64, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F, activation=ActFunc.MISH),
            )

        self._stage_2 = torch.nn.Sequential(
            ResidualBlock(64, 96, stride=2, norm_layer=NormLayer.INSTANCE_NORM_F, activation=ActFunc.MISH),
            ResidualBlock(96, 96, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F, activation=ActFunc.MISH)
            )

        self._head_2 = torch.nn.Conv2d(96, 256, kernel_size=1)

    def forward(self, image):
        image = 2 * image - 1
        x = self._stem(image)
        x = self._stage_1(x)
        x = self._stage_2(x)
        out_2 = self._head_2(x)
        return out_2


class ResNetEncoder(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        backbone = torchvision.models.resnet34()
        backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._encoder = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            # backbone.maxpool,

            backbone.layer1,
            backbone.layer2
            )

    def forward(self, image):
        image = 2 * image - 1
        return self._encoder(image)


class ResNetEncoder123(ResNetEncoder):
    def __init__(self, in_channels):
        super().__init__(in_channels)

        backbone = torchvision.models.resnet34()
        backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._encoder = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            # backbone.maxpool,

            backbone.layer1,
            backbone.layer2,
            backbone.layer3
            )


class ResNetEncoder1(ResNetEncoder):
    def __init__(self, in_channels):
        super().__init__(in_channels)

        backbone = torchvision.models.resnet34()
        backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._encoder = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            # backbone.maxpool,

            backbone.layer1
            )


class ResNetProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        backbone = torchvision.models.resnet34()
        self._prj_header = torch.nn.Sequential(
            backbone.layer3,
            # backbone.layer4,
            backbone.avgpool,
            torch.nn.Flatten()
            )
        
    def forward(self, x):
        return self._prj_header(x)


class ResNetProjector23(ResNetProjector):
    def __init__(self):
        super().__init__()

        backbone = torchvision.models.resnet34()
        self._prj_header = torch.nn.Sequential(
            backbone.layer2,
            backbone.layer3,
            backbone.avgpool,
            torch.nn.Flatten()
            )


class ResNetProjector34(ResNetProjector):
    def __init__(self):
        super().__init__()

        backbone = torchvision.models.resnet34()
        self._prj_header = torch.nn.Sequential(
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
            torch.nn.Flatten()
            )


class ResNetProjector234(ResNetProjector):
    def __init__(self):
        super().__init__()

        backbone = torchvision.models.resnet34()
        self._prj_header = torch.nn.Sequential(
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
            torch.nn.Flatten()
            )


class ResNetProjector4(ResNetProjector):
    def __init__(self):
        super().__init__()

        backbone = torchvision.models.resnet34()
        self._prj_header = torch.nn.Sequential(
            backbone.layer4,
            backbone.avgpool,
            torch.nn.Flatten()
            )


class FeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self._encoder = ResNetEncoder(in_channels)
        self._projector = ResNetProjector()

    def forward(self, image, enc_only):
        f = self._encoder(image)
        if enc_only:
            return f, None

        z = self._projector(f)
        return f, z


class FeatureExtractorE123P4(FeatureExtractor):
    def __init__(self, in_channels):
        super().__init__(in_channels)

        self._encoder = ResNetEncoder123(in_channels)
        self._projector = ResNetProjector4()


class FeatureExtractorE12P34(FeatureExtractor):
    def __init__(self, in_channels):
        super().__init__(in_channels)

        self._encoder = ResNetEncoder(in_channels)
        self._projector = ResNetProjector34()


class FeatureExtractorE1P234(FeatureExtractor):
    def __init__(self, in_channels):
        super().__init__(in_channels)

        self._encoder = ResNetEncoder1(in_channels)
        self._projector = ResNetProjector234()


class FeatureExtractorE1P23(FeatureExtractor):
    def __init__(self, in_channels):
        super().__init__(in_channels)

        self._encoder = ResNetEncoder1(in_channels)
        self._projector = ResNetProjector23()
