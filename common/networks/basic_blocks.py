import torch

from common.definitions import ActFunc, NormLayer
from common.networks import activation_layers, norm_layers


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm_layer=NormLayer.BATCH_NORM, activation=ActFunc.RELU, padding="same"):
        super().__init__()

        bias = not (norm_layer == NormLayer.BATCH_NORM)
        if stride != 1:
            padding = (kernel_size - 1) // 2

        self._block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            norm_layers.get_2d_layer(norm_layer, out_channels),
            activation_layers.get_layer(activation)
            )

    def forward(self, x):
        return self._block(x)


class UpConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, norm_layer=NormLayer.BATCH_NORM, activation=ActFunc.RELU):
        super().__init__()

        bias = not (norm_layer == NormLayer.BATCH_NORM)

        self._block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=bias),
            norm_layers.get_2d_layer(norm_layer, out_channels),
            activation_layers.get_layer(activation)
            )

    def forward(self, x):
        return self._block(x)


class UpConvBlock2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_layer=NormLayer.BATCH_NORM, activation=ActFunc.RELU):
        super().__init__()

        bias = not (norm_layer == NormLayer.BATCH_NORM)

        self._block = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=bias, padding="same"),
            norm_layers.get_2d_layer(norm_layer, out_channels),
            activation_layers.get_layer(activation)
            )

    def forward(self, x):
        return self._block(x)


class ResidualBlock(torch.nn.Module):  # ResNet-D
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=NormLayer.BATCH_NORM, activation=ActFunc.RELU):
        super(ResidualBlock, self).__init__()

        bias = not (norm_layer == NormLayer.BATCH_NORM)

        self._conv_blk = ConvBlock(in_channels, out_channels, 3, stride, norm_layer, activation)
        self._conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self._norm2 = norm_layers.get_2d_layer(norm_layer, out_channels)

        self._adjust_shape = torch.nn.Identity()
        if stride != 1 or in_channels != out_channels:
            pool = torch.nn.Identity() if stride == 1 else torch.nn.AvgPool2d(kernel_size=stride, stride=stride)
            self._adjust_shape = torch.nn.Sequential(
                pool,  # ResNet-D
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
                )

        self._activation = activation_layers.get_layer(activation)

    def forward(self, x):
        skip = x
        skip = self._adjust_shape(skip)  # skip path

        out = self._conv_blk(x)
        out = self._conv2(out)
        out = self._norm2(out)

        out += skip
        out = self._activation(out)

        return out


class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels, emb_dim, patch_size):
        super().__init__()

        self._embedder = torch.nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image):
        x = self._embedder(image)
        x = x.view(x.shape[0], x.shape[1], -1)  # (B, C, L)
        x = x.permute(0, 2, 1)  # (B, L, C)
        return x


class Pos2dEncodingAdder(torch.nn.Module):
    def __init__(self, max_len, emb_dim):
        super().__init__()

        pe = torch.zeros(max_len, emb_dim)  # (L, C)

        pos = torch.arange(0, max_len).unsqueeze(-1)  # (L, 1)
        two_i = torch.arange(0, emb_dim, 2)

        denomi = torch.pow(10000, two_i / emb_dim)

        pe[:, 0::2] = torch.sin(pos / denomi)
        pe[:, 1::2] = torch.cos(pos / denomi)
        pe = pe.unsqueeze(0)  # (1, L, C)

        self.register_buffer('_pe', pe)

    def forward(self, x):
        """
        :param x: (B, L, C)
        :return: (B, L, C)
        """
        batch_size, len = x.shape[:2]
        return x + self._pe.repeat(batch_size, 1, 1)[:, :len, :]


class Pos2dEncodingConcat(torch.nn.Module):
    def __init__(self, max_len):
        super().__init__()

        pos = torch.arange(0, max_len).unsqueeze(-1)  # (L, 1)

        pe = torch.concat([torch.sin(pos), torch.cos(pos)], dim=1)
        pe = pe.unsqueeze(0)  # (1, L, 2)

        self.register_buffer('_pe', pe)

    def forward(self, x):
        """
        :param x: (B, L, C)
        :return: (B, L, C + 2)
        """
        batch_size, len = x.shape[:2]
        pe = self._pe.repeat(batch_size, 1, 1)[:, :len, :]
        result = torch.concat([x, pe], dim=2)
        return result


class MultiHeadLinearAttention(torch.nn.Module):
    """
    batch-first
    """
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self._num_heads = num_heads

        self._prj_q = torch.nn.Linear(d_model, d_model)
        self._prj_k = torch.nn.Linear(d_model, d_model)
        self._prj_v = torch.nn.Linear(d_model, d_model)

    def _map_feature(self, x):
        return torch.nn.functional.elu(x) + 1

    def forward(self, query, key, value, mask_q=None, mask_kv=None):
        """
        :param query: (B, L, C)
        :param key: (B, L, C)
        :param value: (B, L, C)
        :param mask_q: (B, L, C)
        :param mask_kv: (B, L, C)
        :return: (B, L, H, D), D = C//H
        """
        eps = 1e-6

        batch_size, length = query.shape[:2]
        prj_q = self._prj_q(query).view(batch_size, length, self._num_heads, -1)  # (B, L, H, D), D = C//H
        prj_k = self._prj_k(key).view(batch_size, length, self._num_heads, -1)
        prj_v = self._prj_v(value).view(batch_size, length, self._num_heads, -1)

        prj_q = self._map_feature(prj_q)
        prj_k = self._map_feature(prj_k)

        if mask_q is not None:
            prj_q = prj_q * mask_q[:, :, None, None]
        if mask_kv is not None:
            prj_k = prj_k * mask_kv[:, :, None, None]
            prj_v = prj_v * mask_kv[:, :, None, None]

        prj_v = prj_v / prj_v.shape[1]  # prevent fp16 overflow

        summed_k = prj_k.sum(dim=1)  # (B, H, D)
        kv = torch.einsum("nlhd,nlhv->nhdv", prj_k, prj_v)  # along L-axis
        denomi = 1 / (eps + torch.einsum("nlhd,nhd->nlh", prj_q, summed_k))  # along D-axis
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", prj_q, kv, denomi)  # along D-axis

        queried_values = queried_values * prj_v.shape[1]  # restore scale

        return queried_values.contiguous()
