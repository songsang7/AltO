import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

from common.definitions import ActFunc, NormLayer
from common.utils import GeometryUtils
from common.networks.basic_blocks import ConvBlock, ResidualBlock
from image_matching.utils import ImageMatchingUtils


class RHWFNet(torch.nn.Module):
    def __init__(self, in_channels, height, width, use_lv2):
        super().__init__()

        self._num_iter_lv1 = 6
        self._num_iter_lv2 = 6

        self._num_pyramid_lv = 1
        self._radius = 8
        self._kernel_size = 2 * self._radius + 1

        self._feature_extractor = FeatureExtractor(in_channels, 96)
        self._feat_h = height // 4
        self._feat_w = width // 4

        self._focus_kernel_list = [0, 9, 5, 3, 3, 3]  # 0 : global
        self._focus_pad_list = [0, 4, 2, 1, 1, 1]
        self._focus_former = FocusFormer_Attention(96, 1, 96, 96)

        in_channels_gma = (self._radius**2) + ((self._kernel_size - self._radius)**2)
        self._global_motion_aggregator = GlobalMotionAggregator(in_channels_gma, 80)

        self.register_buffer("_grid_q", GeometryUtils.gen_2d_grid_torch(self._feat_h, self._feat_w, False))
        self.register_buffer('_corners_q', GeometryUtils.gen_corners_torch(self._feat_h, self._feat_w))
        self.register_buffer('_corners', GeometryUtils.gen_corners_torch(height, width))

    def forward(self, image_moving, image_fixed):
        image_moving = 2 * image_moving - 1.0
        image_moving = image_moving.contiguous()

        image_fixed = 2 * image_fixed - 1.0
        image_fixed = image_fixed.contiguous()

        batch_size = image_moving.size(0)

        f_f_half, f_f_quad = self._feature_extractor(image_fixed)

        grid_q = self._grid_q.repeat(batch_size, 1, 1, 1).detach()
        corners_q = self._corners_q.repeat(batch_size, 1, 1)
        corners = self._corners.repeat(batch_size, 1, 1)
        displacements = torch.zeros((batch_size, 4, 2), device=image_moving.device)

        displacements_list = []
        for i in range(self._num_iter_lv1):
            h_q = GeometryUtils.get_batch_homography_from_pts_offsets(corners_q, displacements / 4)
            h = GeometryUtils.get_batch_homography_from_pts_offsets(corners, displacements)
            flow_field = GeometryUtils.get_batch_flow_field_from_homography_sub(grid_q, h_q)

            warped_moving = GeometryUtils.get_batch_warped_image_from_homography(image_moving, h.detach())
            f_m_half, f_m_quad = self._feature_extractor(warped_moving)
            f_f, f_m = self._focus_former(f_f_quad, f_m_quad, self._focus_kernel_list[i], self._focus_pad_list[i])

            corr = F.relu(Correlation()(f_f.contiguous(), f_m.contiguous(), self._kernel_size, 0))  # (B, k*k, H, W)
            b, _, h, w = corr.shape
            corr = corr.permute(0, 2, 3, 1).contiguous()
            corr = corr.view(b, h * w, self._kernel_size, self._kernel_size)
            corr_1 = F.avg_pool2d(corr, 2).view(b, h, w, self._radius**2).permute(0, 3, 1, 2)  # (B, r*r, H, W)
            half_r = self._radius // 2
            corr_2 = corr[:, :, half_r:self._kernel_size - half_r, half_r:self._kernel_size - half_r].contiguous().view(b, h, w, -1).permute(0, 3, 1, 2)  # (B, (k-r)^2, H, W)
            corr = torch.cat([corr_1, corr_2], dim=1)

            delta_disps = self._global_motion_aggregator(corr, flow_field)

            displacements = displacements + delta_disps  # += 쓰는 것과 다르다. += 쓰면 in-place operation
            displacements_list.append(displacements)

        h = GeometryUtils.get_batch_homography_from_pts_offsets(self._corners, displacements_list[-1])
        return displacements_list, h


class FeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self._stem = ConvBlock(in_channels, 32, kernel_size=3, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F, padding=1)

        self._stage_1 = torch.nn.Sequential(
            ResidualBlock(32, 56, stride=2, norm_layer=NormLayer.INSTANCE_NORM_F),
            ResidualBlock(56, 56, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F)
            )

        self._head_1 = torch.nn.Conv2d(56, out_channels, kernel_size=1)

        self._stage_2 = torch.nn.Sequential(
            ResidualBlock(56, 80, stride=2, norm_layer=NormLayer.INSTANCE_NORM_F),
            ResidualBlock(80, 80, stride=1, norm_layer=NormLayer.INSTANCE_NORM_F)
            )

        self._head_2 = torch.nn.Conv2d(80, out_channels, kernel_size=1)

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

        self._conv_corr = torch.nn.Sequential(
            nn.Conv2d(in_channels, 128, 1, padding=0, stride=1),
            nn.ReLU()
            )

        self._stem = torch.nn.Sequential(
            torch.nn.Conv2d(128 + 2, 128, 3, padding=1, stride=1),
            torch.nn.ReLU()
            )

        self._layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, hidden_channels, 3, padding=1, stride=1),
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
            corr_block: (B, C, H, W)
            flow_field: (B, H, W, 2)
        Returns: (B, 2*2, 2)
        """
        corr_block = self._conv_corr(corr_block)  # (B, 128, H, W)
        flow_field = flow_field.permute(0, 3, 1, 2)
        x = torch.concat([corr_block, flow_field], dim=1)  # (B, 130, H, W)
        x = self._stem(x)  # (B, 128, H, W)

        x = self._layer1(x)
        x = self._layer2(x)
        x = self._layer3(x)
        x = self._layer4(x)
        x = self._layer10(x)

        x = x.view(x.shape[0], x.shape[1], -1)
        return x.permute(0, 2, 1)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def padding(input, pad_size):
    B, H, W, C = input.shape
    t_input = torch.zeros((B, H + 2 * pad_size, W + 2 * pad_size, C), dtype=torch.float32, device=input.device)
    t_input[:, pad_size:pad_size + H, pad_size:pad_size + W, :] = input.clone()
    return t_input.contiguous()


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, norm=False):
        super().__init__()
        self.w_1 = nn.Conv2d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv2d(d_hid, d_in, 1)  # position-wise
        self.ln = LayerNorm2d(d_in)
        self.norm = norm

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        if self.norm:
            x = self.ln(x)
        x += residual
        return x


class Correlation:
    def __call__(self, q, k, kernel, pad):
        b, c, h, w = q.shape
        pyramid = ImageMatchingUtils.calc_corr_pyramid(q, k, 1, False)
        loc_field = GeometryUtils.gen_2d_grid_torch(h, w).to(q.device).repeat(b, 1, 1, 1)
        radius = (kernel - 1) // 2
        corr_block = ImageMatchingUtils.get_corr_block(loc_field, pyramid, radius)
        return corr_block


class ChannelAttention:
    def __call__(self, qk, v, kernel, pad):
        b, c, h, w = v.shape
        v = v.view(b*c, 1, h, w)  # (BC, 1, H, W)

        radius = (kernel - 1) // 2
        v_block = torch.nn.Unfold(kernel, padding=radius)(v).view(b, c, kernel*kernel, h, w)  # (B, C, kernel*kernel, H, W)
        qk = qk.view(b, 1, kernel*kernel, h, w).repeat(1, c, 1, 1, 1)  # (B, C, k*k, H, W)
        result = (qk * v_block).sum(dim=2)  # (B, C, H, W)
        return result


class single_head_local_attention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, kernel, pad, show=False):
        corr = Correlation()(q, k, kernel, pad)  # (B, kernel*kernel, H, W)
        corr = torch.softmax(corr / self.temperature, dim=1)
        result = ChannelAttention()(corr, v, kernel, pad)
        if show:
            return result, corr
        else:
            return result


def single_head_global_attention(q, k, v, show=False):
    # q, k, v: [B, H*W, C]
    [B, C, H, W] = q.shape
    q = q.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
    k = k.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
    v = v.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, H*W, H*W]
    attn = torch.softmax(scores, dim=2)  # [B, H*W, H*W]
    out = torch.matmul(attn, v)  # [B, H*W, C]
    out = out.permute(0, 2, 1)
    out = out.reshape([B, C, H, W])

    if show:
        return out, attn
    else:
        return out


class Multi_head_focus_attention(nn.Module):
    """
        multi_head_focus_attention
    """

    def __init__(self, n_head, d_model, d_k, d_v, norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Conv2d(d_model, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(d_model, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(d_model, n_head * d_v, 1, bias=False)
        self.fc = nn.Conv2d(n_head * d_v, d_model, 1, bias=False)

        self.ln = LayerNorm2d(d_model)
        self.norm = norm

    def forward(self, q, k, v, kernel, pad, show=False):

        residual = q

        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        if kernel == 0:
            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
            res_q = []
            att_list = []
            for i in range(n_head):
                q_t, k_t, v_t = q[:, i * d_k:(i + 1) * d_k], k[:, i * d_k:(i + 1) * d_k], v[:, i * d_v:(i + 1) * d_v]
                if show:
                    q_t, att = single_head_global_attention(q_t, k_t, v_t, show)
                    att_list.append(att)
                else:
                    q_t = single_head_global_attention(q_t, k_t, v_t)
                res_q.append(q_t)
            q = torch.cat(res_q, dim=1)
        else:
            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
            attention = single_head_local_attention(d_k ** 0.5)
            res_q = []
            att_list = []
            for i in range(n_head):
                q_t, k_t, v_t = q[:, i * d_k:(i + 1) * d_k], k[:, i * d_k:(i + 1) * d_k], v[:, i * d_v:(i + 1) * d_v]
                if show:
                    q_t, att = attention(q_t, k_t, v_t, kernel, pad, show)
                    att_list.append(att)
                else:
                    q_t = attention(q_t, k_t, v_t, kernel, pad)
                res_q.append(q_t)
            q = torch.cat(res_q, dim=1)

        q = self.fc(q)
        q += residual
        if self.norm:
            q = self.ln(q)
        if show:
            return q, att_list
        else:
            return q


class FocusFormer_Attention(nn.Module):
    """
        self_attn + cross_attn + ffn
    """

    def __init__(self, in_planes, n_head, d_k, d_v):
        super(FocusFormer_Attention, self).__init__()
        self.slf_attn = Multi_head_focus_attention(n_head, in_planes, d_k, d_v, norm=True)
        self.crs_attn = Multi_head_focus_attention(n_head, in_planes, d_k, d_v, norm=True)
        self.pos_ffn = PositionwiseFeedForward(in_planes, in_planes, norm=True)

    def forward(self, input1, input2, kernel, pad, show=False):
        if show:
            slf_1, slf_att_1 = self.slf_attn(input1, input1, input1, kernel, pad, show)
            slf_2, slf_att_2 = self.slf_attn(input2, input2, input2, kernel, pad, show)
            crs_1, crs_att_1 = self.crs_attn(slf_1, slf_2, slf_2, kernel, pad, show)
            crs_2, crs_att_2 = self.crs_attn(slf_2, slf_1, slf_1, kernel, pad, show)
            crs_1 = self.pos_ffn(crs_1)
            crs_2 = self.pos_ffn(crs_2)
            return crs_1, crs_2, slf_att_1, slf_att_2, crs_att_1, crs_att_2
        else:
            slf_1 = self.slf_attn(input1, input1, input1, kernel, pad)
            slf_2 = self.slf_attn(input2, input2, input2, kernel, pad)
            crs_1 = self.crs_attn(slf_1, slf_2, slf_2, kernel, pad)
            crs_2 = self.crs_attn(slf_2, slf_1, slf_1, kernel, pad)
            crs_1 = self.pos_ffn(crs_1)
            crs_2 = self.pos_ffn(crs_2)
            return crs_1, crs_2
