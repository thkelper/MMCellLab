import copy
from distutils.dep_util import newer_pairwise
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.encoders._utils import replace_strides_with_dilation
import segmentation_models_pytorch.base.initialization as init
from .builder import BASE_MODELS, build_backbone
# from .ops.compact_bilinear_pooling import CompactBilinearPooling

class DeepLabV3Decoder(nn.Sequential):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(
            ASPP(in_channels, out_channels, atrous_rates),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, *features):
        return super().forward(features[-1])


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        # highres_in_channels = encoder_channels[-4 if output_stride==32 else -3]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        # high_res_features = self.block1(features[-4 if self.output_stride==32 else -3])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super().__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)

    def bayarConstraint(self):
        # self.kernel.data = self.kernel.permute(2, 0, 1)
        # self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        # self.kernel.data = self.kernel.permute(1, 2, 0)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(2).unsqueeze(-1))

        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x


class SRMConv2d(nn.Module):
    def __init__(self, ):
        super().__init__()
        filter1 = torch.tile(torch.tensor([[0., 0, 0, 0, 0],
                                            [0, -1, 2, -1, 0],
                                            [0, 2, -4, 2, 0],
                                            [0, -1, 2, -1, 0],
                                            [0, 0, 0, 0, 0]]) / 4.0, 
                            dims=(3, 1, 1))
        filter2 = torch.tile(torch.tensor([[-1., 2, -2, 2, -1],
                                            [2, -6, 8, -6, 2],
                                            [-2, 8, -12, 8, -2],
                                            [2, -6, 8, -6, 2],
                                            [-1, 2, -2, 2, -1]]) / 12.0, 
                            dims=(3, 1, 1))
        filter3 =  torch.tile(torch.tensor([[0., 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0],
                                            [0, 1, -2, 1, 0],
                                            [0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0]]) / 2.0, 
                             dims=(3, 1, 1))
        self.filters = nn.Parameter(torch.stack([filter1, filter2, filter3], dim=0),
                                    requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.filters, stride=1, padding='same')
        return x


class HighPassConv2d(nn.Module):
    def __init__(self,):
        """
        Reference: Localization of Deep Inpainting Using High-Pass Fully Convolutional Network
        """
        super().__init__()
        filter1 = torch.tensor([[0., 0., 0.],
                                [0., -1., 0.],
                                [0., 1., 0.]])
        filter2 = torch.tensor([[0., 0., 0.],
                                [0., -1., 1.],
                                [0., 0., 0.]])
        filter3 = torch.tensor([[0., 0., 0.], 
                                [0., -1., 0.], 
                                [0., 0., 1.]])
        self.filters1 = torch.nn.Parameter(torch.stack([filter1, filter1, filter1], dim=0).unsqueeze(dim=1), 
                                        requires_grad=True)
        self.filters2 = torch.nn.Parameter(torch.stack([filter2, filter2, filter2], dim=0).unsqueeze(dim=1), 
                                        requires_grad=True)
        self.filters3 = torch.nn.Parameter(torch.stack([filter3, filter3, filter3], dim=0).unsqueeze(dim=1), 
                                        requires_grad=True)
                                      
    def forward(self, x):
        x1 = F.conv2d(x, self.filters1, stride=1, padding='same', groups=3)
        x2 = F.conv2d(x, self.filters2, stride=1, padding='same', groups=3)
        x3 = F.conv2d(x, self.filters3, stride=1, padding='same', groups=3)
        return torch.cat([x1, x2, x3], dim=1)


@BASE_MODELS.register_module()
class DeforgeFormer(SegmentationModel):
    def __init__(
        self,
        pre_conv_cfg=None,
        rgb_backbone=None,
        classifier_cfg=None,
        seg_cfg=None,
        decoder_cfg=None,
        encoder_output_stride: int = 16,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError("Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))

        self.pre_conv = nn.Identity()
        if pre_conv_cfg is not None:
            pre_conv_cfg = copy.deepcopy(pre_conv_cfg)
            name = pre_conv_cfg.pop('name')
            if name == 'BayarConv2d':
                self.pre_conv = BayarConv2d(**pre_conv_cfg)

        self.rgb_encoder = build_backbone(rgb_backbone)
        out_channels = self.rgb_encoder.out_channels[-1]

        self.decoder = None
        if decoder_cfg is not None:
            if encoder_output_stride == 16:
                replace_strides_with_dilation(self.rgb_encoder.patch_embed4, 2)
            elif encoder_output_stride == 8:
                replace_strides_with_dilation(self.rgb_encoder.patch_embed4, 2)
                replace_strides_with_dilation(self.rgb_encoder.patch_embed3, 2)

            self.decoder = DeepLabV3PlusDecoder(
                encoder_channels=self.rgb_encoder.out_channels,
                output_stride=encoder_output_stride,
                **decoder_cfg,
            )
            out_channels = self.decoder.out_channels

        self.segmentation_head = SegmentationHead(out_channels, **seg_cfg)

        if classifier_cfg is not None:
            self.classification_head = ClassificationHead(in_channels=out_channels, **classifier_cfg)
        else:
            self.classification_head = None

        self.initialize()

    def initialize(self):
        if self.decoder is not None:
            init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)
        features = self.rgb_encoder(x)
        if self.decoder is not None:
            decoder_output = self.decoder(*features)
        else:
            decoder_output = features[-1]

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(decoder_output)
            return labels, masks 

        return masks


@BASE_MODELS.register_module()
class DeforgeFormer2(nn.Module):
    def __init__(
        self,
        pre_conv_cfg=None,
        rgb_backbone=None,
        classifier_cfg=None,
        seg_cfg=None,
        decoder_cfg=None,
        encoder_output_stride: int = 16,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError("Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))

        pre_conv_cfg = copy.deepcopy(pre_conv_cfg)
        name = pre_conv_cfg.pop('name')
        self.pre_conv = BayarConv2d(**pre_conv_cfg)
        self.bayar_rgb_encoder = build_backbone(rgb_backbone)

        self.rgb_encoder = build_backbone(rgb_backbone)
        out_channels = self.rgb_encoder.out_channels[-1]

        self.decoder = None
        if decoder_cfg is not None:
            if encoder_output_stride == 16:
                replace_strides_with_dilation(self.rgb_encoder.patch_embed4, 2)
                replace_strides_with_dilation(self.bayar_rgb_encoder.patch_embed4, 2)
            elif encoder_output_stride == 8:
                replace_strides_with_dilation(self.rgb_encoder.patch_embed4, 2)
                replace_strides_with_dilation(self.rgb_encoder.patch_embed3, 2)
                replace_strides_with_dilation(self.bayar_rgb_encoder.patch_embed4, 2)
                replace_strides_with_dilation(self.bayar_rgb_encoder.patch_embed3, 2)

            self.decoder = DeepLabV3PlusDecoder(
                encoder_channels=[c*2 for c in self.rgb_encoder.out_channels],
                output_stride=encoder_output_stride,
                **decoder_cfg,
            )
            out_channels = self.decoder.out_channels

        self.segmentation_head = SegmentationHead(out_channels, **seg_cfg)

        if classifier_cfg is not None:
            self.classification_head = ClassificationHead(in_channels=out_channels, **classifier_cfg)
        else:
            self.classification_head = None

        self.initialize()

    def initialize(self):
        if self.decoder is not None:
            init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        bayar_features = self.pre_conv(x)
        bayar_features = self.bayar_rgb_encoder(bayar_features)

        rgb_features = self.rgb_encoder(x)

        features = []
        for r, b in zip(rgb_features, bayar_features):
            features.append(torch.cat([r, b], dim=1))
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(decoder_output)
            return labels, masks 

        return masks


class Filter(nn.Module):
    def __init__(self, size, 
                 band_start, 
                 band_end, 
                 use_learnable=True, 
                 norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()
        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        return out


def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)

        # init filters
        self.filters = nn.ModuleList([Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i+1), norm=True) for i in range(M)])
    
    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8)/2) + 1
        assert size_after == 149

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)   # [N, C * S * S, L]   L:block num
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        # M kernels filtering
        y_list = []
        for i in range(self._M):
            # y = self.filters[i](x_dct)    # [N, L, C, S, S]
            # y = torch.abs(y)
            # y = torch.sum(y, dim=[2,3,4])   # [N, L]
            # y = torch.log10(y + 1e-15)
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2,3,4])
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)   # [N, 1, 149, 149]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
        return out
    

class MixBlock(nn.Module):
    
    def __init__(self, c_in, width, height):
        super(MixBlock, self).__init__()
        self.FAD_query = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1,1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1,1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)    # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)    # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  #[BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))
        return y_FAD, y_LFS



@BASE_MODELS.register_module()
class DeforgeFormer3(nn.Module):
    def __init__(
        self,
        use_rgb=True,
        rgb_conv_cfg=None,
        bayar_conv_cfg=None,
        only_bayar=False,
        srm_conv=False,
        fad_cfg=None,
        rgb_backbone=None,
        classifier_cfg=None,
        seg_cfg=None,
        decoder_cfg=None,
        encoder_output_stride: int = 16,
        hp_conv=False,
        f_fusion_mode='concat',
        pretrained=None,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError("Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))

        self.use_rgb = use_rgb
        
        self.rgb_conv = None
        if rgb_conv_cfg is not None:
            out_channels = rgb_conv_cfg.out_channels
            if rgb_conv_cfg.relu:
                self.rgb_conv = nn.Sequential(
                    nn.Conv2d(3, out_channels, 
                            kernel_size=rgb_conv_cfg.kernel_size, 
                            padding='same',
                            stride=1,
                            bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            else:
                self.rgb_conv = nn.Sequential(
                    nn.Conv2d(3, out_channels, 
                            kernel_size=rgb_conv_cfg.kernel_size, 
                            padding='same',
                            stride=1,
                            bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        self.bayar_pre_conv = None
        if bayar_conv_cfg is not None:
            self.bayar_pre_conv = BayarConv2d(**bayar_conv_cfg)
        self.only_bayar = only_bayar

        self.srm_conv = None
        if srm_conv:
            self.srm_conv = SRMConv2d()
        
        self.hp_conv = None
        if hp_conv:
            self.hp_conv = HighPassConv2d()

        self.fad = None
        if fad_cfg:
            self.fad = FAD_Head(**fad_cfg)

        self.rgb_encoder = build_backbone(rgb_backbone)
        out_channels = self.rgb_encoder.out_channels[-1]

        self.f_fusion_mode = f_fusion_mode
        # if self.f_fusion_mode == 'bilinear_pool':
        #     self.compact_bilinear_pool = CompactBilinearPooling(3, 3, 3)

        self.decoder = None
        if decoder_cfg is not None:
            if encoder_output_stride == 16:
                replace_strides_with_dilation(self.rgb_encoder.patch_embed4, 2)
            elif encoder_output_stride == 8:
                replace_strides_with_dilation(self.rgb_encoder.patch_embed4, 2)
                replace_strides_with_dilation(self.rgb_encoder.patch_embed3, 2)

            self.decoder = DeepLabV3PlusDecoder(
                encoder_channels=[c for c in self.rgb_encoder.out_channels],
                output_stride=encoder_output_stride,
                **decoder_cfg,
            )
            out_channels = self.decoder.out_channels

        self.segmentation_head = None
        if seg_cfg is not None:
            self.segmentation_head = SegmentationHead(out_channels, **seg_cfg)

        if classifier_cfg is not None:
            self.classification_head = ClassificationHead(in_channels=out_channels, **classifier_cfg)
        else:
            self.classification_head = None

        self.initialize(pretrained)

    def initialize(self, pretrained):
        if pretrained:
            m = torch.load(pretrained)
            ret = self.load_state_dict(m, strict=False)
            print(ret)
        else:
            if self.decoder is not None:
                init.initialize_decoder(self.decoder)

        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        inputs = []
        if self.use_rgb:
            if self.rgb_conv is not None:
                inputs.append(self.rgb_conv(x))
            else:
                inputs.append(x)

        bayar_features = None
        if self.bayar_pre_conv is not None:
            bayar_features = self.bayar_pre_conv(x)
            inputs.append(bayar_features)
        
        if not self.only_bayar:
            if self.srm_conv is not None:
                noise_features = self.srm_conv(x)
                inputs.append(noise_features)

            if self.hp_conv is not None:
                inputs.append(self.hp_conv(x))

            if self.fad is not None:
                inputs.append(self.fad(x) / 2)

        if len(inputs) > 1:
            if self.f_fusion_mode == 'concat':
                # for i, input in enumerate(inputs):
                #     print('---->', i, input.max(), input.min())
                inputs = torch.cat(inputs, dim=1)
            # elif self.f_fusion_mode == 'bilinear_pool':
            #     B, N, H, W  = inputs[0].shape
            #     x = inputs[0].permute(0, 2, 3, 1).reshape(-1, N)
            #     print('------->', x.shape, N)
            #     y = inputs[1].permute(0, 2, 3, 1).reshape(-1, N)
            #     inputs = self.compact_bilinear_pool(x, y)
            #     inputs = inputs.reshape(B, H, W, N).permute(0, 3, 1, 2)
            elif self.f_fusion_mode == 'element_add':
                inputs_ = inputs[0]
                for i  in range(1, len(inputs)):
                    inputs_ = inputs_ + inputs[i]
                inputs = inputs_
            elif self.f_fusion_mode == 'element_mul':
                inputs_ = inputs[0]
                for i  in range(1, len(inputs)):
                    inputs_ = inputs_ * inputs[i]
                inputs = inputs_
        else:
            inputs = inputs[0]

        features = self.rgb_encoder(inputs)
        decoder_output = self.decoder(*features)
        
        masks = self.segmentation_head(decoder_output) if self.segmentation_head else None
        labels = self.classification_head(decoder_output) if self.classification_head else None
    
        if masks is not None and labels is not None:
            return labels, masks
        elif masks is not None:
            return masks
        else:
            return labels

@BASE_MODELS.register_module()
class DeforgeFormer4(nn.Module):
    def __init__(
        self,
        use_rgb=True,
        rgb_conv_cfg=None,
        bayar_conv_cfg=None,
        only_bayar=False,
        srm_conv=False,
        fad_cfg=None,
        rgb_backbone=None,
        classifier_cfg=None,
        seg_cfg=None,
        decoder_cfg=None,
        encoder_output_stride: int = 16,
        hp_conv=False,
        f_fusion_mode='concat',
        pretrained=None,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError("Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))

        self.use_rgb = use_rgb
        
        self.rgb_conv = None
        if rgb_conv_cfg is not None:
            out_channels = rgb_conv_cfg.out_channels
            if rgb_conv_cfg.relu:
                self.rgb_conv = nn.Sequential(
                    nn.Conv2d(3, out_channels, 
                            kernel_size=rgb_conv_cfg.kernel_size, 
                            padding='same',
                            stride=1,
                            bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            else:
                self.rgb_conv = nn.Sequential(
                    nn.Conv2d(3, out_channels, 
                            kernel_size=rgb_conv_cfg.kernel_size, 
                            padding='same',
                            stride=1,
                            bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        self.bayar_pre_conv = None
        if bayar_conv_cfg is not None:
            self.bayar_pre_conv = BayarConv2d(**bayar_conv_cfg)
        self.only_bayar = only_bayar

        self.srm_conv = None
        if srm_conv:
            self.srm_conv = SRMConv2d()
        
        self.hp_conv = None
        if hp_conv:
            self.hp_conv = HighPassConv2d()

        self.fad = None
        if fad_cfg:
            self.fad = FAD_Head(**fad_cfg)

        self.rgb_encoder = build_backbone(rgb_backbone)
        out_channels = self.rgb_encoder.out_channels[-1]

        self.f_fusion_mode = f_fusion_mode
        # if self.f_fusion_mode == 'bilinear_pool':
        #     self.compact_bilinear_pool = CompactBilinearPooling(3, 3, 3)

        self.decoder = None
        if decoder_cfg is not None:
            if encoder_output_stride == 16:
                replace_strides_with_dilation(self.rgb_encoder.patch_embed4, 2)
            elif encoder_output_stride == 8:
                replace_strides_with_dilation(self.rgb_encoder.patch_embed4, 2)
                replace_strides_with_dilation(self.rgb_encoder.patch_embed3, 2)

            self.decoder = DeepLabV3PlusDecoder(
                encoder_channels=[c for c in self.rgb_encoder.out_channels],
                output_stride=encoder_output_stride,
                **decoder_cfg,
            )
            out_channels = self.decoder.out_channels

        self.segmentation_head = None
        if seg_cfg is not None:
            self.segmentation_head = SegmentationHead(out_channels, **seg_cfg)

        if classifier_cfg is not None:
            self.classification_head = ClassificationHead(in_channels=out_channels, **classifier_cfg)
        else:
            self.classification_head = None

        self.initialize(pretrained)

    def initialize(self, pretrained):
        if pretrained:
            m = torch.load(pretrained)
            ret = self.load_state_dict(m, strict=False)
            print(ret)
        if self.decoder is not None:
            init.initialize_decoder(self.decoder)

        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        inputs = []
        if self.use_rgb:
            if self.rgb_conv is not None:
                inputs.append(self.rgb_conv(x))
            else:
                inputs.append(x)

        bayar_features = None
        if self.bayar_pre_conv is not None:
            bayar_features = self.bayar_pre_conv(x)
            inputs.append(bayar_features)
        
        if not self.only_bayar:
            if self.srm_conv is not None:
                noise_features = self.srm_conv(x)
                inputs.append(noise_features)

            if self.hp_conv is not None:
                inputs.append(self.hp_conv(x))

            if self.fad is not None:
                inputs.append(self.fad(x) / 2)

        if len(inputs) > 1:
            if self.f_fusion_mode == 'concat':
                # for i, input in enumerate(inputs):
                #     print('---->', i, input.max(), input.min())
                inputs = torch.cat(inputs, dim=1)
            # elif self.f_fusion_mode == 'bilinear_pool':
            #     B, N, H, W  = inputs[0].shape
            #     x = inputs[0].permute(0, 2, 3, 1).reshape(-1, N)
            #     print('------->', x.shape, N)
            #     y = inputs[1].permute(0, 2, 3, 1).reshape(-1, N)
            #     inputs = self.compact_bilinear_pool(x, y)
            #     inputs = inputs.reshape(B, H, W, N).permute(0, 3, 1, 2)
            elif self.f_fusion_mode == 'element_add':
                inputs_ = inputs[0]
                for i  in range(1, len(inputs)):
                    inputs_ = inputs_ + inputs[i]
                inputs = inputs_
            elif self.f_fusion_mode == 'element_mul':
                inputs_ = inputs[0]
                for i  in range(1, len(inputs)):
                    inputs_ = inputs_ * inputs[i]
                inputs = inputs_
        else:
            inputs = inputs[0]

        features = self.rgb_encoder(inputs)
        decoder_output = self.decoder(*features)
        
        masks = self.segmentation_head(decoder_output) if self.segmentation_head else None
        labels = self.classification_head(decoder_output) if self.classification_head else None
    
        if masks is not None and labels is not None:
            return labels, masks
        elif masks is not None:
            return masks
        else:
            return labels
