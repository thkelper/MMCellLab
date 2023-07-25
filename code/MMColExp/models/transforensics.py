import enum
import math
from turtle import forward
from typing import ForwardRef
import numpy as np
from functools import reduce
import operator
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
from .builder import BASE_MODELS, build_backbone
from .deforgeformer import BayarConv2d
from .psccnet import DCTStream, blocks_dict, BatchNorm2d, BN_MOMENTUM


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, h, w):
        b = x.shape[0]
        not_mask = torch.ones((b, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = torch.cat((pos_y, pos_x), dim=3).reshape(b, h*w, -1)
        return pos


class SeqPositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b, n = x.shape[0:2]
        not_mask = torch.ones((b, n), device=x.device)
        embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            embed = embed / (embed[:, -1:, :] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=2).flatten(2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(hidden_dim, position_embedding='v2'):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif position_embedding in ('seq', ):
        position_embedding = SeqPositionEmbeddingSine(num_pos_feats=hidden_dim)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module): 
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class AttentionLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads = heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, MLP(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    

class AttentionDecoder(nn.Module):
    def __init__(self, f_dim, dim, depth, heads, mlp_dim, dropout = 0., 
                fusion_mode='mul', 
                # fusion_thresh=0.5
                ):
        super().__init__()
        self.conv1 = nn.Conv2d(f_dim, dim, 1)
        self.p_sino_pos_embed = build_position_encoding(dim)
        self.att_layer = AttentionLayer(dim, depth, heads, dim//heads, mlp_dim, dropout)
        self.fusion_mode = fusion_mode
        # self.fusion_thresh = fusion_thresh

    def forward(self, c_f, p_f=None):
        c_f = self.conv1(c_f)
        b, c, h, w = c_f.shape
        c_f = c_f.reshape(b, c, -1).permute(0, 2, 1)
        c_f = c_f + self.p_sino_pos_embed(c_f, h, w)
        out = self.att_layer(c_f).permute(0, 2, 1).reshape(b, c, h, w)

        if p_f is not None:
            p_f = F.interpolate(p_f, (h, w), mode='bilinear', align_corners=True)
            mask = torch.sigmoid(p_f)
            # TODO: remove hard mode
            # mask[mask > self.fusion_thresh] = 1
            # mask[mask <= self.fusion_thresh] = 1e-5
            
            if self.fusion_mode == 'mul':
                out = out * mask 
            elif self.fusion_mode == 'add':
                out = out + mask 

        return out


def replace_strides_with_dilation(module, dilation_rate, stride=1):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (stride, stride)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()


@BASE_MODELS.register_module()
class TransForensics(nn.Module):
    """
    Unofficial implementation of TransForensics. 
    Paper: TransForensics: Image Forgery Localization with Dense Self-Attention
    """
    def __init__(self, 
                backbone, 
                f_dims=[256, 512, 1024, 2048],
                dim=256, 
                mlp_dim=2048,
                depth=6,
                heads=8,
                dropout=0.1,
                classifier_cfg=None,
                seg_cfg=None,
                out_layer=3,
                fusion_mode='mul', 
                # fusion_thresh=0.5
                bayar_conv_cfg=None,
                base_fusion_mode = 'element_mul',
                baseline_dialate=1,
                dialate_stages=None,
                upsamplings=[4, 8, 16, 32],
                dialate_strides=None,
                ):
        super().__init__()

        if 'type' in backbone and 'PVT' in backbone.type:
            self.backbone = build_backbone(backbone)

            if baseline_dialate > 1 and dialate_stages:
                for k, r in enumerate(dialate_stages):
                    replace_strides_with_dilation(getattr(self.backbone, f'patch_embed{r}'), baseline_dialate,
                                                stride=dialate_strides[k])
        else:
            self.backbone = get_encoder(**backbone)

        self.bayar_pre_conv = None
        if bayar_conv_cfg is not None:
            self.bayar_pre_conv = BayarConv2d(**bayar_conv_cfg)

        self.att_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(4):
            if i >= out_layer:
                self.att_layers.append(AttentionDecoder(f_dims[i], dim, depth, heads, 
                                                        mlp_dim, dropout = dropout,
                                                        fusion_mode=fusion_mode, 
                                                        # fusion_thresh=fusion_thresh,
                                                        ))

                self.cls_layers.append(ClassificationHead(dim, **classifier_cfg))

                # if baseline_dialate > 1 and str(i+1) in dialate_stages:
                #     upsampling = upsamplings[i]
                # else:
                #     upsampling = 2**(i+2)
                upsampling = upsamplings[i]
                self.seg_layers.append(SegmentationHead(dim,
                                                    upsampling=upsampling, 
                                                    **seg_cfg))
            else:
                self.att_layers.append(None)
                self.cls_layers.append(None)
                self.seg_layers.append(None)

        self.out_layer = out_layer
        self.base_fusion_mode = base_fusion_mode

    def forward(self, x):
        inputs = [x, ]
        
        if self.bayar_pre_conv is not None:
            bayar_features = self.bayar_pre_conv(x)
            inputs.append(bayar_features)
        
        if len(inputs) > 1:
            if self.base_fusion_mode == 'concat':
                inputs = torch.cat(inputs, dim=1)
            # elif self.base_fusion_mode == 'bilinear_pool':
            #     B, N, H, W  = inputs[0].shape
            #     x = inputs[0].permute(0, 2, 3, 1).reshape(-1, N)
            #     y = inputs[1].permute(0, 2, 3, 1).reshape(-1, N)
            #     inputs = self.compact_bilinear_pool(x, y)
            #     inputs = inputs.reshape(B, H, W, N).permute(0, 3, 1, 2)
            elif self.base_fusion_mode == 'element_add':
                inputs = reduce(operator.add, inputs)
            elif self.base_fusion_mode == 'element_mul':
                inputs = reduce(operator.mul, inputs)
        else:
            inputs = inputs[0]

        features = self.backbone(inputs)

        p_f = None
        outs = []
        for i in range(3, -1, -1):
            if len(features) == 5:
                idx = i + 2
            else:
                idx = i
            c_f = features[idx]
            # c_f = features[i]
            out = self.att_layers[i](c_f, p_f)
            outs.append((self.cls_layers[i](out), self.seg_layers[i](out)))
            # if not self.training and idx == self.out_layer:
            if idx == self.out_layer:
                break
            p_f = out

        outs.reverse()

        return outs


@BASE_MODELS.register_module()
class TransForensics_v2(nn.Module):
    """
    Unofficial implementation of TransForensics. 
    Paper: TransForensics: Image Forgery Localization with Dense Self-Attention
    """
    def __init__(self, 
                g_backbone, 
                l_backbone,
                f_dims=[64, 128, 320, 512],
                l_f_dims=[256, 512, 1024, 2048],
                dim=256, 
                mlp_dim=2048,
                depth=6,
                heads=8,
                dropout=0.1,
                classifier_cfg=None,
                seg_cfg=None,
                out_layer=3,
                fusion_mode='mul', 
                # fusion_thresh=0.5
                # bayar_conv_cfg=None,
                base_fusion_mode = 'element_mul',
                ):
        super().__init__()

        if 'type' in g_backbone and 'PVT' in g_backbone.type:
            self.g_backbone = build_backbone(g_backbone)
        else:
            self.g_backbone = get_encoder(**g_backbone)

        self.l_backbone = get_encoder(**l_backbone)

        # self.bayar_pre_conv = None
        # if bayar_conv_cfg is not None:
        #     self.bayar_pre_conv = BayarConv2d(**bayar_conv_cfg)

        self.att_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        for i in range(4):
            if i >= out_layer:
                self.fusion_layers.append(nn.Conv2d(f_dims[i]+l_f_dims[i], f_dims[i], 1))
                self.att_layers.append(AttentionDecoder(f_dims[i], dim, depth, heads, 
                                                        mlp_dim, dropout = dropout,
                                                        fusion_mode=fusion_mode, 
                                                        # fusion_thresh=fusion_thresh,
                                                        ))
                

                self.cls_layers.append(ClassificationHead(dim, **classifier_cfg))
                self.seg_layers.append(SegmentationHead(dim,
                                                    upsampling=2**(i+2), 
                                                    **seg_cfg))
            else:
                self.fusion_layers.append(None)
                self.att_layers.append(None)
                self.cls_layers.append(None)
                self.seg_layers.append(None)

        self.out_layer = out_layer
        self.base_fusion_mode = base_fusion_mode

    def forward(self, x):
        # inputs = [x, ]
        
        # if self.bayar_pre_conv is not None:
        #     bayar_features = self.bayar_pre_conv(x)
        #     inputs.append(bayar_features)
        
        # if len(inputs) > 1:
        #     if self.base_fusion_mode == 'concat':
        #         inputs = torch.cat(inputs, dim=1)
        #     # elif self.base_fusion_mode == 'bilinear_pool':
        #     #     B, N, H, W  = inputs[0].shape
        #     #     x = inputs[0].permute(0, 2, 3, 1).reshape(-1, N)
        #     #     y = inputs[1].permute(0, 2, 3, 1).reshape(-1, N)
        #     #     inputs = self.compact_bilinear_pool(x, y)
        #     #     inputs = inputs.reshape(B, H, W, N).permute(0, 3, 1, 2)
        #     elif self.base_fusion_mode == 'element_add':
        #         inputs = reduce(operator.add, inputs)
        #     elif self.base_fusion_mode == 'element_mul':
        #         inputs = reduce(operator.mul, inputs)
        # else:
        #     inputs = inputs[0]

        g_features = self.g_backbone(x)
        l_features = self.l_backbone(x)
        

        p_f = None
        outs = []
        for i in range(3, -1, -1):
            if len(g_features) == 5:
                g_idx = i + 2
            else:
                g_idx = i
            g_f = g_features[g_idx]

            if len(l_features) == 6:
                l_idx = i + 2
            else:
                l_idx = i
            l_f = l_features[l_idx]
            f_f = torch.concat([g_f, l_f], dim=1)
            c_f = self.fusion_layers[i](f_f)

            out = self.att_layers[i](c_f, p_f)
            outs.append((self.cls_layers[i](out), self.seg_layers[i](out)))
            # if not self.training and idx == self.out_layer:
            if g_idx == self.out_layer:
                break
            p_f = out

        outs.reverse()

        return outs


@BASE_MODELS.register_module()
class TransForensics_v3(nn.Module):
    """
    Unofficial implementation of TransForensics. 
    Paper: TransForensics: Image Forgery Localization with Dense Self-Attention
    """
    def __init__(self, 
                g_backbone, 
                l_backbone,
                f_dims=[64, 128, 320, 512],
                l_f_dims=[256, 512, 1024, 2048],
                dim=256, 
                mlp_dim=2048,
                depth=6,
                heads=8,
                dropout=0.1,
                classifier_cfg=None,
                seg_cfg=None,
                out_layer=3,
                fusion_mode='mul', 
                # fusion_thresh=0.5
                # bayar_conv_cfg=None,
                base_fusion_mode = 'element_mul',
                ):
        super().__init__()

        if 'type' in g_backbone and 'PVT' in g_backbone.type:
            self.g_backbone = build_backbone(g_backbone)
        else:
            self.g_backbone = get_encoder(**g_backbone)

        self.l_backbone = get_encoder(**l_backbone)

        # self.bayar_pre_conv = None
        # if bayar_conv_cfg is not None:
        #     self.bayar_pre_conv = BayarConv2d(**bayar_conv_cfg)

        self.att_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        for i in range(4):
            if i >= out_layer:
                self.fusion_layers.append(nn.Conv2d(dim+l_f_dims[i], dim, 1))
                self.att_layers.append(AttentionDecoder(f_dims[i], dim, depth, heads, 
                                                        mlp_dim, dropout = dropout,
                                                        fusion_mode=fusion_mode, 
                                                        # fusion_thresh=fusion_thresh,
                                                        ))
                

                self.cls_layers.append(ClassificationHead(dim, **classifier_cfg))
                self.seg_layers.append(SegmentationHead(dim,
                                                    upsampling=2**(i+2), 
                                                    **seg_cfg))
            else:
                self.fusion_layers.append(None)
                self.att_layers.append(None)
                self.cls_layers.append(None)
                self.seg_layers.append(None)

        self.out_layer = out_layer
        self.base_fusion_mode = base_fusion_mode

    def forward(self, x):
        # inputs = [x, ]
        
        # if self.bayar_pre_conv is not None:
        #     bayar_features = self.bayar_pre_conv(x)
        #     inputs.append(bayar_features)
        
        # if len(inputs) > 1:
        #     if self.base_fusion_mode == 'concat':
        #         inputs = torch.cat(inputs, dim=1)
        #     # elif self.base_fusion_mode == 'bilinear_pool':
        #     #     B, N, H, W  = inputs[0].shape
        #     #     x = inputs[0].permute(0, 2, 3, 1).reshape(-1, N)
        #     #     y = inputs[1].permute(0, 2, 3, 1).reshape(-1, N)
        #     #     inputs = self.compact_bilinear_pool(x, y)
        #     #     inputs = inputs.reshape(B, H, W, N).permute(0, 3, 1, 2)
        #     elif self.base_fusion_mode == 'element_add':
        #         inputs = reduce(operator.add, inputs)
        #     elif self.base_fusion_mode == 'element_mul':
        #         inputs = reduce(operator.mul, inputs)
        # else:
        #     inputs = inputs[0]

        g_features = self.g_backbone(x)
        l_features = self.l_backbone(x)
        
        p_f = None
        outs = []
        for i in range(3, -1, -1):
            if len(g_features) == 5:
                g_idx = i + 2
            else:
                g_idx = i
            g_f = g_features[g_idx]

            if len(l_features) == 6:
                l_idx = i + 2
            else:
                l_idx = i
            l_f = l_features[l_idx]

            g_f = self.att_layers[i](g_f, p_f)
            final_f = torch.concat([g_f, l_f], dim=1)
            final_f = self.fusion_layers[i](final_f)
            outs.append((self.cls_layers[i](final_f), self.seg_layers[i](final_f)))
            if g_idx == self.out_layer:
                break
            p_f = g_f

        outs.reverse()

        return outs


class RANSegHead(nn.Module):
    """
    Paper: Semantic Segmentation with Reverse Attention
    """
    def __init__(self, in_channels, out_channels=1, kernel_size=3, upsampling=1, 
                uniform_range=4, delta=0.125, mode='norm'):
        super().__init__()
        self.uniform_range = uniform_range
        self.delta = delta
        self.upsampling = upsampling
        self.mode = mode

        self.org_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.rev_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        org_pred = self.org_conv2d(x)
        rev_pred = self.rev_conv2d(x)

        if self.mode == 'norm':
            attn_mask = torch.sigmoid(torch.pow(F.relu(org_pred)+self.delta, -1)-self.uniform_range)
        elif self.mode == 'simple':
            attn_mask = torch.sigmoid(-org_pred)

        attn_mask = attn_mask * rev_pred
        comb_pred = org_pred - attn_mask
        comb_pred = F.interpolate(comb_pred, scale_factor=self.upsampling,
                                mode='bilinear', align_corners=True)

        if not self.training:
            return comb_pred
        
        rev_pred = F.interpolate(-rev_pred, scale_factor=self.upsampling,
                                mode='bilinear', align_corners=True)
        org_pred = F.interpolate(org_pred, scale_factor=self.upsampling,
                                mode='bilinear', align_corners=True)

        return comb_pred, org_pred, rev_pred


@BASE_MODELS.register_module()
class TransForensics_v4(nn.Module):
    """
    Unofficial implementation of TransForensics. 
    Paper: TransForensics: Image Forgery Localization with Dense Self-Attention
    """
    def __init__(self, 
                backbone, 
                f_dims=[256, 512, 1024, 2048],
                dim=256, 
                mlp_dim=2048,
                depth=6,
                heads=8,
                dropout=0.1,
                classifier_cfg=None,
                seg_cfg=None,
                out_layer=3,
                fusion_mode='mul', 
                # fusion_thresh=0.5
                bayar_conv_cfg=None,
                base_fusion_mode = 'element_mul',
                baseline_dialate=1,
                dialate_stages=None,
                upsamplings=[4, 8, 16, 32],
                dialate_strides=None,
                ):
        super().__init__()

        if 'type' in backbone and 'PVT' in backbone.type:
            self.backbone = build_backbone(backbone)

            if baseline_dialate > 1 and dialate_stages:
                for k, r in enumerate(dialate_stages):
                    replace_strides_with_dilation(getattr(self.backbone, f'patch_embed{r}'), baseline_dialate,
                                                stride=dialate_strides[k])
        else:
            self.backbone = get_encoder(**backbone)

        self.bayar_pre_conv = None
        if bayar_conv_cfg is not None:
            self.bayar_pre_conv = BayarConv2d(**bayar_conv_cfg)

        self.att_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(4):
            if i >= out_layer:
                self.att_layers.append(AttentionDecoder(f_dims[i], dim, depth, heads, 
                                                        mlp_dim, dropout = dropout,
                                                        fusion_mode=fusion_mode, 
                                                        # fusion_thresh=fusion_thresh,
                                                        ))

                self.cls_layers.append(ClassificationHead(dim, **classifier_cfg))

                upsampling = upsamplings[i]
                self.seg_layers.append(RANSegHead(dim,
                                                upsampling=upsampling, 
                                                **seg_cfg))                                                
            else:
                self.att_layers.append(None)
                self.cls_layers.append(None)
                self.seg_layers.append(None)

        self.out_layer = out_layer
        self.base_fusion_mode = base_fusion_mode

    def forward(self, x):
        inputs = [x, ]
        
        if self.bayar_pre_conv is not None:
            bayar_features = self.bayar_pre_conv(x)
            inputs.append(bayar_features)
        
        if len(inputs) > 1:
            if self.base_fusion_mode == 'concat':
                inputs = torch.cat(inputs, dim=1)
            # elif self.base_fusion_mode == 'bilinear_pool':
            #     B, N, H, W  = inputs[0].shape
            #     x = inputs[0].permute(0, 2, 3, 1).reshape(-1, N)
            #     y = inputs[1].permute(0, 2, 3, 1).reshape(-1, N)
            #     inputs = self.compact_bilinear_pool(x, y)
            #     inputs = inputs.reshape(B, H, W, N).permute(0, 3, 1, 2)
            elif self.base_fusion_mode == 'element_add':
                inputs = reduce(operator.add, inputs)
            elif self.base_fusion_mode == 'element_mul':
                inputs = reduce(operator.mul, inputs)
        else:
            inputs = inputs[0]

        features = self.backbone(inputs)

        p_f = None
        outs = []
        for i in range(3, -1, -1):
            if len(features) == 5:
                idx = i + 2
            else:
                idx = i
            c_f = features[idx]
            out = self.att_layers[i](c_f, p_f)
            outs.append((self.cls_layers[i](out), self.seg_layers[i](out)))
            # if not self.training and idx == self.out_layer:
            if idx == self.out_layer:
                break
            p_f = out

        outs.reverse()

        return outs


@BASE_MODELS.register_module()
class TransForensics_v5(nn.Module):
    """
    Unofficial implementation of TransForensics. 
    Paper: TransForensics: Image Forgery Localization with Dense Self-Attention
    """
    def __init__(self, 
                backbone, 
                f_dims=[256, 512, 1024, 2048],
                dim=256, 
                mlp_dim=2048,
                depth=6,
                heads=8,
                dropout=0.1,
                classifier_cfg=None,
                seg_cfg=None,
                out_layer=3,
                fusion_mode='mul', 
                # fusion_thresh=0.5
                bayar_conv_cfg=None,
                base_fusion_mode = 'element_mul',
                baseline_dialate=1,
                dialate_stages=None,
                upsamplings=[4, 8, 16, 32],
                dialate_strides=None,
                ):
        super().__init__()

        if 'type' in backbone and 'PVT' in backbone.type:
            self.backbone = build_backbone(backbone)

            if baseline_dialate > 1 and dialate_stages:
                for k, r in enumerate(dialate_stages):
                    replace_strides_with_dilation(getattr(self.backbone, f'patch_embed{r}'), baseline_dialate,
                                                stride=dialate_strides[k])
        else:
            self.backbone = get_encoder(**backbone)

        self.bayar_pre_conv = None
        if bayar_conv_cfg is not None:
            self.bayar_pre_conv = BayarConv2d(**bayar_conv_cfg)

        self.att_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(4):
            if i >= out_layer:
                self.att_layers.append(AttentionDecoder(f_dims[i], dim, depth, heads, 
                                                        mlp_dim, dropout = dropout,
                                                        fusion_mode=fusion_mode, 
                                                        # fusion_thresh=fusion_thresh,
                                                        ))

                self.cls_layers.append(ClassificationHead(dim, **classifier_cfg))

                # if baseline_dialate > 1 and str(i+1) in dialate_stages:
                #     upsampling = upsamplings[i]
                # else:
                #     upsampling = 2**(i+2)
                upsampling = upsamplings[i]
                self.seg_layers.append(SegmentationHead(dim,
                                                    upsampling=upsampling, 
                                                    **seg_cfg))
            else:
                self.att_layers.append(None)
                self.cls_layers.append(None)
                self.seg_layers.append(None)

        self.out_layer = out_layer
        self.base_fusion_mode = base_fusion_mode

    def forward(self, x):
        inputs = [x, ]
        
        if self.bayar_pre_conv is not None:
            bayar_features = self.bayar_pre_conv(x)
            inputs.append(bayar_features)
        
        if len(inputs) > 1:
            if self.base_fusion_mode == 'concat':
                inputs = torch.cat(inputs, dim=1)
            # elif self.base_fusion_mode == 'bilinear_pool':
            #     B, N, H, W  = inputs[0].shape
            #     x = inputs[0].permute(0, 2, 3, 1).reshape(-1, N)
            #     y = inputs[1].permute(0, 2, 3, 1).reshape(-1, N)
            #     inputs = self.compact_bilinear_pool(x, y)
            #     inputs = inputs.reshape(B, H, W, N).permute(0, 3, 1, 2)
            elif self.base_fusion_mode == 'element_add':
                inputs = reduce(operator.add, inputs)
            elif self.base_fusion_mode == 'element_mul':
                inputs = reduce(operator.mul, inputs)
        else:
            inputs = inputs[0]

        features = self.backbone(inputs)

        p_f = None
        outs = []
        for i in range(3, -1, -1):
            if len(features) == 5:
                idx = i + 2
            else:
                idx = i
            c_f = features[idx]
            # c_f = features[i]
            out = self.att_layers[i](c_f, p_f)
            outs.append((self.cls_layers[i](out), self.seg_layers[i](out)))
            # if not self.training and idx == self.out_layer:
            if idx == self.out_layer:
                break
            p_f = out

        outs.reverse()

        return outs


@BASE_MODELS.register_module()
class TransForensicsDCT(nn.Module):
    """
    TransForensics with DCT branch in CAT-Net.
    """
    def __init__(self, 
                backbone, 
                f_dims=[256, 512, 1024, 2048],
                dim=256, 
                mlp_dim=2048,
                depth=6,
                heads=8,
                dropout=0.1,
                classifier_cfg=None,
                seg_cfg=None,
                out_layer=3,
                fusion_mode='mul', 
                dct_configs=None,
                baseline_dialate=1,
                dialate_stages=None,
                upsamplings=[4, 8, 16, 32],
                dialate_strides=None,
                ):
        super().__init__()

        if 'type' in backbone and 'PVT' in backbone.type:
            self.rgb_stream = build_backbone(backbone)

            if baseline_dialate > 1 and dialate_stages:
                for k, r in enumerate(dialate_stages):
                    replace_strides_with_dilation(getattr(self.rgb_stream, f'patch_embed{r}'), baseline_dialate,
                                                stride=dialate_strides[k])
        else:
            self.rgb_stream = get_encoder(**backbone)
        self.dct_stream = DCTStream(dct_configs, dct_configs.dct_pretrained)

        # self.stage5_cfg = dct_configs['STAGE5']
        # num_channels = self.stage5_cfg['NUM_CHANNELS']
        # block = blocks_dict[self.stage5_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))]
        # self.transition4 = self._make_transition_layer(
        #     [i+j for (i, j) in zip(self.rgb_stream.out_channels, self.dct_stream.dc_final_stage_channels)], num_channels)

        self.att_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(4):
            if i >= out_layer:
                self.att_layers.append(AttentionDecoder(f_dims[i], dim, depth, heads, 
                                                        mlp_dim, dropout=dropout,
                                                        fusion_mode=fusion_mode, 
                                                        ))

                self.cls_layers.append(ClassificationHead(dim, **classifier_cfg))

                # if baseline_dialate > 1 and str(i+1) in dialate_stages:
                #     upsampling = upsamplings[i]
                # else:
                #     upsampling = 2**(i+2)
                upsampling = upsamplings[i]
                self.seg_layers.append(SegmentationHead(dim,
                                                    upsampling=upsampling, 
                                                    **seg_cfg))
            else:
                self.att_layers.append(None)
                self.cls_layers.append(None)
                self.seg_layers.append(None)

        self.out_layer = out_layer

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def forward(self, img, dcT_vol, qtable):
        rgb_feats = self.rgb_stream(img)
        dct_feats = self.dct_stream(dcT_vol, qtable)

        x_list = [torch.cat([rgb_feats[i+1], dct_feats[i]], 1) for i in range(3)]
        x_list.insert(0, rgb_feats[0])
        # for i in range(self.stage5_cfg['NUM_BRANCHES']):
        #     if self.transition4[i] is not None:
        #         x_list.append(self.transition4[i](x[i]))
        #     else:
        #         x_list.append(x[i])

        p_f = None
        outs = []
        for i in range(3, -1, -1):
            if len(x_list) == 5:
                idx = i + 2
            else:
                idx = i
            c_f = x_list[idx]
            # c_f = features[i]
            out = self.att_layers[i](c_f, p_f)
            outs.append((self.cls_layers[i](out), self.seg_layers[i](out)))
            # if not self.training and idx == self.out_layer:
            if idx == self.out_layer:
                break
            p_f = out

        outs.reverse()

        return outs

