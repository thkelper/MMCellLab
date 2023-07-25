import copy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
from .builder import BASE_MODELS
from .efficientnet.model import EfficientNet
from .init import init_weights


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
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

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


class MLP(nn.Module):
    def __init__(self, dim, drop=0.0, bias=True):
        super().__init__()
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.fc1 = nn.Linear(dim, dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        o = self.drop1(self.act(self.fc1(x)))
        o = x + self.drop2(self.fc2(o))
        return o


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module): 
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b_q, sz_b_kv, len_q, len_k, len_v = q.size(0), k.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b_q, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b_kv, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b_kv, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        out, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(sz_b_kv, len_q, -1)
        
        out = self.dropout(self.fc(out))
        out += residual

        out = self.layer_norm(out)

        return out, attn


class ObjectEncoder(nn.Module):
    def __init__(self, objects_N, dim, att_dropout=0.0, drop=0.0, heads=1, qkv_bias=False):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        # self.layer_norm2 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(heads, 
                                            dim, 
                                            dim,
                                            dim,
                                            dropout=att_dropout)
        # self.dropout1 = nn.Dropout(att_dropout)
        self.dropout2 = nn.Dropout(drop)
        self.W_c = nn.Linear(objects_N, objects_N, bias=True)
        self.mlp = MLP(dim=dim, drop=drop, bias=True)

        self.apply(self._init_weights)

    def forward(self, o_i, p_i):
        o_i = q = self.layer_norm1(o_i)
        # p_i = k = v = self.layer_norm2(p_i)
        k = v = p_i
        atten = self.self_attn(q, k, v)[0]
        # o_i_1 = o_i + self.dropout1(atten)
        o_i_1 = o_i + atten
        o_i_1 = o_i_1 + self.dropout2(self.W_c(o_i_1.transpose(2, 1)).transpose(2, 1))
        o_i_1 = self.mlp(o_i_1)

        return o_i_1
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class PatchDecoder(nn.Module):
    def __init__(self, dim, att_dropout=0.0, drop=0.0, heads=1, qkv_bias=False, sim_mask=None,
                bcim_mode='add'):
        super().__init__()
        # self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(heads, 
                                            dim, 
                                            dim,
                                            dim,
                                            dropout=att_dropout)
        # self.dropout1 = nn.Dropout(att_dropout)
        self.mlp = MLP(dim, drop=drop, bias=True)
        self.sim_mask = sim_mask

        assert bcim_mode in ('add', 'mul')
        self.bcim_mode = bcim_mode

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, o_i_1, p_i, h, w):
        # attention
        # p_i = q = self.layer_norm1(p_i)
        q = p_i
        k = v = self.layer_norm2(o_i_1)
        attn = self.self_attn(q, k, v)[0]
        p_i_1 = p_i + attn
        # p_i_1 = p_i + self.dropout1(attn)
        p_i_1 = p_i_1 + self.mlp(p_i_1)

        # BCIM (Boundary-sensitive Contextual Incoherence Modeling)
        # if self.sim_mask is not None:
        #     B, N, C = p_i_1.shape
        #     # n=h*w
        #     n = N // 2
        #     sim_mask = self.sim_mask[0:h, 0:w, 0:h, 0:w].reshape(n, n)
        #     # p_i_1 = p_i_1.reshape(B, n, 2*C)
        #     p_i_1 = torch.cat([p_i_1[:, 0:n, :], p_i_1[:, n:, :]], dim=-1)
        #     normed_p_i_1 = torch.nn.functional.normalize(p_i_1, p=2, dim=-1)
        #     # (b,n,n)
        #     cos_sim = normed_p_i_1 @ normed_p_i_1.permute(0, 2, 1)
        #     # (b,n), mask the cos_sim outside k*k window
        #     cos_sim_sum = (cos_sim * sim_mask).sum(dim=-1)
        #     sim = cos_sim_sum / sim_mask.sum(dim=1)
        #     # element-wise addition
        #     # TODO: maybe multiplication better
        #     p_i_1 = p_i_1 + sim[:, :, None]
        #     # p_i_1 = p_i_1.reshape(B, 2*n, C)
        #     p_i_1 = p_i_1.reshape(B, n, 2, C).permute(0, 2, 1, 3).reshape(B, 2*n, C)

        if self.sim_mask is not None:
            B, N, C = p_i_1.shape
            # N=h*w
            sim_mask = self.sim_mask[0:h, 0:w, 0:h, 0:w].reshape(N, N)
            normed_p_i_1 = torch.nn.functional.normalize(p_i_1, p=2, dim=-1)
            # (B,N,N)
            cos_sim = normed_p_i_1 @ normed_p_i_1.permute(0, 2, 1)
            # (B, N), mask the cos_sim outside of the k*k window
            cos_sim_sum = (cos_sim * sim_mask).sum(dim=-1)
            sim = cos_sim_sum / sim_mask.sum(dim=1)
            # TODO: element-wise addition in paper, maybe multiplication is better
            if self.bcim_mode == 'add':
                p_i_1 = p_i_1 + sim[:, :, None]
            elif self.bcim_mode == 'mul':
                p_i_1 = p_i_1 * sim[:, :, None]
        
        return p_i_1


class AttentionLayer(nn.Module):
    def __init__(self, objects_N, dim, att_dropout=0.0, drop=0.0, heads=1, qkv_bias=False,
                sim_mask=None, bcim_mode='add'):
        super().__init__()
        self.p_i_layer_norm = nn.LayerNorm(dim)
        self.object_encoder = ObjectEncoder(objects_N, 
                                            dim, 
                                            att_dropout=att_dropout, 
                                            drop=drop,
                                            heads=heads, 
                                            qkv_bias=qkv_bias)
        self.patch_decoder = PatchDecoder(dim, 
                                          att_dropout=att_dropout, 
                                          drop=drop, 
                                          heads=heads, 
                                          qkv_bias=qkv_bias, 
                                          sim_mask=sim_mask,
                                          bcim_mode=bcim_mode)

    def forward(self, o_i, p_i, h, w):
        normed_p_i = self.p_i_layer_norm(p_i)
        o_i_1 = self.object_encoder(o_i, normed_p_i)
        p_i_1 = self.patch_decoder(o_i_1, normed_p_i, h, w)
        return o_i_1, p_i_1


@BASE_MODELS.register_module()
class ObjectFormer(nn.Module):
    """
    Unofficial implementation of ObjectFormer. 
    Paper: ObjectFormer for Image Manipulation Detection and Localization.
    """
    def __init__(self, 
                rgb_backbone={'name': 'efficientnet-b4', 'pretrained':True}, 
                objects_N=16,
                high_freq_extr_channels=[24, 48, 96, 192, 320],
                embedding_channels=640, 
                rgb_embed_extractor=False,
                num_att_layers=8,
                att_dropout=0.0,
                drop=0.0,
                heads=1,
                qkv_bias=False,
                bcim=True,
                win_size=3,
                max_mask_size=60,
                bcim_mode='add',
                classifier_cfg=None,
                seg_cfg=None,
                freq_backbone=None,
                custom_init=False,
                channel_fusion=False,
                o_pos_embedding=False,
                ):
        super().__init__()
        assert 'efficientnet-b' in rgb_backbone.name
        assert embedding_channels % 8 == 0
        feature_channels = embedding_channels // 2
        
        # rgb_backbone = copy.deepcopy(rgb_backbone)
        # self.rgb_backbone = getattr(tv_models, name)(**rgb_backbone)
        self.rgb_backbone = EfficientNet.from_pretrained(rgb_backbone['name'])
        if rgb_embed_extractor:
            self.rgb_embed_extractor = nn.Identity()
        else:
            # self.rgb_embed_extractor = nn.Identity()
            self.rgb_embed_extractor = nn.Conv2d(self.rgb_backbone.out_channels, 
                                                feature_channels, 
                                                1, 
                                                stride=1, 
                                                bias=True)
        self.freq_embed_extractor = nn.Identity()
        if freq_backbone.name == 'default':
            self.freq_extractors = nn.ModuleList()
            in_ch = 3
            for out_ch in high_freq_extr_channels:
                self.freq_extractors.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU()
                    )
                )
                in_ch = out_ch
            if out_ch != feature_channels:
                self.freq_embed_extractor = nn.Conv2d(in_ch, feature_channels, 1, stride=1, bias=True)
        elif 'efficientnet-b' in freq_backbone.name:
            self.freq_extractors = EfficientNet.from_pretrained(freq_backbone.name)
            self.freq_embed_extractor = nn.Conv2d(self.rgb_backbone.out_channels, 
                                                feature_channels, 
                                                1, 
                                                stride=1, 
                                                bias=True)

        if channel_fusion:
            self.channel_fusion = nn.Linear(embedding_channels, embedding_channels, bias=True)
        else:
            self.channel_fusion = nn.Identity()

        self.p_sino_pos_embed = build_position_encoding(embedding_channels)
        
        self.o_0 = nn.Parameter(torch.zeros(1, objects_N, embedding_channels))
        nn.init.trunc_normal_(self.o_0, std=.02)

        self.o_sino_pos_embed = None
        if o_pos_embedding:
            self.o_sino_pos_embed = build_position_encoding(embedding_channels, 
                                                            position_embedding='seq')

        self.sim_mask = None
        if bcim:
            self.sim_mask = self._gen_bcim_mask(win_size, max_mask_size)

        self.att_layers = nn.ModuleList()
        for _ in range(num_att_layers):
            self.att_layers.append(AttentionLayer(
                                        objects_N, 
                                        dim=embedding_channels, 
                                        att_dropout=att_dropout, 
                                        drop=drop, 
                                        heads=heads, 
                                        qkv_bias=qkv_bias,
                                        sim_mask=self.sim_mask,
                                        bcim_mode=bcim_mode))

        self.clssifier_head = ClassificationHead(embedding_channels, **classifier_cfg)
        self.segment_head = SegmentationHead(embedding_channels,
                                            upsampling=2**len(high_freq_extr_channels), 
                                            **seg_cfg)

        if custom_init:
            self.init_parmas()

    def init_parmas(self):
        for name, m in self.named_modules():
            if 'rgb_backbone' in name or 'freq_extractors' in name:
                continue
            init_weights(m)

    def _gen_bcim_mask(self, win_size, max_size):
        mask = nn.Parameter(torch.zeros(max_size, max_size, max_size, max_size, dtype=torch.float32), 
                            requires_grad=False)
        for i in range(max_size):
            for j in range(max_size):
                mask[i, j,
                     max(i-win_size, 0):min(i+win_size+1, max_size),
                     max(j-win_size, 0):min(j+win_size+1, max_size)] = 1 
                    
        return mask


    def forward(self, rgb_img, freq_img):
        rgb_features = self.rgb_embed_extractor(self.rgb_backbone(rgb_img))
        B, C, h, w = rgb_features.shape
        rgb_embeddings = rgb_features.reshape(B, C, h*w).permute(0, 2, 1)

        freq_embeddings = freq_img
        if isinstance(self.freq_extractors, nn.ModuleList):
            for extractor in self.freq_extractors:
                freq_embeddings = extractor(freq_embeddings)
        else:
            freq_embeddings = self.freq_extractors(freq_embeddings)
        freq_embeddings = self.freq_embed_extractor(freq_embeddings)
        freq_embeddings = freq_embeddings.reshape(B, C, h*w).permute(0, 2, 1)

        # B, C, 2*h*w
        o = self.o_0
        if self.o_sino_pos_embed is not None:
            o = o + self.o_sino_pos_embed(o)
        # Dimension of concat is 1 in paper, maybe dimension 2 is better
        p = torch.cat([rgb_embeddings, freq_embeddings], dim=2)
        # TODO: Adding an 1x1 conv to fuse embeddings
        p = self.channel_fusion(p)

        # p = p + self.p_sino_pos_embed(p, h, w).reshape(B, h*w, 2*C)
        pos = self.p_sino_pos_embed(p, h, w)
        p = p + pos
        for att_layer in self.att_layers:
            o, p = att_layer(o, p, h, w)

        features = p.permute(0, 2, 1).reshape(B, -1, h, w)
        cls = self.clssifier_head(features)
        seg = self.segment_head(features)
        
        return cls, seg
