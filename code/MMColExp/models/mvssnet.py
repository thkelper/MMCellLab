import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
from mmcv.runner import BaseModule
from segmentation_models_pytorch.encoders._utils import replace_strides_with_dilation
from .init import init_weights
from .builder import BASE_MODELS
from .pvtv2 import PVTv2B2
from .deforgeformer import DeepLabV3PlusDecoder
from .init import init_weights
import segmentation_models_pytorch.base.initialization as init


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2)) 
    return torch.sigmoid(g) * input


def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)

    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=rate, dilation=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, n_input=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        rates = [1, 2, 4]
        self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)  # stride 2 => stride 1
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(pretrained=False, layers=[3,4,6,3], backbone='resnet50', n_input=3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, layers, n_input=n_input, **kwargs)

    # pretrain_dict = model_zoo.load_url(model_urls[backbone])
    # try:
    #     model.load_state_dict(pretrain_dict,strict=False)
    # except:
    #     print("loss conv1")
    #     model_dict = {}
    #     for k, v in pretrain_dict.items():
    #         if k in pretrain_dict and 'conv1' not in k:
    #             model_dict[k] = v
    #     model.load_state_dict(model_dict, strict=False)
    # print("load pretrain success")

    return model


# class ResNet50(BaseModule):
class ResNet50(nn.Module):
    def __init__(self, pretrained=True,n_input=3):
        """Declare all needed layers."""
        super(ResNet50, self).__init__()
        self.model = resnet(n_input=n_input, layers=[3, 4, 6, 3], backbone='resnet50')
        self.pretrained = pretrained
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def base_forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out

    # def init_weights(self):
    #     super(ResNet50, self).init_weights()
    #     # for m in self.modules():
    #     #     if isinstance(m, nn.Conv2d):
    #     #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     #     elif isinstance(m, nn.BatchNorm2d):
    #     #         nn.init.constant_(m.weight, 1)
    #     #         nn.init.constant_(m.bias, 0)

    #     if self.pretrained:
    #         pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
    #         try:
    #             self.model.load_state_dict(pretrain_dict, strict=False)
    #         except:
    #             print("loss conv1")
    #             model_dict = {}
    #             for k, v in pretrain_dict.items():
    #                 if k in pretrain_dict and 'conv1' not in k:
    #                     model_dict[k] = v
    #             self.model.load_state_dict(model_dict, strict=False)
    #         print("load pretrain success in {}".format(self.__class__.__name__))


class ERB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x+res


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)


@BASE_MODELS.register_module()
class MVSSNet(ResNet50):
    def __init__(self, nclass,
                 aux=False,
                 sobel=False,
                 constrain=False,
                 n_input=3,
                 pretrained_base=True,
                 **kwargs):
        super(MVSSNet, self).__init__(pretrained=pretrained_base, n_input=n_input)
        self.num_class = nclass
        self.aux = aux
        if aux:
            self.outputs = []
        self.__setattr__('exclusive', ['head'])

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.sobel = sobel
        self.constrain = constrain

        # self.erb_db_1 = ERB(256, self.num_class-1)
        # self.erb_db_2 = ERB(512, self.num_class-1)
        # self.erb_db_3 = ERB(1024, self.num_class-1)
        # self.erb_db_4 = ERB(2048, self.num_class-1)

        # self.erb_trans_1 = ERB(self.num_class-1, self.num_class-1)
        # self.erb_trans_2 = ERB(self.num_class-1, self.num_class-1)
        # self.erb_trans_3 = ERB(self.num_class-1, self.num_class-1)

        self.erb_db_1 = ERB(256, self.num_class)
        self.erb_db_2 = ERB(512, self.num_class)
        self.erb_db_3 = ERB(1024, self.num_class)
        self.erb_db_4 = ERB(2048, self.num_class)

        self.erb_trans_1 = ERB(self.num_class, self.num_class)
        self.erb_trans_2 = ERB(self.num_class, self.num_class)
        self.erb_trans_3 = ERB(self.num_class, self.num_class)

        if self.sobel:
            self.sobel_x1, self.sobel_y1 = get_sobel(256, 1)
            self.sobel_x2, self.sobel_y2 = get_sobel(512, 1)
            self.sobel_x3, self.sobel_y3 = get_sobel(1024, 1)
            self.sobel_x4, self.sobel_y4 = get_sobel(2048, 1)

        if self.constrain:
            self.noise_extractor = ResNet50(n_input=3, pretrained=pretrained_base)
            self.constrain_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)
            self.head = _DAHead(2048+2048, self.num_class, False, **kwargs)
        else:
            self.head = _DAHead(2048, self.num_class, False, **kwargs)

        self.init_parmas()

    def _load_pretrained(self, model):
        pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
        try:
            model.load_state_dict(pretrain_dict, strict=False)
        except:
            print("loss conv1")
            model_dict = {}
            for k, v in pretrain_dict.items():
                if k in pretrain_dict and 'conv1' not in k:
                    model_dict[k] = v
            model.load_state_dict(model_dict, strict=False)
        print("load pretrain success ")
        
    def init_parmas(self):
        for m in self.modules():
            init_weights(m)

        if self.pretrained:
            self._load_pretrained(self.model)
            self._load_pretrained(self.noise_extractor)

    def forward(self, x):
        size = x.size()[2:]
        input_ = x.clone()
        feature_map, _ = self.base_forward(input_)
        c1, c2, c3, c4 = feature_map

        if self.training or self.aux:
            # Edge-Supervised Branch
            if self.sobel:
                res1 = self.erb_db_1(run_sobel(self.sobel_x1, self.sobel_y1, c1))
                res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(run_sobel(self.sobel_x2, self.sobel_y2, c2))))
                res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(run_sobel(self.sobel_x3, self.sobel_y3, c3))))
                res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(run_sobel(self.sobel_x4, self.sobel_y4, c4))), relu=False)
            else:
                res1 = self.erb_db_1(c1)
                res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(c2)))
                res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(c3)))
                res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(c4)), relu=False)

        # Noise-Sensitive Branch
        if self.constrain:
            x = rgb2gray(x)
            noise = self.constrain_conv(x)
            constrain_features, _ = self.noise_extractor.base_forward(noise)
            constrain_feature = constrain_features[-1]
            c4 = torch.cat([c4, constrain_feature], dim=1)

        # Dual Attention
        x = self.head(c4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)

        # if self.aux:
        #     self.outputs = []
        #     self.outputs.append(F.interpolate(noise, size, mode='bilinear', align_corners=True))
        #     self.outputs.append(F.interpolate(res1, size, mode='bilinear', align_corners=True))
            # x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            # x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            # self.outputs.append(x1)
            # self.outputs.append(x2)

        if self.training:
            return res1, x0
        else:
            return x0


@BASE_MODELS.register_module()
class MVSSFormer(nn.Module):
    def __init__(self, nclass,
                 aux=False,
                 sobel=True,
                 constrain=True,
                 n_input=3,
                 pretrained=None):
        super().__init__()

        self.pretrained = pretrained
        self.model = PVTv2B2(in_chans=n_input)

        self.num_class = nclass
        self.aux = aux
        if aux:
            self.outputs = []
        self.__setattr__('exclusive', ['head'])

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.sobel = sobel
        self.constrain = constrain

        self.erb_db_1 = ERB(64, self.num_class)
        self.erb_db_2 = ERB(128, self.num_class)
        self.erb_db_3 = ERB(320, self.num_class)
        self.erb_db_4 = ERB(512, self.num_class)

        self.erb_trans_1 = ERB(self.num_class, self.num_class)
        self.erb_trans_2 = ERB(self.num_class, self.num_class)
        self.erb_trans_3 = ERB(self.num_class, self.num_class)

        if self.sobel:
            self.sobel_x1, self.sobel_y1 = get_sobel(64, 1)
            self.sobel_x2, self.sobel_y2 = get_sobel(128, 1)
            self.sobel_x3, self.sobel_y3 = get_sobel(320, 1)
            self.sobel_x4, self.sobel_y4 = get_sobel(512, 1)

        if self.constrain:
            self.noise_extractor = PVTv2B2(in_chans=3)
            self.constrain_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)
            self.head = _DAHead(512+512, self.num_class, False)
        else:
            self.head = _DAHead(512, self.num_class, False)

        self.init_parmas()

    def _load_pretrained(self, model):
        pretrain_dict = torch.load(self.pretrained)
        ret = model.load_state_dict(pretrain_dict, strict=False)
        print(ret)
        print("load pretrain success ")
        
    def init_parmas(self):
        for m in self.modules():
            init_weights(m)

        if self.pretrained:
            self._load_pretrained(self.model)
            self._load_pretrained(self.noise_extractor)

    def forward(self, x):
        size = x.size()[2:]
        # input_ = x.clone()
        input_ = x
        feature_map = self.model(input_)
        c1, c2, c3, c4 = feature_map

        if self.training or self.aux:
            # Edge-Supervised Branch
            if self.sobel:
                res1 = self.erb_db_1(run_sobel(self.sobel_x1, self.sobel_y1, c1))
                res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(run_sobel(self.sobel_x2, self.sobel_y2, c2))))
                res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(run_sobel(self.sobel_x3, self.sobel_y3, c3))))
                res1 = self.erb_trans_3(res1 + self.upsample_8(self.erb_db_4(run_sobel(self.sobel_x4, self.sobel_y4, c4))), relu=False)
            else:
                res1 = self.erb_db_1(c1)
                res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(c2)))
                res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(c3)))
                res1 = self.erb_trans_3(res1 + self.upsample_8(self.erb_db_4(c4)), relu=False)

        # Noise-Sensitive Branch
        if self.constrain:
            x = rgb2gray(x)
            noise = self.constrain_conv(x)
            constrain_feature = self.noise_extractor(noise)[-1]
            c4 = torch.cat([c4, constrain_feature], dim=1)

        # Dual Attention
        x = self.head(c4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)

        # if self.aux:
        #     self.outputs = []
        #     self.outputs.append(F.interpolate(noise, size, mode='bilinear', align_corners=True))
        #     self.outputs.append(F.interpolate(res1, size, mode='bilinear', align_corners=True))
            # x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            # x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            # self.outputs.append(x1)
            # self.outputs.append(x2)

        if self.training:
            return res1, x0
        else:
            return x0


@BASE_MODELS.register_module()
class MVSSFormer2(nn.Module):
    def __init__(self, nclass,
                 aux=False,
                 sobel=True,
                 constrain=True,
                 n_input=3,
                 rgb_decoder_cfg=None,
                 noise_decoder_cfg=None,
                 encoder_output_stride=16,
                 pretrained=None):
        super().__init__()

        self.pretrained = pretrained
        self.model = PVTv2B2(in_chans=n_input)

        self.rgb_decoder = None
        if rgb_decoder_cfg is not None:
            if encoder_output_stride == 16:
                replace_strides_with_dilation(self.model.patch_embed4, 2)
            elif encoder_output_stride == 8:
                replace_strides_with_dilation(self.model.patch_embed4, 2)
                replace_strides_with_dilation(self.model.patch_embed3, 2)

            self.rgb_decoder = DeepLabV3PlusDecoder(
                encoder_channels=[c for c in self.model.out_channels],
                output_stride=encoder_output_stride,
                **rgb_decoder_cfg,
            )

        self.num_class = nclass
        self.aux = aux
        if aux:
            self.outputs = []
        self.__setattr__('exclusive', ['head'])

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.sobel = sobel
        self.constrain = constrain

        self.erb_db_1 = ERB(64, self.num_class)
        self.erb_db_2 = ERB(128, self.num_class)
        self.erb_db_3 = ERB(320, self.num_class)
        self.erb_db_4 = ERB(512, self.num_class)

        self.erb_trans_1 = ERB(self.num_class, self.num_class)
        self.erb_trans_2 = ERB(self.num_class, self.num_class)
        self.erb_trans_3 = ERB(self.num_class, self.num_class)

        if self.sobel:
            self.sobel_x1, self.sobel_y1 = get_sobel(64, 1)
            self.sobel_x2, self.sobel_y2 = get_sobel(128, 1)
            self.sobel_x3, self.sobel_y3 = get_sobel(320, 1)
            self.sobel_x4, self.sobel_y4 = get_sobel(512, 1)

        if self.constrain:
            self.noise_extractor = PVTv2B2(in_chans=3)

            self.noise_decoder = None
            if noise_decoder_cfg is not None:
                if encoder_output_stride == 16:
                    replace_strides_with_dilation(self.noise_extractor.patch_embed4, 2)
                elif encoder_output_stride == 8:
                    replace_strides_with_dilation(self.noise_extractor.patch_embed4, 2)
                    replace_strides_with_dilation(self.noise_extractor.patch_embed3, 2)

                self.noise_decoder = DeepLabV3PlusDecoder(
                    encoder_channels=[c for c in self.noise_extractor.out_channels],
                    output_stride=encoder_output_stride,
                    **noise_decoder_cfg,
                )

            self.constrain_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)
            self.head = _DAHead(512, self.num_class, False)
        else:
            self.head = _DAHead(256, self.num_class, False)

        self.init_parmas()

    def _load_pretrained(self, model):
        pretrain_dict = torch.load(self.pretrained)
        ret = model.load_state_dict(pretrain_dict, strict=False)
        print(ret)
        print("load pretrain success ")
        
    def init_parmas(self):
        for m in self.modules():
            init_weights(m)

        if self.rgb_decoder is not None:
            init.initialize_decoder(self.rgb_decoder)

        if self.noise_decoder is not None:
            init.initialize_decoder(self.noise_decoder)

        if self.pretrained:
            self._load_pretrained(self.model)
            self._load_pretrained(self.noise_extractor)

    def forward(self, x):
        size = x.size()[2:]
        # input_ = x.clone()
        input_ = x
        feature_map = self.model(input_)
        c1, c2, c3, c4 = feature_map

        if self.training or self.aux:
            # Edge-Supervised Branch
            if self.sobel:
                res1 = self.erb_db_1(run_sobel(self.sobel_x1, self.sobel_y1, c1))
                res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(run_sobel(self.sobel_x2, self.sobel_y2, c2))))
                res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(run_sobel(self.sobel_x3, self.sobel_y3, c3))))
                res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(run_sobel(self.sobel_x4, self.sobel_y4, c4))), relu=False)
            else:
                res1 = self.erb_db_1(c1)
                res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(c2)))
                res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(c3)))
                res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(c4)), relu=False)

        if self.rgb_decoder is not None:
            c4 = self.rgb_decoder(*feature_map)

        # Noise-Sensitive Branch
        if self.constrain:
            x = rgb2gray(x)
            noise = self.constrain_conv(x)
            constrain_features = self.noise_extractor(noise)
            if self.noise_decoder:
                constrain_feature = self.noise_decoder(*constrain_features)
            else:
                constrain_feature = constrain_features[-1]
        
            c4 = torch.cat([c4, constrain_feature], dim=1)

        # Dual Attention
        x = self.head(c4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)

        if self.training:
            return res1, x0
        else:
            return x0


@BASE_MODELS.register_module()
class MVSSFormer3(nn.Module):
    def __init__(self, nclass,
                 aux=False,
                 sobel=True,
                 constrain=True,
                 n_input=3,
                 encoder_output_stride=16,
                 pretrained=None):
        super().__init__()

        self.pretrained = pretrained
        self.model = PVTv2B2(in_chans=n_input)

        if encoder_output_stride is not None:
            if encoder_output_stride == 16:
                replace_strides_with_dilation(self.model.patch_embed4, 2)
            elif encoder_output_stride == 8:
                replace_strides_with_dilation(self.model.patch_embed4, 2)
                replace_strides_with_dilation(self.model.patch_embed3, 2)

        self.num_class = nclass
        self.aux = aux
        if aux:
            self.outputs = []
        self.__setattr__('exclusive', ['head'])

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.sobel = sobel
        self.constrain = constrain

        self.erb_db_1 = ERB(64, self.num_class)
        self.erb_db_2 = ERB(128, self.num_class)
        self.erb_db_3 = ERB(320, self.num_class)
        self.erb_db_4 = ERB(512, self.num_class)

        self.erb_trans_1 = ERB(self.num_class, self.num_class)
        self.erb_trans_2 = ERB(self.num_class, self.num_class)
        self.erb_trans_3 = ERB(self.num_class, self.num_class)

        if self.sobel:
            self.sobel_x1, self.sobel_y1 = get_sobel(64, 1)
            self.sobel_x2, self.sobel_y2 = get_sobel(128, 1)
            self.sobel_x3, self.sobel_y3 = get_sobel(320, 1)
            self.sobel_x4, self.sobel_y4 = get_sobel(512, 1)

        if self.constrain:
            self.noise_extractor = PVTv2B2(in_chans=3)

            if encoder_output_stride is not None:
                if encoder_output_stride == 16:
                    replace_strides_with_dilation(self.noise_extractor.patch_embed4, 2)
                elif encoder_output_stride == 8:
                    replace_strides_with_dilation(self.noise_extractor.patch_embed4, 2)
                    replace_strides_with_dilation(self.noise_extractor.patch_embed3, 2)

            self.constrain_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)
            self.head = _DAHead(1024, self.num_class, False)
        else:
            self.head = _DAHead(512, self.num_class, False)

        self.init_parmas()

    def _load_pretrained(self, model):
        pretrain_dict = torch.load(self.pretrained)
        ret = model.load_state_dict(pretrain_dict, strict=False)
        print(ret)
        print("load pretrain success ")
        
    def init_parmas(self):
        for m in self.modules():
            init_weights(m)

        if self.pretrained:
            self._load_pretrained(self.model)
            self._load_pretrained(self.noise_extractor)

    def forward(self, x):
        size = x.size()[2:]
        # input_ = x.clone()
        input_ = x
        feature_map = self.model(input_)
        c1, c2, c3, c4 = feature_map

        if self.training or self.aux:
            # Edge-Supervised Branch
            if self.sobel:
                res1 = self.erb_db_1(run_sobel(self.sobel_x1, self.sobel_y1, c1))
                res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(run_sobel(self.sobel_x2, self.sobel_y2, c2))))
                res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(run_sobel(self.sobel_x3, self.sobel_y3, c3))))
                res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(run_sobel(self.sobel_x4, self.sobel_y4, c4))), relu=False)
            else:
                res1 = self.erb_db_1(c1)
                res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(c2)))
                res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(c3)))
                res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(c4)), relu=False)

        # Noise-Sensitive Branch
        if self.constrain:
            x = rgb2gray(x)
            noise = self.constrain_conv(x)
            constrain_feature = self.noise_extractor(noise)[-1]
        
            c4 = torch.cat([c4, constrain_feature], dim=1)

        # Dual Attention
        x = self.head(c4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)

        if self.training:
            return res1, x0
        else:
            return x0


