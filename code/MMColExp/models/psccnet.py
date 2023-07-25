import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead
from .seg_hrnet import get_hrnet_cfg, get_seg_model
from .builder import BASE_MODELS, build_backbone
from .seg_hrnet import BasicBlock, Bottleneck as Bottleneck_, HighResolutionModule


BN_MOMENTUM = 0.01
BatchNorm2d = nn.BatchNorm2d


class NonLocalMask(nn.Module):
    def __init__(self, in_channels, reduce_scale):
        super(NonLocalMask, self).__init__()

        self.r = reduce_scale

        # input channel number
        self.ic = in_channels * self.r * self.r

        # middle channel number
        self.mc = self.ic

        self.g = nn.Conv2d(in_channels=self.ic, out_channels=self.ic,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                             kernel_size=1, stride=1, padding=0)

        self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.W_c = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.gamma_s = nn.Parameter(torch.ones(1))

        self.gamma_c = nn.Parameter(torch.ones(1))

        self.getmask = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            value :
                f: B X (HxW) X (HxW)
                ic: intermediate channels
                z: feature maps( B X C X H X W)
            output:
                mask: feature maps( B X 1 X H X W)
        """

        b, c, h, w = x.shape

        x1 = x.reshape(b, self.ic, h // self.r, w // self.r)

        # g x
        g_x = self.g(x1).view(b, self.ic, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta
        theta_x = self.theta(x1).view(b, self.mc, -1)

        theta_x_s = theta_x.permute(0, 2, 1)
        theta_x_c = theta_x

        # phi x
        phi_x = self.phi(x1).view(b, self.mc, -1)

        phi_x_s = phi_x
        phi_x_c = phi_x.permute(0, 2, 1)

        # non-local attention
        f_s = torch.matmul(theta_x_s, phi_x_s)
        f_s_div = F.softmax(f_s, dim=-1)

        f_c = torch.matmul(theta_x_c, phi_x_c)
        f_c_div = F.softmax(f_c, dim=-1)

        # get y_s
        y_s = torch.matmul(f_s_div, g_x)
        y_s = y_s.permute(0, 2, 1).contiguous()
        y_s = y_s.view(b, c, h, w)

        # get y_c
        y_c = torch.matmul(g_x, f_c_div)
        y_c = y_c.view(b, c, h, w)

        # get z
        z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)

        # get mask
        logit = self.getmask(z.clone())
        # mask = torch.sigmoid(self.getmask(z.clone()))

        return logit, z


class NLCDetection(nn.Module):
    def __init__(self, crop_size, num_channels, ret_mask=True, reshape=True):
        super(NLCDetection, self).__init__()

        self.crop_size = crop_size

        # FENet_cfg = get_hrnet_cfg()
        # num_channels = FENet_cfg['STAGE4']['NUM_CHANNELS']

        feat1_num, feat2_num, feat3_num, feat4_num = num_channels

        self.getmask4 = NonLocalMask(feat4_num, 1)
        self.getmask3 = NonLocalMask(feat3_num, 2)
        self.getmask2 = NonLocalMask(feat2_num, 2)
        self.getmask1 = NonLocalMask(feat1_num, 4)

        self.ret_mask = ret_mask
        self.reshape = reshape

    def forward(self, feat):
        """
            inputs :
                feat : a list contains features from s1, s2, s3, s4
            output:
                mask1: output mask ( B X 1 X H X W)
                pred_cls: output cls (B X 4)
        """
        s1, s2, s3, s4 = feat

        if not self.reshape or s1.shape[2:] == self.crop_size:
            pass
        else:
            s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=True)
            s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=True)
            s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=True)
            s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=True)

        logit4, z4 = self.getmask4(s4)
        mask4 = torch.sigmoid(logit4)
        mask4U = F.interpolate(mask4, size=s3.size()[2:], mode='bilinear', align_corners=True)

        s3 = s3 * mask4U
        logit3, z3 = self.getmask3(s3)
        mask3 = torch.sigmoid(logit3)
        mask3U = F.interpolate(mask3, size=s2.size()[2:], mode='bilinear', align_corners=True)

        s2 = s2 * mask3U
        logit2, z2 = self.getmask2(s2)
        mask2 = torch.sigmoid(logit2)
        mask2U = F.interpolate(mask2, size=s1.size()[2:], mode='bilinear', align_corners=True)

        s1 = s1 * mask2U
        logit1, z1 = self.getmask1(s1)

        if self.ret_mask:
            return torch.sigmoid(logit1), mask2, mask3, mask4
        else:
            return logit1, logit2, logit3, logit4


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
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


class DetectionHead(nn.Module):
    def __init__(self, crop_size, pre_stage_channels, reshape=True):
        super(DetectionHead, self).__init__()
        self.crop_size = crop_size

        # FENet_cfg = get_hrnet_cfg()
        # pre_stage_channels = FENet_cfg['STAGE4']['NUM_CHANNELS']

        # classification head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)

        self.classifier = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2)
        )
        self.reshape = reshape

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = pre_stage_channels

        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, feat):
        s1, s2, s3, s4 = feat

        if not self.reshape or s1.shape[2:] == self.crop_size:
            pass
        else:
            s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=True)
            s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=True)
            s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=True)
            s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=True)

        y_list = [s1, s2, s3, s4]

        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)

        # average and flatten
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        logit = self.classifier(y)

        return logit


@BASE_MODELS.register_module()
class PSCCNet(nn.Module):
    def __init__(self, crop_size, backbone=None, pretrained=None, num_channels=(18, 36, 72, 144), ret_mask=True):
        super().__init__()
        
        if backbone is None:
            hr_cfg = get_hrnet_cfg()
            hr_cfg.PRETRAINED = pretrained
            self.FENet = get_seg_model(hr_cfg)
        else:
            self.FENet = build_backbone(backbone)
        self.SegNet = NLCDetection(crop_size, num_channels, ret_mask=ret_mask)
        self.ClsNet = DetectionHead(crop_size, num_channels)

    def forward(self, x):
        feats = self.FENet(x)
        pred_mask1, pred_mask2, pred_mask3, pred_mask4 = self.SegNet(feats)
        pred_logit = self.ClsNet(feats)

        return [pred_mask1, pred_mask2, pred_mask3, pred_mask4], pred_logit


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck_
}


class DCTStream(nn.Module):
    def __init__(self, dc_extra, dct_pretrained=None):
        super().__init__()

        if 'dc_other_layer' in dc_extra:
            out_channels = dc_extra.dc_other_layer.layer0_out
        else:
            out_channels = 64

        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(in_channels=21,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      dilation=8,
                      padding=8),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.dc_layer2 = self._make_layer(BasicBlock, inplanes=4 * 64 * 2, planes=96, blocks=4, stride=1)

        self.dc_stage3_cfg = dc_extra['DC_STAGE3']
        num_channels = self.dc_stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.dc_stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.dc_transition2 = self._make_transition_layer(
            [96], num_channels)
        self.dc_stage3, pre_stage_channels = self._make_stage(
            self.dc_stage3_cfg, num_channels)

        self.dc_stage4_cfg = dc_extra['DC_STAGE4']
        num_channels = self.dc_stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.dc_stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.dc_transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.dc_stage4, self.dc_final_stage_channels = self._make_stage(
            self.dc_stage4_cfg, num_channels, multi_scale_output=True)

        self.dc_final_stage_channels.insert(0, 0)  # to match # branches

        if dct_pretrained is not None and os.path.isfile(dct_pretrained):
            pretrained_dict = torch.load(dct_pretrained, map_location='cpu')
            print('=> loading DCTStream pretrained model {}'.format(dct_pretrained))
            model_dict = self.state_dict()
            pretrained_dict_used = {}

            nopretrained_dict = {k: v for k, v in model_dict.items()}

            for k, v in pretrained_dict.items():
                if 'model.' in k:
                    k = k.replace('model.', '')

                if k in model_dict.keys():
                    pretrained_dict_used[k] = v
                    nopretrained_dict.pop(k)

            for k, _ in nopretrained_dict.items():
                print('not loading pretrained weights for {}'.format(k))

            model_dict.update(pretrained_dict_used)
            ret = self.load_state_dict(model_dict)
            print('=> loaded DCTStream pretrained model {}'.format(dct_pretrained), ret)

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

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, qtable):
        x = self.dc_layer0_dil(x)
        x = self.dc_layer1_tail(x)
        B, C, H, W = x.shape
        x0 = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)  # [B, C, 8, 8, 32, 32]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, 32, 32]
        x1 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x = torch.cat([x0, x1], dim=1)
        x = self.dc_layer2(x)  # x.shape = torch.Size([1, 96, 64, 64])

        x_list = []
        for i in range(self.dc_stage3_cfg['NUM_BRANCHES']):
            if self.dc_transition2[i] is not None:
                x_list.append(self.dc_transition2[i](x))
            else:
                x_list.append(x)
        y_list = self.dc_stage3(x_list)

        x_list = []
        for i in range(self.dc_stage4_cfg['NUM_BRANCHES']):
            if self.dc_transition3[i] is not None:
                x_list.append(self.dc_transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        DC_list = self.dc_stage4(x_list)

        return DC_list


@BASE_MODELS.register_module()
class PSCCNetv3(nn.Module):
    def __init__(self, crop_size, extra, backbone=None, pretrained=None, rgb_final_channels=(18, 36, 72, 144), ret_mask=True, dct_pretrained=True,
                reshape=True):
        super().__init__()
        
        if backbone is None:
            hr_cfg = get_hrnet_cfg()
            hr_cfg.PRETRAINED = pretrained
            self.rgb_stream = get_seg_model(hr_cfg)
        else:
            self.rgb_stream = build_backbone(backbone)
        self.dct_stream = DCTStream(extra, dct_pretrained)

        self.stage5_cfg = extra['STAGE5']
        num_channels = self.stage5_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage5_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition4 = self._make_transition_layer(
            [i+j for (i, j) in zip(rgb_final_channels, self.dct_stream.dc_final_stage_channels)], num_channels)

        self.seg_net = NLCDetection(crop_size, num_channels, ret_mask=ret_mask, reshape=reshape)
        self.cls_net = DetectionHead(crop_size, num_channels, reshape=reshape)

    def forward(self, x, qtable):
        rgb_feats = self.rgb_stream(x[:, :3, :, :])
        dct_feats = self.dct_stream(x[:, 3:, :, :], qtable)

        x = [torch.cat([rgb_feats[i+1], dct_feats[i]], 1) for i in range(self.stage5_cfg['NUM_BRANCHES']-1)]
        x.insert(0, rgb_feats[0])
        x_list = []
        for i in range(self.stage5_cfg['NUM_BRANCHES']):
            if self.transition4[i] is not None:
                x_list.append(self.transition4[i](x[i]))
            else:
                x_list.append(x[i])

        pred_mask1, pred_mask2, pred_mask3, pred_mask4 = self.seg_net(x_list)
        pred_logit = self.cls_net(x_list)

        return [pred_mask1, pred_mask2, pred_mask3, pred_mask4], pred_logit

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

