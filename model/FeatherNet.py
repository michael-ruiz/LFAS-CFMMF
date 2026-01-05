import torch.nn as nn
import torch
from model.backbone.Common_fun import *
from model.backbone.Feather_base import *

class Single_branchNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')

    def __init__(self, num_class=2):
        super(Single_branchNet,self).__init__()
        self.backbone = FeatherNetB(model=1, init_channel=3)
        #  building last several layers
        input_channel = 64
        self.final_DW = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1,
                                                groups=input_channel, bias=False),)
        self.dec = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.backbone(x)
        x_map = torch.sigmoid(self.dec(x))
        x = self.final_DW(x)
        x = x.view(x.size(0), -1)
        return x, x_map


class Two_StreamNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #r aise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')

    def __init__(self, num_class=2):
        super(Two_StreamNet,self).__init__()
        self.raw_backbone = FeatherNetB(model=1, init_channel=3)
        self.wave_backbone = FeatherNetB(model=2, init_channel=9)
        #  building last several layers
        input_channel = 64
        self.bottleneck = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(input_channel),
                                        h_swish(),)
        self.final_DW1 = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0))
        # self.final_DW = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1, groups=input_channel, bias=False),)

    def forward(self, r_img, w_img):
        wave_img_feas = self.wave_backbone.forward(w_img)
        raw_img_feas = self.raw_backbone.forward(r_img)
        # 融合后的后续操作
        # fusion_fea = torch.cat([wave_img_feas, raw_img_feas], dim=1)
        fusion_fea = wave_img_feas + raw_img_feas
        fusion_fea = self.bottleneck(fusion_fea)
        x = self.final_DW1(fusion_fea)
        x = self.final_DW1(x)
        x = x.view(x.size(0), -1)
        # 分别将原始图像分支和小波分支平铺，作为各个分支的输出
        x_wave = wave_img_feas.view(wave_img_feas.size(0), -1)
        x_raw = raw_img_feas.view(raw_img_feas.size(0), -1)

        return x, x_wave, x_raw

"""
class Multi_FusionNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')

    def __init__(self, num_class=2):
        super(Multi_FusionNet,self).__init__()
        self.rgb_backbone = FeatherNetB(model=1, init_channel=3)
        self.depth_backbone = FeatherNetB(model=1, init_channel=3)
        self.ir_backbone = FeatherNetB(model=1, init_channel=3)
        #  building last several layers for fusion
        init_channel = 64
        self.color_SE = SELayer(init_channel, reduction=16)
        self.depth_SE = SELayer(init_channel, reduction=16)
        self.ir_SE = SELayer(init_channel, reduction=16)
        self.bottleneck = nn.Sequential(nn.Conv2d(init_channel * 3, init_channel, kernel_size=1, padding=0),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.final_DW = nn.Sequential(
            nn.Conv2d(init_channel, init_channel, kernel_size=3, stride=2, padding=1, groups=64, bias=False))

    def forward(self, x):
        color, depth, ir = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:]
        # backbone net
        color_feas = self.rgb_backbone(color)
        depth_feas = self.depth_backbone(depth)
        ir_feas = self.ir_backbone(ir)
        # 注意力机制
        color_feas = self.color_SE(color_feas)
        depth_feas = self.depth_SE(depth_feas)
        ir_feas = self.ir_SE(ir_feas)
        # 融合后的后续操作
        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        fea = self.bottleneck(fea)
        x = self.final_DW(fea)
        x = x.view(x.size(0), -1)

        return x
"""

###########################################################################################
class Multi_FusionNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=10):
        super(Multi_FusionNet,self).__init__()
        
        self.rgb_backbone = FeatherNetB(model=1, init_channel=3)
        self.depth_backbone = FeatherNetB(model=1, init_channel=3)
        self.ir_backbone = FeatherNetB(model=1, init_channel=3)
        """
        self.rgb_backbone = FeatherNetA(model=1, init_channel=3)
        self.depth_backbone = FeatherNetA(model=1, init_channel=3)
        self.ir_backbone = FeatherNetA(model=1, init_channel=3)
        """
        init_channel = 64
        self.cross_atten = CrossAtten(channel=init_channel)
        self.bottleneck = nn.Sequential(nn.Conv2d(init_channel*3, init_channel, kernel_size=1, padding=0),
                                        nn.BatchNorm2d(init_channel),
                                        nn.ReLU(inplace=True))

        # self.final_DW = nn.Sequential(nn.Conv2d(init_channel, init_channel, kernel_size=3, stride=1, padding=1, groups=64, bias=False))
        self.dec = nn.Conv2d(init_channel, 1, kernel_size=1, stride=1, padding=0)
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        color, depth, ir = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:]
        # backbone net
        color_feas = self.rgb_backbone(color)
        depth_feas = self.depth_backbone(depth)
        ir_feas = self.ir_backbone(ir)
        depth_rgb = self.cross_atten(depth_feas, color_feas)
        depth_ir = self.cross_atten(depth_feas, ir_feas)
        # 融合后的后续操作
        # fea = depth_rgb + depth_feas + depth_ir
        fea = torch.cat([depth_rgb, depth_feas, depth_ir], dim=1)
        x = self.bottleneck(fea)
        # x = self.final_DW(x)

        x_map = torch.sigmoid(self.dec(x))

        # x = x.view(x.size(0), -1)
        regmap8 =  self.avgpool8(x)
        x = self.fc(self.drop(regmap8.squeeze(-1).squeeze(-1)))


        depth_feas = depth_feas.view(depth_feas.size(0), -1)
        color_feas = color_feas.view(color_feas.size(0), -1)
        ir_feas = ir_feas.view(ir_feas.size(0), -1)

        return x, depth_feas, color_feas, ir_feas, x_map

class CrossAtten(nn.Module):
    def __init__(self, channel=64):
        super(CrossAtten, self).__init__()

        self.to_q = nn.Linear(channel, channel, bias = False)
        self.to_k = nn.Linear(channel, channel, bias = False)
        self.to_v = nn.Linear(channel, channel, bias = False)
        self.scale = channel ** -0.5


    def forward(self, x1, x2):

        classtoken1 = x1
        classtoken2 = x2
        
        B,C,H,W = classtoken1.shape
        h1_temp = classtoken1.view(B,C,-1)  # B, C, H*W
        h2_temp = classtoken2.view(B,C,-1)
        
        q = self.to_q(h1_temp.transpose(-2, -1))  # B, H*W, C
        k = self.to_k(h2_temp.transpose(-2, -1))
        v = self.to_v(h2_temp.transpose(-2, -1))
        crossh1_h2 = (q.transpose(-2, -1) @ k) * self.scale # B, C, C
        crossh1_h2 =F.softmax(crossh1_h2, dim=-1)
        crossedh1_h2 = (crossh1_h2 @ v.transpose(-2, -1)).contiguous()  # B, C, H*W
        crossedh1_h2 = crossedh1_h2.view(B,C,H,W)  # B, C, H, W
        """
        crossh1_h2 = h2_temp @ h1_temp.transpose(-2, -1)   # B, C, C
        crossh1_h2 =F.softmax(crossh1_h2, dim=-1)  
        crossedh1_h2 = (crossh1_h2 @ h2_temp).contiguous()  # B, C, H*W
        crossedh1_h2 = crossedh1_h2.view(B,C,H,W)  # B, C, H, W
        """
        logit = crossedh1_h2
        return logit