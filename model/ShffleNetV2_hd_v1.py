import torch.nn as nn
import torch
from model.backbone.ShffleNetv2_base_hd_v1 import ShuffleNetV2
from model.backbone.Common_fun import SELayer, h_swish
import torch.nn.functional as F

###########################################################################################
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
        self.raw_img_moudle = ShuffleNetV2()
        last_channel = 64
        self.bottleneck = nn.Sequential(nn.Conv2d(last_channel, last_channel, kernel_size=1, padding=0),
                                        nn.BatchNorm2d(last_channel),
                                        nn.ReLU(inplace=True))
        self.dec = nn.Conv2d(last_channel, 1, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(last_channel, num_class)
   
    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = self.raw_img_moudle.forward(x)
        x = self.bottleneck(x)
        x_map = torch.sigmoid(self.dec(x))

        output = self.avg_pool(x)
        
        x = self.fc(self.drop(output.squeeze(-1).squeeze(-1)))
        return x, x_map
"""
###################TWO MODALITY##########################################
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
        self.one_backbone = ShuffleNetV2()
        self.two_backbone = ShuffleNetV2()
        init_channel = 64
        self.cross_atten = CrossAtten(channel=init_channel)
        self.bottleneck = nn.Sequential(nn.Conv2d(init_channel * 2, init_channel, kernel_size=1, padding=0),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))

        self.dec = nn.Conv2d(init_channel, 1, kernel_size=1, stride=1, padding=0)
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        color, depth, ir = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:]
        one = ir
        two = depth
        # two = color
        # backbone net
        one_feas = self.one_backbone(one)
        two_feas = self.two_backbone(two)

        two_one = self.cross_atten(two_feas, one_feas)

        # 融合后的后续操作
        fea = torch.cat([two_one, two_feas], dim=1)
        x = self.bottleneck(fea)
    
        x_map = torch.sigmoid(self.dec(x))

        # x = x.view(x.size(0), -1)

        regmap8 =  self.avgpool8(x)
        x = self.fc(self.drop(regmap8.squeeze(-1).squeeze(-1)))

        two_feas = two_feas.view(two_feas.size(0), -1)
        one_feas = one_feas.view(one_feas.size(0), -1)
        
        return x, two_feas, one_feas, None, x_map
"""
"""
###################SCORE FUSION##########################################
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
        self.rgb_backbone = ShuffleNetV2()
        self.depth_backbone = ShuffleNetV2()
        self.ir_backbone = ShuffleNetV2()
    
    def forward(self, x):
        color, depth, ir = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:]
        # backbone net
        color_feas = self.rgb_backbone(color)
        depth_feas = self.depth_backbone(depth)
        ir_feas = self.ir_backbone(ir)
        
        x = depth_feas

        depth_feas = depth_feas.view(depth_feas.size(0), -1)
        color_feas = color_feas.view(color_feas.size(0), -1)
        ir_feas = ir_feas.view(ir_feas.size(0), -1)

        return x, depth_feas, color_feas, ir_feas, ir_feas
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
        self.rgb_backbone = ShuffleNetV2()
        self.depth_backbone = ShuffleNetV2()
        self.ir_backbone = ShuffleNetV2()
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
        # 3-modality version (commented out):
        # color, depth, ir = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:]

        # 4-modality version (FIXED - removed duplicate channel indices):
        # Input x expected shape: (batch, 12, H, W) where channels are:
        # [0:3] = Color, [3:6] = Depth, [6:9] = IR, [9:12] = Thermal
        color, depth, ir, thermal = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:], x[:, 9:12,:,:]
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