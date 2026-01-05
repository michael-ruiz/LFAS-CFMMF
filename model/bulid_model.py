def get_model(config, num_class, is_pruning=False):
    myself_net = ['FeatherNetB','FeatherNetA', 'ResNet_hd', 'ShffleNetV2_hd', 'ShffleNetV2_hd_v3','ShffleNetV2_hd_v4','ShffleNetV2_hd_v5',
    'ShffleNetV2_hd_v1','ShffleNetV2_hd_v2', 'MobileNetV2_hd', 'GhostNet_hd', 'FiveNet','FiveNet_5','FiveNet_1','FiveNet_6', 'FiveNet_4', 
    'FiveNet_3','FiveNet_2', 'FiveNet_2_1','FiveNet_2_2','FiveNet_2_3','FiveNet_2_4','FiveNet_2_5','SixNet']
    if config.model in myself_net:
        if config.model in ['FeatherNetB','FeatherNetA']:
            from model.FeatherNet import Multi_FusionNet, Two_StreamNet, Single_branchNet
        elif config.model == 'ResNet_hd':
            from model.ResNet_hd import Multi_FusionNet, Single_branchNet
        elif config.model == 'ShffleNetV2_hd':
            from model.ShffleNetV2_hd import Multi_FusionNet, Single_branchNet
        elif config.model in ['ShffleNetV2_hd_v1','ShffleNetV2_hd_v2','ShffleNetV2_hd_v3','ShffleNetV2_hd_v4','ShffleNetV2_hd_v5']:
            from model.ShffleNetV2_hd_v1 import Multi_FusionNet, Single_branchNet
        elif config.model == 'MobileNetV2_hd':
            from model.MobileNetV2_hd import Multi_FusionNet, Single_branchNet
        elif config.model == 'GhostNet_hd':
            from model.GhostNet_hd import Multi_FusionNet, Single_branchNet
        elif config.model == 'SixNet':
            from model.SixNet import Multi_FusionNet, Single_branchNet
        elif config.model in ['FiveNet_5','FiveNet_1','FiveNet_6', 'FiveNet_4', 'FiveNet_3','FiveNet_2','FiveNet']:
            from model.FiveNet import Multi_FusionNet, Two_StreamNet, Single_branchNet, Two_StreamNet_Pruning
        elif config.model in ['FiveNet_2_1','FiveNet_2_2','FiveNet_2_3','FiveNet_2_4','FiveNet_2_5'] :
            from model.FiveNet_1 import Multi_FusionNet, Two_StreamNet, Single_branchNet
        else:
            raise Exception('This model name is not implemented yet.')

        if config.is_Multi:
            net = Multi_FusionNet()
        elif config.is_Wave:
            if is_pruning:
                net = Two_StreamNet_Pruning()
            else:
                net = Two_StreamNet()
        else:
            net = Single_branchNet()
    elif config.model == 'Two_stream':  # 仅限于图像进行了小波变换处理
        from model_single.Two_stream import FusionNet
        net = FusionNet()
    elif config.model == 'LMFFNet':
        from model_single.LMFFNet import LFEM_B
        net = LFEM_B()
    elif config.model == 'Two_stream1':  # 仅限于图像进行了小波变换处理
        from model_single.Two_stream1 import FusionNet
        net = FusionNet()
    elif config.model == 'FaceBagNet':
        if config.is_Multi:
            from model.FaceBagNet import FusionNet
            net = FusionNet()
        else:
            from model.FaceBagNet import Net
            net = Net()
    elif config.model == 'FourNet_5':  # 仅限于图像进行了小波变换处理
        from model.FourNet_2 import Two_StreamNet
        net = Two_StreamNet()
    elif config.model == 'inceptionv4':
        from model.ref_model.InceptionV4 import inceptionv4
        net = inceptionv4()
    elif config.model == 'LightFASNet':
        from model.LightFASNet import FeatherNet_G_B
        net = FeatherNet_G_B()
    return net