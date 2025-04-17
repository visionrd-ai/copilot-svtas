import torch
import torch.nn as nn
from models.head import MemoryTCNHead
from models.neck import AvgPoolNeck
from models.resnet import ResNet
from models.resnetTSM import ResNetTSM
from models.postprocessing import StreamScorePostProcessing

class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class vfuseACT(nn.Module):
    def __init__(self, clip_seg_num=32, sample_rate=4):
        super(vfuseACT, self).__init__()
        # import pdb; pdb.set_trace()
        self.backbone = ResNetTSM(
            depth=50, 
            pretrained="../clsLess_tsm_r50_256p_1x1x8_kinetics400_rgb_.pth" ,
            clip_seg_num=clip_seg_num, 
            shift_div= 8, 
            out_indices= (3, ))
        
        self.det_backbone =  ResNet(
            depth=50, 
            pretrained='cleaned.pth')
       
        self.neck = AvgPoolNeck(
            num_classes= 5, 
            in_channels= 2048, 
            clip_seg_num= clip_seg_num, 
            drop_ratio= 0.5 ,
            need_pool= True)
        
        self.head = MemoryTCNHead(
            num_stages= 1, 
            num_layers= 4, 
            num_f_maps= 64 ,
            dim= 2048, 
            num_classes= 5 ,
            sample_rate= sample_rate)
        self.feat_combine = torch.nn.Sequential(
            conv_bn_relu(in_channels=4096, out_channels=2048, kernel_size=3, padding=1, dilation=1),
            torch.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
        )

        self.feat_refine = torch.nn.Sequential(
            conv_bn_relu(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, dilation=1)
        )

        self.sample_rate = self.head.sample_rate
        self.det_weights = torch.load('../new_cleaned.pth')
        self.det_backbone =  ResNet(depth=50, pretrained='new_cleaned.pth')
        self.init_weights()

    def init_weights(self):
        if self.backbone is not None:
            self.backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])

        if self.det_backbone is not None:
            # self.det_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r''), (r'conv.net', 'conv')])
            self.det_backbone.load_state_dict(self.det_weights, strict=True)
            print('det backbone loaded')
            for param in self.det_backbone.parameters(): 
                param.requires_grad=False
            print('det backbone freezed')

        if self.neck is not None:
            self.neck.init_weights()
        if self.head is not None:
            self.head.init_weights()
    
    def _clear_memory_buffer(self):
        if self.backbone is not None:
            self.backbone._clear_memory_buffer()
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.head is not None:
            self.head._clear_memory_buffer()

    def feature_fusion(self, temporal_fet, spatial_feat):

        x_1 = temporal_fet
        x_2 = spatial_feat
        x_concat = torch.cat([x_1, x_2], dim=1)
        x_3 = self.feat_combine(x_concat)
        feature = self.feat_refine(x_3)

        return feature
        
    
    def forward(self, input_data):
        
        masks = input_data['masks']
        imgs = input_data['imgs']

        # import pdb; pdb.set_trace()
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)
       
        imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))

        backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_1 = self.backbone(imgs, backbone_masks)
        x_2 = self.det_backbone(imgs, backbone_masks)

        x = self.feature_fusion(x_1, x_2)
        
        x = self.neck(x, masks[:, :, ::self.sample_rate])
        x = self.head(x, masks)

        return x
    

class vfuseACT_multihead(nn.Module):
    def __init__(self, clip_seg_num=32, sample_rate=4):
        super(vfuseACT_multihead, self).__init__()
        # import pdb; pdb.set_trace()
        self.backbone = ResNetTSM(
            depth=50, 
            pretrained="../data/clsLess_tsm_r50_256p_1x1x8_kinetics400_rgb_.pth" ,
            clip_seg_num=clip_seg_num, 
            shift_div= 8, 
            out_indices= (3, ))
        
        self.det_backbone =  ResNet(
            depth=50, 
            pretrained='../data/new_cleaned.pth')
       
        self.neck = AvgPoolNeck(
            num_classes= 20, 
            in_channels= 2048, 
            clip_seg_num= clip_seg_num, 
            drop_ratio= 0.5 ,
            need_pool= True)
        
        self.action_head = MemoryTCNHead(
            num_stages= 1, 
            num_layers= 4, 
            num_f_maps= 64 ,
            dim= 2048, 
            num_classes= 5 ,
            sample_rate= sample_rate)
        
        self.branch_head = MemoryTCNHead(
            num_stages= 1, 
            num_layers= 4, 
            num_f_maps= 64 ,
            dim= 2048, 
            num_classes= 15 ,
            sample_rate= sample_rate)
        
        self.feat_combine = torch.nn.Sequential(
            conv_bn_relu(in_channels=4096, out_channels=2048, kernel_size=3, padding=1, dilation=1),
            torch.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
        )

        self.feat_refine = torch.nn.Sequential(
            conv_bn_relu(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, dilation=1)
        )

        self.sample_rate = self.action_head.sample_rate
        self.det_weights = torch.load('../data/new_cleaned.pth')
        self.det_backbone =  ResNet(depth=50, pretrained='../data/new_cleaned.pth')
        self.init_weights()

    def init_weights(self):
        if self.backbone is not None:
            self.backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])

        if self.det_backbone is not None:
            # self.det_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r''), (r'conv.net', 'conv')])
            self.det_backbone.load_state_dict(self.det_weights, strict=True)
            print('det backbone loaded')
            for param in self.det_backbone.parameters(): 
                param.requires_grad=False
            print('det backbone freezed')

        if self.neck is not None:
            self.neck.init_weights()
        if self.action_head is not None:
            self.action_head.init_weights()
        if self.branch_head is not None:
            self.branch_head.init_weights()
    
    def _clear_memory_buffer(self):
        if self.backbone is not None:
            self.backbone._clear_memory_buffer()
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.action_head is not None:
            self.action_head._clear_memory_buffer()
        if self.branch_head is not None:
            self.branch_head._clear_memory_buffer()

    def feature_fusion(self, temporal_fet, spatial_feat):

        x_1 = temporal_fet
        x_2 = spatial_feat
        x_concat = torch.cat([x_1, x_2], dim=1)
        x_3 = self.feat_combine(x_concat)
        feature = self.feat_refine(x_3)

        return feature
        
    
    def forward(self, input_data):
        
        masks = input_data['masks']
        imgs = input_data['imgs']

        # import pdb; pdb.set_trace()
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)
       
        imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))

        backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_1 = self.backbone(imgs, backbone_masks)
        x_2 = self.det_backbone(imgs, backbone_masks)

        feature = self.feature_fusion(x_1, x_2)
        
        seg_feature = self.neck(feature, masks[:, :, ::self.sample_rate])
        action_head_score = self.action_head(seg_feature, masks)
        branch_head_score = self.branch_head(seg_feature, masks)


        return {'branch_score':branch_head_score, 'action_score':action_head_score}
    


class vfuseACT_multihead_small(nn.Module):
    def __init__(self, clip_seg_num=32, sample_rate=4):
        super(vfuseACT_multihead_small, self).__init__()
        # import pdb; pdb.set_trace()
        self.backbone = ResNetTSM(
            depth=50, 
            pretrained="../data/clsLess_tsm_r50_256p_1x1x8_kinetics400_rgb_.pth" ,
            clip_seg_num=clip_seg_num, 
            shift_div= 8, 
            out_indices= (3, ))
        
       
        self.neck = AvgPoolNeck(
            num_classes= 37, 
            in_channels= 2048, 
            clip_seg_num= clip_seg_num, 
            drop_ratio= 0.5 ,
            need_pool= True)
        
        self.action_head = MemoryTCNHead(
            num_stages= 1, 
            num_layers= 4, 
            num_f_maps= 64 ,
            dim= 2048, 
            num_classes= 11 ,
            sample_rate= sample_rate)
        
        self.branch_head = MemoryTCNHead(
            num_stages= 1, 
            num_layers= 4, 
            num_f_maps= 64 ,
            dim= 2048, 
            num_classes= 26 ,
            sample_rate= sample_rate)
        
        self.feat_combine = torch.nn.Sequential(
            conv_bn_relu(in_channels=4096, out_channels=2048, kernel_size=3, padding=1, dilation=1),
            torch.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
        )

        self.feat_refine = torch.nn.Sequential(
            conv_bn_relu(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, dilation=1)
        )

        self.sample_rate = self.action_head.sample_rate
        self.init_weights()

    def init_weights(self):
        if self.backbone is not None:
            self.backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])

        if self.neck is not None:
            self.neck.init_weights()
        if self.action_head is not None:
            self.action_head.init_weights()
        if self.branch_head is not None:
            self.branch_head.init_weights()
    
    def _clear_memory_buffer(self):
        if self.backbone is not None:
            self.backbone._clear_memory_buffer()
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.action_head is not None:
            self.action_head._clear_memory_buffer()
        if self.branch_head is not None:
            self.branch_head._clear_memory_buffer()
    
    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']
        masks = masks.unsqueeze(1)
        imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
        if self.backbone is not None:
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            feature = self.backbone(imgs, backbone_masks)
        else:
            feature = imgs
        
        if self.neck is not None:
            seg_feature = self.neck(feature, masks[:, :, ::self.sample_rate])
        else:
            seg_feature = feature

        if self.action_head is not None:
            action_head_score = self.action_head(seg_feature, masks)
        else:
            action_head_score = seg_feature

        if self.branch_head is not None:
            branch_head_score = self.branch_head(seg_feature, masks)
        else:
            branch_head_score = seg_feature


        return {'branch_score':branch_head_score, 'action_score':action_head_score}