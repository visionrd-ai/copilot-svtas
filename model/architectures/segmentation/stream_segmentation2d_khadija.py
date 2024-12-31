'''
Author       : Thyssen Wen
Date         : 2022-06-13 16:22:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-06 20:37:46
Description  : Stream Segmentation 2D without backbone loss
FilePath     : /ETESVS/model/architectures/segmentation/stream_segmentation2d.py
'''
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from utils.logger import get_logger

from ...builder import build_backbone
from ...builder import build_neck
from ...builder import build_head

from ...builder import ARCHITECTURE


from model.backbones.image.resnet import ResNet
from model.backbones.video.resnet_tsm import ResNetTSM
from model.necks.avg_pool_neck import AvgPoolNeck
from model.heads.segmentation.memory_tcn import MemoryTCNHead

@ARCHITECTURE.register()
class StreamSegmentation2D(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__()
        # self.backbone = build_backbone(backbone)
        self.backbone = ResNetTSM(depth=50, pretrained="data/tsm_r50_dense_256p_1x1x8_100e_kinetics400_rgb_20200727-e1e0c785.pth" ,clip_seg_num=32, shift_div= 8, out_indices= (3, ))
        self.det_backbone =  ResNet(depth=50, pretrained='data/cleaned.pth')
        self.neck = AvgPoolNeck(num_classes= 5, in_channels= 2048, clip_seg_num= 32, drop_ratio= 0.5 ,need_pool= True)
        self.head = MemoryTCNHead(num_stages= 1, num_layers= 4, num_f_maps= 64 ,dim= 2048, num_classes= 5 ,sample_rate= 4)



        # self.backbone.init_weights()
        # self.det_backbone.init_weights()

        self.init_weights()
        self.sample_rate = head.sample_rate

    def init_weights(self):
        if self.backbone is not None:
            self.backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])

        if self.det_backbone is not None:
            self.det_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r''), (r'conv.net', 'conv')])
            for param in self.det_backbone.parameters(): 
                param.requires_grad=False
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

    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']

        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)

        # x.shape=[N,T,C,H,W], for most commonly case
        imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
        # x [N * T, C, H, W]

        if self.backbone is not None:
             # masks.shape [N * T, 1, 1, 1]
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            feature = self.backbone(imgs, backbone_masks)
            feature_det = self.det_backbone(imgs, backbone_masks)

        else:
            feature = imgs
        feature = 0.5*feature+0.5*feature_det
        # feature [N * T , F_dim, 7, 7]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature = self.neck(feature, masks[:, :, ::self.sample_rate])
            
        else:
            seg_feature = feature

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.head is not None:
            head_score = self.head(seg_feature, masks)
        else:
            head_score = seg_feature
        # seg_score [stage_num, N, C, T]
        # cls_score [N, C, T]
        return head_score