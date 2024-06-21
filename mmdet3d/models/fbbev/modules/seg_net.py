import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS
from mmcv.runner import force_fp32
from mmdet.models.backbones.resnet import BasicBlock

@HEADS.register_module()
class SegNet(BaseModule): 
    def __init__(self,
                 in_channels=80,
                 seg_channels=1,
                 mid_channels=64,
                 downsample=8,):
        super(SegNet, self).__init__()
        # 2 fully connected layers
        self.seg_channels = seg_channels
        # self.seg_mlp = nn.Sequential(
            # nn.Linear(in_channels, mid_channels),
            # nn.ReLU(inplace=True),
            # nn.Linear(mid_channels, seg_channels),
            # nn.ReLU(inplace=True)
        # )
        self.seg_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1 ,1), # kernel_size, stride, padding
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1), 
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ) 
        self.seg_head = nn.Conv2d(mid_channels, seg_channels, 1, 1, 0)
        
        self.downsample = downsample

  
    @force_fp32()
    def forward(self, x):
        # input shape: (B, N, C, H, W)
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = self.seg_block(x)
        x = self.seg_head(x)
        x = x.view(B, N, self.seg_channels, H, W)
        x = x.squeeze(-1)
        #x = F.sigmoid(x)
        return x

    def get_downsampled_gt_seg(self, gt_seg):
        """
        Input:
            gt_seg: [B, N, H, W]
        Output:
            gt_seg: [B*N, h, w]
        """
        downsample = self.downsample
        # if self.downsample == 8 and self.se_depth_map:
        #    downsample = 16
        B, N, H, W = gt_seg.shape
        
        # Reshape gt_depths to treat each image in the batch separately
        gt_seg = gt_seg.view(B * N, 1, H, W)  # Adding a channel dimension
        
        # Apply max pooling to downsample
        gt_seg = F.max_pool2d(gt_seg, kernel_size=downsample, stride=downsample)
        
        # Reshape back to the expected output size without the channel dimension
        gt_seg = gt_seg.view(B * N, H // downsample, W // downsample)
        
        return gt_seg

    @force_fp32()
    def get_seg_loss(self, seg_gt, seg_pred):
        seg_gt = self.get_downsampled_gt_seg(seg_gt)
        #seg_gt = torch.clamp(seg_gt, min=0, max=1)  
        seg_pred = seg_pred.view(-1)
        seg_gt = seg_gt.view(-1)
        mask = seg_gt >= 0
        seg_gt = seg_gt[mask]
        seg_pred = seg_pred[mask]
        
        total_pos_samples = seg_gt.sum()
        total_neg_samples = seg_gt.numel() - total_pos_samples
        pos_weight = torch.min(total_neg_samples / total_pos_samples, torch.tensor(4.0))
        pos_weight = pos_weight.to(seg_pred.device)
        
        seg_loss = F.binary_cross_entropy_with_logits(seg_pred, seg_gt, pos_weight=pos_weight) 
        loss_dict = dict(loss_seg=seg_loss)
        return loss_dict