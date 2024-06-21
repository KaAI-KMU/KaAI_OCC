import torch
import numpy as np


gt_bev_mask = torch.tensor(range(100)).reshape(2, 5, 10)
gt_bev_mask, _ = torch.max(gt_bev_mask, dim=-1, keepdim=True)



npz_file = np.load('data/nuscenes/gts/scene-0411/6240ddda9d1f4aeeabf7ab0e02403a72/labels.npz')
    
sem = torch.tensor(npz_file['semantics'])
mask_lidar = torch.tensor(npz_file['mask_lidar'])
mask_camera = torch.tensor(npz_file['mask_camera'])

print("hello world")