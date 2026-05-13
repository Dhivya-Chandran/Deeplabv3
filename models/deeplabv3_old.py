import torch
import torch.nn as nn
import torch.nn.functional as F

from .assp import ASSP
from .resnet_50 import ResNet_50

class DeepLabv3(nn.Module):
  
  def __init__(self, nc, dropout_rate=0.0, output_stride=16):
    """DeepLabv3 semantic segmentation model.
    
    Args:
        nc: Number of classes
        dropout_rate: Dropout rate (not in original paper, set to 0.0)
        output_stride: 16 or 8 (encoder downsampling factor)
    """
    super(DeepLabv3, self).__init__()
    
    self.nc = nc
    self.output_stride = output_stride
    
    # Encoder: ResNet50 with output_stride control
    self.resnet = ResNet_50(output_stride=output_stride)
    
    # ASSP: Atrous Spatial Pyramid Pooling
    self.assp = ASSP(in_channels=1024, out_channels=256, output_stride=output_stride)
    
    # Decoder: Low-level feature projection + decoder conv layers
    # Low-level features from layer1 have 256 channels and stride 4
    self.conv_low = nn.Sequential(
        nn.Conv2d(256, 48, kernel_size=1, bias=False),
        nn.BatchNorm2d(48),
        nn.ReLU(inplace=True)
    )
    
    # Decoder convolutions: concatenate ASSP (256ch) + low-level (48ch) = 304ch
    self.decoder = nn.Sequential(
        nn.Conv2d(256 + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
    )
    
    # Update BN momentum in decoder
    for m in self.decoder.modules():
      if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.99
    for m in self.conv_low.modules():
      if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.99
    
    # Final classification layer
    self.classifier = nn.Conv2d(256, nc, kernel_size=1, stride=1)
        
  def forward(self, x):
    h, w = x.shape[-2:]
    
    # Encoder: ResNet50 returns encoder output (stride 16 or 8) and low-level features (stride 4)
    x_enc, x_low = self.resnet(x)
    
    # ASSP: multi-scale feature extraction at stride 16/8
    x_assp = self.assp(x_enc)
    
    # Decoder: 
    # 1. Upsample ASSP output 4× to match stride 4 with low-level features
    x_assp_4x = F.interpolate(x_assp, scale_factor=4, mode='bilinear', align_corners=False)
    
    # 2. Project low-level features to 48 channels
    x_low_proj = self.conv_low(x_low)
    
    # 3. Concatenate and decode
    x_cat = torch.cat([x_assp_4x, x_low_proj], dim=1)  # [256+48, H/4, W/4]
    x_dec = self.decoder(x_cat)  # [256, H/4, W/4]
    
    # 4. Classify and upsample to original resolution
    x_out = self.classifier(x_dec)  # [nc, H/4, W/4]
    x_out = F.interpolate(x_out, size=(h, w), mode='bilinear', align_corners=False)
    
    return x_out
