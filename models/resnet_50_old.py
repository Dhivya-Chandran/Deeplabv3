import torch
import torch.nn as nn
from torchvision import models

class ResNet_50 (nn.Module):
  def __init__(self, in_channels=3, conv1_out=64, output_stride=16):
    super(ResNet_50, self).__init__()
    
    # Load pretrained ResNet50
    self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Update BatchNorm momentum to 0.99 as per DeepLabv3 paper
    for m in self.resnet.modules():
      if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.99
    
    # Set output stride (16 or 8)
    # output_stride=16: dilate layer3 (stride 2 → 1, dilate 2)
    # output_stride=8: also dilate layer2 (stride 2 → 1, dilate 2)
    if output_stride == 16:
      # Dilate layer3 with dilation 2
      for m in self.resnet.layer3.modules():
        if isinstance(m, nn.Conv2d):
          if m.stride == (2, 2):
            m.stride = (1, 1)
            m.padding = (m.padding[0] * 2, m.padding[1] * 2)
            m.dilation = (2, 2)
    elif output_stride == 8:
      # Dilate layer2 and layer3
      for m in self.resnet.layer2.modules():
        if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
          m.stride = (1, 1)
          m.padding = (m.padding[0] * 2, m.padding[1] * 2)
          m.dilation = (2, 2)
      for m in self.resnet.layer3.modules():
        if isinstance(m, nn.Conv2d):
          if m.stride == (2, 2):
            m.stride = (1, 1)
            m.padding = (m.padding[0] * 2, m.padding[1] * 2)
            m.dilation = (2, 2)
    
    self.relu = nn.ReLU(inplace=True)
  
  def forward(self, x):
    # Stage 1: Input conv→BN→ReLU + maxpool (stride 4)
    x = self.relu(self.resnet.bn1(self.resnet.conv1(x)))
    x = self.resnet.maxpool(x)
    
    # Stage 2: layer1 (stride 4, channels 256) - save for decoder
    low_level = self.resnet.layer1(x)
    
    # Stage 3: layer2 (stride 8, channels 512) 
    x = self.resnet.layer2(low_level)
    
    # Stage 4: layer3 (stride 16 if output_stride=16, channels 1024)
    x = self.resnet.layer3(x)
    
    return x, low_level  # Return both encoder output and low-level features
