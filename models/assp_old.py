import torch
import torch.nn as nn
import torch.nn.functional as F

class ASSP(nn.Module):
  def __init__(self, in_channels, out_channels=256, output_stride=16):
    super(ASSP, self).__init__()
    
    # Dilation rates depend on output_stride
    # output_stride=16: dilations [1, 6, 12, 18]
    # output_stride=8: dilations [1, 12, 24, 36]
    if output_stride == 16:
      dilations = [1, 6, 12, 18]
    elif output_stride == 8:
      dilations = [1, 12, 24, 36]
    else:
      dilations = [1, 6, 12, 18]
    
    self.relu = nn.ReLU(inplace=True)
    
    self.conv1 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          padding = 0,
                          dilation=1,
                          bias=False)
    
    self.bn1 = nn.BatchNorm2d(out_channels)
    
    self.conv2 = nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=dilations[1],
                          dilation=dilations[1],
                          bias=False)
    
    self.bn2 = nn.BatchNorm2d(out_channels)
    
    self.conv3 = nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=dilations[2],
                          dilation=dilations[2],
                          bias=False)
    
    self.bn3 = nn.BatchNorm2d(out_channels)
    
    self.conv4 = nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=dilations[3],
                          dilation=dilations[3],
                          bias=False)
    
    self.bn4 = nn.BatchNorm2d(out_channels)
    
    self.conv5 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride=1,
                          padding = 0,
                          dilation=1,
                          bias=False)
    
    self.bn5 = nn.BatchNorm2d(out_channels)
    
    self.convf = nn.Conv2d(in_channels = out_channels * 5, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride=1,
                          padding = 0,
                          dilation=1,
                          bias=False)
    
    self.bnf = nn.BatchNorm2d(out_channels)
    
    # Set BN momentum to 0.99 as per DeepLabv3 paper
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.99
    
    self.adapool = nn.AdaptiveAvgPool2d(1)  
   
  
  def forward(self,x):
    
    x1 = self.conv1(x)
    x1 = self.bn1(x1)
    x1 = self.relu(x1)
    
    x2 = self.conv2(x)
    x2 = self.bn2(x2)
    x2 = self.relu(x2)
    
    x3 = self.conv3(x)
    x3 = self.bn3(x3)
    x3 = self.relu(x3)
    
    x4 = self.conv4(x)
    x4 = self.bn4(x4)
    x4 = self.relu(x4)
    
    x5 = self.adapool(x)
    x5 = self.conv5(x5)
    x5 = self.bn5(x5)
    x5 = self.relu(x5)
    x5 = F.interpolate(x5, size = tuple(x4.shape[-2:]), mode='bilinear')
    
    x = torch.cat((x1,x2,x3,x4,x5), dim = 1) #channels first
    x = self.convf(x)
    x = self.bnf(x)
    x = self.relu(x)
    
    return x
