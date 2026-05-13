import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASSP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels=2048, out_channels=256, output_stride=16):
        super(ASSP, self).__init__()
        
        # Dilation rates for output_stride=16
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            dilations = [1, 6, 12, 18]
        
        # 1×1 conv
        self.aspp_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 3×3 convs with different dilations
        self.aspp_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.aspp_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.aspp_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Image pool (global average pooling + projection)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Projection after concatenation (5 * out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Set BN momentum to 0.99 as per paper
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.99
    
    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Multi-scale features
        feat1 = self.aspp_1(x)
        feat6 = self.aspp_6(x)
        feat12 = self.aspp_12(x)
        feat18 = self.aspp_18(x)
        
        # Image pool
        feat_pool = self.image_pool(x)
        feat_pool = F.interpolate(feat_pool, (h, w), mode='bilinear', align_corners=False)
        
        # Concatenate all features
        x = torch.cat([feat1, feat6, feat12, feat18, feat_pool], dim=1)
        x = self.project(x)
        
        return x


class DeepLabv3(nn.Module):
    """DeepLabv3 semantic segmentation model"""
    
    def __init__(self, num_classes=19, output_stride=16):
        super(DeepLabv3, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Update BN momentum to 0.99
        for m in resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.99
        
        # Encoder: use layers 1-3 (layer4 removed to maintain stride 16)
        self.input_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )  # Output: stride 4, 64 channels
        
        self.layer1 = resnet.layer1  # Output: stride 4, 256 channels (low-level features)
        self.layer2 = resnet.layer2  # Output: stride 8, 512 channels
        self.layer3 = resnet.layer3  # Output: stride 16, 1024 channels (encoder output)
        
        # ASSP for multi-scale feature extraction
        self.aspp = ASSP(in_channels=1024, out_channels=256, output_stride=16)
        
        # Decoder: combine ASSP output with low-level features
        # Low-level: 256 channels at stride 4
        # ASSP: 256 channels at stride 16 (will be upsampled 4×)
        
        self.decoder_conv_low = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Update decoder BN momentum
        for m in self.decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.99
        
        # Classification head
        self.classifier = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Encoder
        x = self.input_conv(x)  # stride 4
        low_level = self.layer1(x)  # stride 4, 256 channels
        x = self.layer2(low_level)  # stride 8
        x = self.layer3(x)  # stride 16, 1024 channels
        
        # ASPP
        x = self.aspp(x)  # stride 16, 256 channels
        
        # Decoder
        # Upsample ASPP output 4× to match stride 4
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)  # stride 4
        
        # Project low-level features
        low_level = self.decoder_conv_low(low_level)  # stride 4, 48 channels
        
        # Concatenate and refine
        x = torch.cat([x, low_level], dim=1)  # stride 4, 304 channels
        x = self.decoder(x)  # stride 4, 256 channels
        
        # Classification and final upsample
        x = self.classifier(x)  # stride 4, num_classes
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        return x
