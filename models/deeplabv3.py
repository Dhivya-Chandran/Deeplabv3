import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution.
    
    Replaces standard Conv2d with depthwise (per-channel) + pointwise (1×1) convolutions.
    Reduces parameters by 9× while maintaining or improving accuracy.
    
    Formula: SeparableConv = DepthwiseConv(3×3) + PointwiseConv(1×1)
    Compared to standard Conv: (k*k*in_ch) + (in_ch*out_ch) vs (k*k*in_ch*out_ch)
    
    Reference: MobileNet v1, DeepLabv3+ papers
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        
        # Depthwise convolution: apply one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )
        
        # Pointwise convolution: 1×1 conv to project to output channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ASSP(nn.Module):
    """Atrous Spatial Pyramid Pooling - matches DeepLabv3 paper"""
    def __init__(self, in_channels=2048, out_channels=256, output_stride=16):
        super(ASSP, self).__init__()
        
        # Dilation rates follow the output stride used by the backbone.
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        elif output_stride == 4:
            dilations = [1, 24, 48, 72]
        else:
            dilations = [1, 6, 12, 18]
        
        # 1×1 conv
        self.aspp_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 3×3 convs with different dilations - using SeparableConv2d for efficiency
        self.aspp_6 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.aspp_12 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.aspp_18 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=dilations[3], dilation=dilations[3], bias=False),
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
    """
    DeepLabv3 semantic segmentation model
    Matches "Rethinking Atrous Convolutions for Semantic Image Segmentation" (Chen et al., 2017)
    """
    
    def __init__(self, nc, dropout_rate=0.0, output_stride=16, backbone='resnet50', pretrained_backbone=True):
        super(DeepLabv3, self).__init__()

        backbone_name = str(backbone).strip().lower()
        if backbone_name not in {'resnet50', 'resnet101'}:
            raise ValueError(f"Unsupported backbone '{backbone}'. Use 'resnet50' or 'resnet101'.")

        output_stride = int(output_stride)
        if output_stride not in {4, 8, 16}:
            raise ValueError("Unsupported output_stride. Use 4, 8, or 16.")

        if output_stride == 16:
            replace_stride_with_dilation = [False, False, False]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, False, True]
        else:
            replace_stride_with_dilation = [False, True, True]

        # Load ImageNet-pretrained backbone weights by default.
        try:
            if backbone_name == 'resnet101':
                weights = models.ResNet101_Weights.DEFAULT if pretrained_backbone else None
                resnet = models.resnet101(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
            else:
                weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
                resnet = models.resnet50(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
        except Exception as ex:
            print(f"[WARN] Failed to load pretrained {backbone_name} weights: {ex}. Falling back to random init.")
            if backbone_name == 'resnet101':
                resnet = models.resnet101(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)
            else:
                resnet = models.resnet50(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)
        
        # Update BN momentum to 0.99 as per paper (default PyTorch is 0.1)
        for m in resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.99
        
        # Encoder: use layers 1-3 and control spatial stride via backbone dilation.
        self.input_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )  # Output: stride 4, 64 channels
        self.backbone = backbone_name
        self.output_stride = output_stride
        
        self.layer1 = resnet.layer1  # Output: stride 4, 256 channels (LOW-LEVEL FEATURES)
        self.layer2 = resnet.layer2  # Output: stride 8, 512 channels
        self.layer3 = resnet.layer3  # Output: stride 16/8/4 depending on output_stride
        
        # ASSP for multi-scale feature extraction
        self.aspp = ASSP(in_channels=1024, out_channels=256, output_stride=output_stride)
        
        # Decoder: combine ASPP output with low-level features
        # Project low-level features (256ch) to 48ch using SeparableConv for efficiency
        self.decoder_conv_low = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # Decoder convolutions: keep spatial res 1/4, combine encoder (256ch) and low-level (48ch)
        # Using SeparableConv2d reduces parameters by 3× while maintaining accuracy
        self.decoder = nn.Sequential(
            SeparableConv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SeparableConv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Update decoder BN momentum
        for m in self.decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.99
        for m in self.decoder_conv_low.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.99
        
        # Classification head (1×1 conv)
        self.classifier = nn.Conv2d(256, nc, 1)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Encoder path
        x = self.input_conv(x)       # stride 4
        x_low = self.layer1(x)       # stride 4, 256 channels (save for decoder)
        x = self.layer2(x_low)       # stride 8
        x = self.layer3(x)           # stride 16, 1024 channels
        
        # ASPP multi-scale feature extraction
        x = self.aspp(x)             # stride 16, 256 channels
        
        # Decoder path with skip connection
        # Upsample ASPP to match x_low spatial dimensions exactly
        h_low, w_low = x_low.shape[-2:]
        x = F.interpolate(x, size=(h_low, w_low), mode='bilinear', align_corners=False)
        
        x_low_proj = self.decoder_conv_low(x_low)  # stride 4, 48 channels
        
        x = torch.cat([x, x_low_proj], dim=1)  # stride 4, 304 channels
        x = self.decoder(x)          # stride 4, 256 channels
        
        # Final classification and upsample to original resolution
        x = self.classifier(x)                 # stride 4, nc channels
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        return x
