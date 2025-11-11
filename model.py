import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------
# ASPP Module
# ---------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()

        # 1x1 conv branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # atrous conv branches
        for d in dilations[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # image pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # final projection
        n_branches = len(self.branches) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        out = [b(x) for b in self.branches]
        img = self.image_pool(x)
        img = F.interpolate(img, size=size, mode='bilinear', align_corners=False)
        out.append(img)
        out = torch.cat(out, dim=1)
        return self.project(out)

# ---------------------------
# MobileNetV3-small ASPP + improved decoder + 3 skips
# ---------------------------
class MobileNetV3_ASPP_Seg(nn.Module):
    def __init__(self, num_classes=21, aspp_out=256):
        super().__init__()
        backbone = models.mobilenet_v3_small(pretrained=True)
        self.encoder = backbone.features

        # -------------------------
        # Three skip connections: low, mid, higher
        # -------------------------
        self.skip_layers = [2, 4, 6]  # low, mid, higher level
        self.skip_channels = [self.encoder[i].out_channels for i in self.skip_layers]

        self.skip_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for ch in self.skip_channels
        ])

        # -------------------------
        # ASPP Module
        # -------------------------
        self.backbone_out_channels = 576  # last layer of MobileNetV3-small
        self.aspp = ASPP(self.backbone_out_channels, out_channels=aspp_out)

        # -------------------------
        # Decoder
        # -------------------------
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(aspp_out + len(self.skip_layers)*64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(0.3)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        out = x
        skip_feats = []

        # Encoder forward
        for i, layer in enumerate(self.encoder):
            out = layer(out)
            if i in self.skip_layers:
                idx = self.skip_layers.index(i)
                skip_feats.append(self.skip_convs[idx](out))

        # ASPP
        x = self.aspp(out)

        # Progressive upsampling and concatenation of skip features
        for feat in skip_feats[::-1]:  # deep -> shallow
            x = F.interpolate(x, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, feat], dim=1)

        # Decoder
        x = self.decoder_conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder_conv2(x)

        # Dropout + classifier
        x = self.dropout(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        logits = self.classifier(x)
        return logits

# ---------------------------
# Test
# ---------------------------
if __name__ == '__main__':
    model = MobileNetV3_ASPP_Seg(num_classes=21)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print('Output shape:', out.shape)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params:,}')


'''
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (lightweight variant).
    Produces a fixed number of output channels.
    """

    def __init__(self, in_channels, out_channels=256, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()

        # 1x1 conv branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # atrous conv branches
        for d in dilations[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # image pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # final projection
        n_branches = len(self.branches) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n_branches, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        out = []
        for b in self.branches:
            out.append(b(x))
        img = self.image_pool(x)
        img = F.interpolate(img, size=size, mode='bilinear', align_corners=False)
        out.append(img)
        out = torch.cat(out, dim=1)
        return self.project(out)


class MobileNetV3_ASPP_Seg(nn.Module):
    """Segmentation model using MobileNetV3 small as encoder + ASPP + simple decoder.

    - backbone_pretrained: use torchvision pretrained MobileNetV3 weights if True
    - num_classes: number of segmentation classes
    - aspp_out: internal channel size for ASPP
    """

    def __init__(self, num_classes=21, aspp_out=256):
        super().__init__()
        # Load backbone
        backbone = models.mobilenet_v3_small(pretrained=True)
        # We will use backbone.features as encoder
        self.encoder = backbone.features

        # find backbone output channels (last conv in features)
        backbone_out_channels = None
        for module in reversed(self.encoder):
            if hasattr(module, 'out_channels'):
                backbone_out_channels = module.out_channels
                break
        if backbone_out_channels is None:
            # fallback
            backbone_out_channels = 576

        # ASPP
        self.aspp = ASPP(backbone_out_channels, out_channels=aspp_out)

        # Simple decoder: reduce and upsample
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(aspp_out, aspp_out // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_out // 2, aspp_out // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out // 2),
            nn.ReLU(inplace=True)
        )

        # Dropout before classifier head
        self.dropout = nn.Dropout2d(0.3)

        self.classifier = nn.Conv2d(aspp_out // 2, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder forward: pass through backbone.features
        feats = self.encoder(x)
        # backbone.features returns activation of last layer

        # ASPP
        x = self.aspp(feats)

        # Decoder conv
        x = self.decoder_conv(x)

        # Dropout before classifier
        x = self.dropout(x)

        # Upsample to input size using bilinear interpolation
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        logits = self.classifier(x)
        return logits
'''
r'''
if __name__ == '__main__':
    # quick smoke test
    model = MobileNetV3_ASPP_Seg(num_classes=21)
    model.eval()

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print('Output shape:', out.shape)  # expected [2, num_classes, 224, 224]

    # Count params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params:,}')
'''