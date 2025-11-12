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
        self.skip_channels = []
        for i in self.skip_layers:
            layer = self.encoder[i]
            if hasattr(layer, 'out_channels'):
                self.skip_channels.append(layer.out_channels)
            else:
                # For InvertedResidual blocks, get the last conv's out_channels
                for module in reversed(list(layer.modules())):
                    if hasattr(module, 'out_channels'):
                        self.skip_channels.append(module.out_channels)
                        break

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

        self.feature_proj = nn.ModuleDict({
            'low': nn.Sequential(
                nn.Conv2d(64, 256, 1, bias=False),  # 64 -> 256
                nn.BatchNorm2d(256)
            ),
            'mid': nn.Sequential(
                nn.Conv2d(64, 512, 1, bias=False),  # 64 -> 512
                nn.BatchNorm2d(512)
            ),
            'high': nn.Sequential(
                nn.Conv2d(64, 1024, 1, bias=False),  # 64 -> 1024
                nn.BatchNorm2d(1024)
            ),
            'aspp': nn.Sequential(
                nn.Conv2d(aspp_out, 2048, 1, bias=False),  # 256 -> 2048
                nn.BatchNorm2d(2048)
            )
        })

    def forward(self, x, return_features=False):
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
        high_feat = x  # feature for KD

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

        if return_features:
            # return logits + chosen internal features
            return logits, {
                'low': skip_feats[0],
                'mid': skip_feats[1],
                'high': skip_feats[2],
                'aspp': high_feat
            }

        return logits

# ---------------------------
# Test
# ---------------------------
if __name__ == '__main__':
    model = MobileNetV3_ASPP_Seg()
    dummy = torch.randn(2, 3, 224, 224)
    out, feats = model(dummy, return_features=True)
    for k, v in feats.items():
        print(k, v.shape)
    print("Output shape:", out.shape)
