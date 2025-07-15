import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchvision.models import resnet34
from timm.models.vision_transformer import VisionTransformer

class ResNetUNetViT(nn.Module):
    def __init__(self, n_classes=20, input_channels=29):
        super().__init__()

        # Load ResNet-34 base model
        base_model = resnet34(weights=None)

        # Replace input conv to match input channel count (29 bands)
        self.input_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1 = nn.Sequential(self.input_conv, base_model.bn1, base_model.relu)
        self.pool = base_model.maxpool
        self.encoder2 = base_model.layer1
        self.encoder3 = base_model.layer2
        self.encoder4 = base_model.layer3
        self.encoder5 = base_model.layer4  # Output shape ~ [B, 512, 8, 8] for 512×512 inputs

        # Vision Transformer expects [B, C, H, W], handles patchification internally
        self.vit = VisionTransformer(
            img_size=16,  # ← matches e5 spatial size
            patch_size=2,  # ← 16x16 / 2x2 → 8x8 = 64 patches
            in_chans=512,
            embed_dim=512,
            depth=4,
            num_heads=8,
            num_classes=0,
            qkv_bias=True,
            global_pool='',
            class_token=False
        )

        # Decoder layers
        self.up4 = self._decoder_block(512, 256)
        self.up3 = self._decoder_block(512, 128)
        self.up2 = self._decoder_block(256, 64)
        self.up1 = self._decoder_block(128, 32)
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        x_vit = self.vit(e5)  # shape [B, 64, 512]
        x_vit = rearrange(x_vit, 'b (h w) c -> b c h w', h=8, w=8)
        x_vit = torch.nn.functional.interpolate(x_vit, size=e4.shape[2:], mode='bilinear', align_corners=False)

        d4 = self.up4(x_vit)
        e4 = F.interpolate(e4, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.up3(torch.cat([d4, e4], dim=1))

        e3 = F.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up2(torch.cat([d3, e3], dim=1))

        e2 = F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.up1(torch.cat([d2, e2], dim=1))

        out = self.final(d1)
        out = torch.nn.functional.interpolate(
            out, size=(x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=False
        )
        return out



