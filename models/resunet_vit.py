import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet34
from timm.models.vision_transformer import VisionTransformer

class ResNetUNetViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_classes = config["num_classes"]
        input_channels = config["input_channels"]
        vit_cfg = config.get("model", {})

        # ─── Encoder: ResNet-34 backbone ────────────────────────────────────────────
        base_model = resnet34(weights=None)

        # Swap in a conv that accepts all our input bands
        self.input_conv = nn.Conv2d(input_channels, 64,
                                    kernel_size=7, stride=2, padding=3,
                                    bias=False)

        # replicate ResNet’s first few layers
        self.encoder1 = nn.Sequential(
            self.input_conv,
            base_model.bn1,
            base_model.relu,
        )
        self.pool      = base_model.maxpool
        self.encoder2  = base_model.layer1
        self.encoder3  = base_model.layer2
        self.encoder4  = base_model.layer3
        self.encoder5  = base_model.layer4

        # ─── Bottleneck: Vision Transformer ─────────────────────────────────────────
        self.vit = VisionTransformer(
            img_size=vit_cfg.get("vit_img_size", 16),
            patch_size=vit_cfg.get("vit_patch_size", 2),
            in_chans=512,
            embed_dim=vit_cfg.get("vit_embed_dim", 512),
            depth=vit_cfg.get("vit_depth", 4),
            num_heads=vit_cfg.get("vit_heads", 8),
            num_classes=0,
            qkv_bias=True,
            global_pool='',
            class_token=False
        )

        # ─── Decoder ────────────────────────────────────────────────────────────────
        self.up4 = self._decoder_block(512, 256)
        self.up3 = self._decoder_block(512, 128)
        self.up2 = self._decoder_block(256, 64)
        self.up1 = self._decoder_block(128, 32)

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        v = self.vit(e5)  # [B, N, C]
        B, N, C = v.shape
        patches_per_dim = int(N ** 0.5)
        assert patches_per_dim ** 2 == N, f"ViT output tokens ({N}) is not a perfect square"
        v = rearrange(v, 'b (h w) c -> b c h w', h=patches_per_dim, w=patches_per_dim)
        v = F.interpolate(v, size=e4.shape[2:], mode='bilinear', align_corners=False)

        d4 = self.up4(v)
        e4_ = F.interpolate(e4, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.up3(torch.cat([d4, e4_], dim=1))

        e3_ = F.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up2(torch.cat([d3, e3_], dim=1))

        e2_ = F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.up1(torch.cat([d2, e2_], dim=1))

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
