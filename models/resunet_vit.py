import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet34
from timm.models.vision_transformer import VisionTransformer

class ResNetUNetViT(nn.Module):
    def __init__(self, n_classes=20, input_channels=29):
        super().__init__()

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
        self.encoder2  = base_model.layer1   # → [B,  64, 128,128]
        self.encoder3  = base_model.layer2   # → [B, 128,  64, 64]
        self.encoder4  = base_model.layer3   # → [B, 256,  32, 32]
        self.encoder5  = base_model.layer4   # → [B, 512,  16, 16]  (if input was 512×512)

        # ─── Bottleneck: Vision Transformer ─────────────────────────────────────────
        # We know e5 is 16×16 spatially, ViT’s img_size=16 & patch_size=2 → 8×8 tokens
        self.vit = VisionTransformer(
            img_size=16,
            patch_size=2,
            in_chans=512,
            embed_dim=512,
            depth=4,
            num_heads=8,
            num_classes=0,
            qkv_bias=True,
            global_pool='',
            class_token=False
        )

        # ─── Decoder ────────────────────────────────────────────────────────────────
        # Each upX must match the #channels it actually *receives*
        # 1) up4 only sees the ViT output, which we’ll project back to 256:
        self.up4 = self._decoder_block(512, 256)
        # 2) up3 sees cat([d4(256), e4(256)]) = 512
        self.up3 = self._decoder_block(512, 128)
        # 3) up2 sees cat([d3(128), e3(128)]) = 256
        self.up2 = self._decoder_block(256, 64)
        # 4) up1 sees cat([d2(64),  e2(64)])  = 128
        self.up1 = self._decoder_block(128, 32)

        # final 1×1 to get to n_classes
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # ─── Encoder ───────────────────────────────────────────────────────────────
        e1 = self.encoder1(x)                # [B,  64, 256,256]
        e2 = self.encoder2(self.pool(e1))    # [B,  64, 128,128]
        e3 = self.encoder3(e2)               # [B, 128,  64, 64]
        e4 = self.encoder4(e3)               # [B, 256,  32, 32]
        e5 = self.encoder5(e4)               # [B, 512,  16, 16]

        # ─── Bottleneck ────────────────────────────────────────────────────────────
        v = self.vit(e5)                     # [B, 8*8=64, 512]
        v = rearrange(v, 'b (h w) c -> b c h w', h=8, w=8)
        # bring it back to e4’s spatial
        v = F.interpolate(v,
                          size=e4.shape[2:],
                          mode='bilinear',
                          align_corners=False)

        # ─── Decoder ───────────────────────────────────────────────────────────────
        d4 = self.up4(v)                     # [B,256, 32,32]
        e4_ = F.interpolate(e4,
                            size=d4.shape[2:],
                            mode='bilinear',
                            align_corners=False)
        d3 = self.up3(torch.cat([d4, e4_], dim=1))  # → [B,128,64,64]

        e3_ = F.interpolate(e3,
                            size=d3.shape[2:],
                            mode='bilinear',
                            align_corners=False)
        d2 = self.up2(torch.cat([d3, e3_], dim=1))  # → [B, 64,128,128]

        e2_ = F.interpolate(e2,
                            size=d2.shape[2:],
                            mode='bilinear',
                            align_corners=False)
        d1 = self.up1(torch.cat([d2, e2_], dim=1))  # → [B, 32,256,256]

        out = self.final(d1)                 # → [B, n_classes, 256,256]
        out = F.interpolate(out,
                            size=x.shape[2:],
                            mode='bilinear',
                            align_corners=False)
        return out
