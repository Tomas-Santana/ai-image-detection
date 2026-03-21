import torch
import torch.nn as nn
import open_clip
from typing import Tuple


class CLIPResNet(nn.Module):
    visual: open_clip.model.ModifiedResNet

    def __init__(self, model_name="RN50", frozen=True, unfreeze_last_layer=False):
        super(CLIPResNet, self).__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", jit=False
        )
        self.visual: open_clip.model.ModifiedResNet = self.clip_model.visual  # type:ignore

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

        if unfreeze_last_layer:
            for param in self.visual.layer4.parameters():
                param.requires_grad = True

    def stem_no_pool(self, x: torch.Tensor) -> torch.Tensor:
        x = self.visual.act1(self.visual.bn1(self.visual.conv1(x)))  # type:ignore
        x = self.visual.act2(self.visual.bn2(self.visual.conv2(x)))  # type:ignore
        x = self.visual.act3(self.visual.bn3(self.visual.conv3(x)))  # type:ignore
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.type(self.visual.conv1.weight.dtype)  # X shape: [B, 3, H, W]

        shallow = self.stem_no_pool(x)  # X shape: [B, 64, H/4, W/4]

        # 2. Mid level features
        x = self.visual.avgpool(shallow) # [B, 64, H/32, W/32]
        x = self.visual.layer1(x)  # [B, 256, H/32, W/32]
        middle = self.visual.layer2(x) # [B, 512, H/32, W/32]
        x = self.visual.layer3(middle)   # [B, 1024, H/32, W/32]
        high = self.visual.layer4(x)  # [B, 2048, H/32, W/32]

        return shallow, middle, high
