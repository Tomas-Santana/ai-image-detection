import torch
import torch.nn as nn
import open_clip
from typing import Tuple


class CLIPResNet(nn.Module):
    visual: open_clip.model.ModifiedResNet

    def __init__(self, model_name="RN50", frozen=True):
        super(CLIPResNet, self).__init__()
        # Load CLIP model (jit=False to allow hooking/getting attributes)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", jit=False
        )
        self.visual: open_clip.model.ModifiedResNet = self.clip_model.visual

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def stem_no_pool(self, x: torch.Tensor) -> torch.Tensor:
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x))) # ty:ignore[call-non-callable]
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x))) # ty:ignore[call-non-callable]
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x))) # ty:ignore[call-non-callable]
        return x
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = x.type(self.visual.conv1.weight.dtype)  # X shape: [B, 3, H, W]

        shallow = self.stem_no_pool(x)  # X shape: [B, 64, H/4, W/4]

        # 2. High Level Features
        x = self.visual.avgpool(shallow)
        x = self.visual.layer1(x)
        x = self.visual.layer2(x)
        x = self.visual.layer3(x)
        high = self.visual.layer4(x)  # X shape: [B, 2048, H/32, W/32]

        return shallow, high

    
    