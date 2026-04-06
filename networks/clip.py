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

class CLIPViT(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained="openai", frozen=True, partial_unfreeze=False):
        super(CLIPViT, self).__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, jit=False
        )
        self.visual = self.clip_model.visual
        
        if frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.eval() # Ensure dropout and batchnorm are frozen
            
        if partial_unfreeze:
            for i, (name, block) in enumerate(self.visual.transformer.resblocks.named_children()): # type:ignore
                if int(name) >= 9:  # Descongelar bloques 9, 10, 11
                    for param in block.parameters():
                        param.requires_grad = True

        self.intermediate_features = {}
        
        # Register PyTorch Forward Hooks on blocks 3, 7, and 11
        # ViT-B has 12 blocks (0 to 11)
        for name, module in self.visual.transformer.resblocks.named_children(): # type:ignore
            if name in ['3', '7', '11']:
                module.register_forward_hook(self._get_hook(name))
                
    def _get_hook(self, layer_name: str):
        def hook(module, input, output):
            # Output from OpenCLIP ViT blocks is [Sequence_Len, Batch, Dim] -> [197, B, 768]
            self.intermediate_features[layer_name] = output
        return hook
    
    def _process_feature(self, feat: torch.Tensor):
        if feat.shape[0] == 197:
            feat = feat.permute(1, 0, 2)
            
        # Capture the CLS token BEFORE dropping it!
        cls_token = feat[:, 0, :] # [B, 768]
        feat = feat[:, 1:, :] 
        
        B, N, D = feat.shape
        H = W = int(N ** 0.5) 
        
        spatial_map = feat.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        return spatial_map, cls_token

    def forward(self, x: torch.Tensor):
        self.intermediate_features.clear()
        
        with torch.no_grad() if not self.visual.conv1.weight.requires_grad else torch.enable_grad():  #type:ignore
            _ = self.visual(x.type(self.visual.conv1.weight.dtype)) #type:ignore
            
        early_map, early_cls = self._process_feature(self.intermediate_features['3'])
        mid_map, mid_cls = self._process_feature(self.intermediate_features['7'])
        late_map, late_cls = self._process_feature(self.intermediate_features['11'])

        # Return Spatial Maps (for fusion) AND Cls tokens (for final embeddings)
        return (early_map, mid_map, late_map), (early_cls, mid_cls, late_cls)
