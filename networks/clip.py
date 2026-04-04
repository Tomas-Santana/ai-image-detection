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
    def __init__(self, model_name="ViT-B-16", pretrained="openai", frozen=True):
        super(CLIPViT, self).__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, jit=False
        )
        self.visual = self.clip_model.visual
        
        if frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.eval() # Ensure dropout and batchnorm are frozen
            
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
    

class DFGM(nn.Module):
    """Deepfake-Specific Feature Guidance Module (GFF, sec. 3.3).
    Bottleneck trainable insertado entre MHSA y MLP de cada bloque ViT.
    CLIP permanece frozen; solo DFGM se entrena.
    """
    def __init__(self, d: int = 768, d_mid: int = 256):
        super().__init__()
        self.down = nn.Linear(d, d_mid)
        self.mid  = nn.Linear(d_mid, d_mid)
        self.up   = nn.Linear(d_mid, d)
        self.act  = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [seq_len, B, 768]  — residual addition
        return self.up(self.act(self.mid(self.act(self.down(z))))) + z


class CLIPViTWithDFGM(nn.Module):
    """Frozen CLIP ViT-B/16 con DFGM entrenable en cada bloque.

    SOLO para la rama global (cropped_img, batch=B).
    Para la rama local usa CLIPViT normal (sin DFGM).

    Retorna:
        spatial_maps : (early, mid, late)  — [B, 768, 14, 14] c/u
        cls_tokens   : (early, mid, late)  — [B, 768] c/u
        all_cls      : lista de 12 tensores [B, 768], uno por bloque
    """

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai", d_mid: int = 256):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, jit=False
        )
        self.visual = clip_model.visual

        # Congelar TODO CLIP
        for param in self.visual.parameters(): #type:ignore
            param.requires_grad = False

        # Detectar el nombre del atributo de atención en esta versión de OpenCLIP
        sample_block = self.visual.transformer.resblocks[0]  # type: ignore
        if hasattr(sample_block, 'attn'):
            self._attn_attr = 'attn'
        elif hasattr(sample_block, 'attention'):
            self._attn_attr = 'attention'
        else:
            raise RuntimeError(
                f"No se encontró atributo de atención en el bloque ViT. "
                f"Atributos disponibles: {[a for a in dir(sample_block) if 'att' in a.lower()]}"
            )

        # Un DFGM por bloque (12 en ViT-B)
        n_blocks = len(self.visual.transformer.resblocks)  # type: ignore
        self.dfgm_modules = nn.ModuleList([
            DFGM(d=768, d_mid=d_mid) for _ in range(n_blocks)
        ])

    def _patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Replica el patch embedding de CLIP ViT-B/16."""
        # x: [B, 3, 224, 224]
        x = self.visual.conv1(x)                                          # type: ignore [B, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)       # [B, 196, 768]
        # Prepend CLS
        cls = self.visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)  # type: ignore
        x = torch.cat([cls, x], dim=1)                                    # [B, 197, 768]
        x = x + self.visual.positional_embedding.to(x.dtype)              # type: ignore
        x = self.visual.ln_pre(x)                                         # type: ignore [B, 197, 768]
        return x.permute(1, 0, 2)                                          # [197, B, 768] — seq-first

    def _process_feature(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """feat: [197, B, 768] -> (spatial_map [B,768,14,14], cls [B,768])"""
        feat = feat.permute(1, 0, 2)          # [B, 197, 768]
        cls_token = feat[:, 0, :]             # [B, 768]
        patches   = feat[:, 1:, :]            # [B, 196, 768]
        B, N, D   = patches.shape
        H = W     = int(N ** 0.5)
        spatial   = patches.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        return spatial, cls_token

    def forward(self, x: torch.Tensor) -> Tuple:
        x = x.type(self.visual.conv1.weight.dtype)  # type: ignore

        # Patch embedding (frozen, no grad)
        with torch.no_grad():
            h = self._patch_embed(x)   # [197, B, 768]

        all_cls: list = []
        intermediate: dict = {}

        for idx, block in enumerate(self.visual.transformer.resblocks):  # type: ignore
            attn_module = getattr(block, self._attn_attr)

            # MHSA step — frozen
            with torch.no_grad():
                z = attn_module(block.ln_1(h)) + h   # [197, B, 768]

            # DFGM step — entrenable, residual ya incluido en DFGM.forward
            z = self.dfgm_modules[idx](z)             # [197, B, 768]

            # MLP step — frozen
            with torch.no_grad():
                h = block.mlp(block.ln_2(z)) + z      # [197, B, 768]

            intermediate[str(idx)] = h
            all_cls.append(h.permute(1, 0, 2)[:, 0, :])  # [B, 768]

        early_map, early_cls = self._process_feature(intermediate['3'])
        mid_map,   mid_cls   = self._process_feature(intermediate['7'])
        late_map,  late_cls  = self._process_feature(intermediate['11'])

        return (early_map, mid_map, late_map), (early_cls, mid_cls, late_cls), all_cls
