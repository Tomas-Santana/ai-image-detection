import torch
import torch.nn as nn
import open_clip

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
