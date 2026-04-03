from typing import Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.clip import CLIPViT 
from networks.layers import ViTLevelFusion 
from networks.cooi import ViTCOOI 

class Patch5Model(nn.Module):
    def __init__(self):
        super(Patch5Model, self).__init__()
        
        self.clip = CLIPViT(model_name="ViT-B-16", pretrained="openai", frozen=True)
        self.mid_dims = 256
        self.COOI = ViTCOOI()
        self.fusion = ViTLevelFusion(self.mid_dims)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.mid_dims,
            nhead=4,
            dim_feedforward=self.mid_dims,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.mha_list = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc1 = nn.Linear(768, self.mid_dims)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(self.mid_dims, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Global + Local Sequence Assembly
        # 1 CLS Token + 1 Global Token + 6 Local Tokens = Sequence of 8
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.mid_dims))
        self.seq_pos_embed = nn.Parameter(torch.randn(1, 8, self.mid_dims))

    def forward(
        self, input_img: torch.Tensor, cropped_img: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        amp = cast(Any, torch.amp)
        use_amp = cropped_img.device.type == "cuda"
        
        # We enforce no_grad on the backbone to save memory if it's truly frozen
        with torch.no_grad():
            early, mid, late = self.clip(cropped_img)
            
        with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            batch_size, _, _, _ = cropped_img.shape
            
            fused_global_maps = self.fusion(early, mid, late) # [B, 256, 14, 14]

            global_embedding = self.avgpool(fused_global_maps).flatten(1)
            global_embedding = self.ac(global_embedding).view(-1, 1, self.mid_dims) # [B, 1, 256]

            input_loc, _ = self.COOI.get_coordinates(fused_global_maps.detach(), scale)

            proposal_size = input_loc.size(1) 
            window_imgs = input_img.new_zeros(batch_size, proposal_size, 3, 224, 224)

            for batch_no in range(batch_size):
                for proposal_no in range(proposal_size):
                    t, left, b, r = input_loc[batch_no, proposal_no]
                    img_patch = input_img[batch_no][:, t:b, left:r]
                    _, patch_height, patch_width = img_patch.size()
                    if patch_height == 224 and patch_width == 224:
                        window_imgs[batch_no, proposal_no] = img_patch
                    else:
                        window_imgs[batch_no, proposal_no : proposal_no + 1] = (
                            F.interpolate(
                                img_patch[None, ...],
                                size=(224, 224),
                                mode="bilinear",
                                align_corners=True,
                            )
                        )

            window_imgs = window_imgs.reshape(
                batch_size * proposal_size, 3, 224, 224
            ).to(fused_global_maps.device)
            
        with torch.no_grad():
            # Get just the final layer maps for the local crops
            _, _, local_maps = self.clip(window_imgs.detach()) # [B*6, 768, 14, 14]
            
        with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            local_embedding = self.avgpool(local_maps).flatten(1)  # [B*6, 768]
            local_embedding = self.ac(self.fc1(local_embedding))  # [B*6, 256]
            local_embedding = local_embedding.view(batch_size, proposal_size, self.mid_dims) # [B, 6, 256]

            cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [B, 1, 256]
            
            # Assembly
            all_embeddings = torch.cat(
                (cls_tokens, global_embedding, local_embedding), dim=1
            ) # [B, 8, 256]
            
            all_embeddings = all_embeddings + self.seq_pos_embed 
            all_embeddings = self.mha_list(all_embeddings)
            
            # Predict based on the CLS token containing the summarized context
            all_logits = self.fc(all_embeddings[:, 0])

        return all_logits