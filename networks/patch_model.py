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
        
        with torch.no_grad():
            # Unpack the new dual returns
            spatial_maps, cls_tokens = self.clip(cropped_img)
            early, mid, late = spatial_maps
            _, _, global_cls = cls_tokens # We use the 'late' block CLS token
            
        with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            batch_size = cropped_img.shape[0]
            
            fused_global_maps = self.fusion(early, mid, late) # [B, 256, 14, 14]

            # FIX: Use the native CLS token instead of Average Pooling!
            global_embedding = self.ac(self.fc1(global_cls)) # [B, 256]
            global_embedding = global_embedding.view(-1, 1, self.mid_dims) # [B, 1, 256]

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
            # Unpack the local crops
            _, local_cls_tokens = self.clip(window_imgs.detach()) 
            _, _, local_cls = local_cls_tokens
            
        with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            # FIX: Use local CLS token instead of Average Pooling!
            local_embedding = self.ac(self.fc1(local_cls))  # [B*6, 256]
            local_embedding = local_embedding.view(batch_size, proposal_size, self.mid_dims) # [B, 6, 256]

            cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [B, 1, 256]
            
            all_embeddings = torch.cat(
                (cls_tokens, global_embedding, local_embedding), dim=1
            ) # [B, 8, 256]
            
            all_embeddings = all_embeddings + self.seq_pos_embed 
            all_embeddings = self.mha_list(all_embeddings)
            
            all_logits = self.fc(all_embeddings[:, 0])

        return all_logits