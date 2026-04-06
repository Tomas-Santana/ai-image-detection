from typing import Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.clip import CLIPViTWithDFGM, CLIPViT
from networks.layers import ViTLevelFusion, FuseFormer
from networks.cooi import ViTCOOI


class Patch5Model(nn.Module):
    def __init__(self, dfgm_dim: int = 256, fuseformer_layers: int = 2):
        super().__init__()
        self.mid_dims = 256

        # Rama global — CLIP con DFGM entrenable
        self.clip_global = CLIPViTWithDFGM(
            model_name="ViT-B-16", pretrained="openai", d_mid=dfgm_dim
        )

        # Rama local — CLIP completamente frozen (sin DFGM, sin gradientes)
        self.clip_local = CLIPViT(model_name="ViT-B-16", pretrained="openai", frozen=True)

        self.COOI   = ViTCOOI()
        self.fusion = ViTLevelFusion(self.mid_dims)

        # FuseFormer — fusiona los 12 CLS tokens post-DFGM
        self.fuse_former = FuseFormer(
            d_model=768, n_layers=fuseformer_layers, out_dim=self.mid_dims
        )

        # Proyección local
        self.fc1_local = nn.Linear(768, self.mid_dims)
        self.ac        = nn.ReLU()

        # Clasificador final: 1 CLS + 1 global + 6 local = 8 tokens
        self.cls_token     = nn.Parameter(torch.randn(1, 1, self.mid_dims))
        self.seq_pos_embed = nn.Parameter(torch.randn(1, 8, self.mid_dims))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.mid_dims, nhead=4,
            dim_feedforward=self.mid_dims, dropout=0.1,
            activation="relu", batch_first=True,
        )
        self.mha_list = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(self.mid_dims, 1)

    def forward(
        self, input_img: torch.Tensor, cropped_img: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        amp        = cast(Any, torch.amp)
        use_amp    = cropped_img.device.type == "cuda"
        batch_size = cropped_img.shape[0]

        # ── Rama global (CLIP + DFGM + FuseFormer) ──────────────────────────
        spatial_maps, _, all_cls = self.clip_global(cropped_img)
        early, mid, late = spatial_maps

        with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            fused_global_maps = self.fusion(early, mid, late)              # [B, 256, 14, 14]
            global_embedding  = self.fuse_former(all_cls).unsqueeze(1)     # [B, 1, 256]

            # ── Selección de parches locales via COOI ────────────────────────
            input_loc, _ = self.COOI.get_coordinates(fused_global_maps.detach(), scale)
            proposal_size = input_loc.size(1)   # 6
            window_imgs   = input_img.new_zeros(batch_size, proposal_size, 3, 224, 224)

            for b in range(batch_size):
                for p in range(proposal_size):
                    t, left, bot, r = input_loc[b, p]
                    patch = input_img[b][:, t:bot, left:r]
                    _, ph, pw = patch.size()
                    if ph == 224 and pw == 224:
                        window_imgs[b, p] = patch
                    else:
                        window_imgs[b, p:p+1] = F.interpolate(
                            patch[None], size=(224, 224),
                            mode="bilinear", align_corners=True,
                        )

            window_imgs = window_imgs.reshape(
                batch_size * proposal_size, 3, 224, 224
            ).to(fused_global_maps.device)

        # ── Rama local (CLIP frozen, sin gradientes) ─────────────────────────
        with torch.no_grad():
            _, local_cls_tokens = self.clip_local(window_imgs)
            _, _, local_cls     = local_cls_tokens   # [B*6, 768]

        with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            local_embedding = self.ac(self.fc1_local(local_cls))
            local_embedding = local_embedding.view(batch_size, proposal_size, self.mid_dims)  # [B, 6, 256]

            # ── Fusión final ─────────────────────────────────────────────────
            cls_tok = self.cls_token.expand(batch_size, -1, -1)            # [B, 1, 256]
            seq     = torch.cat((cls_tok, global_embedding, local_embedding), dim=1)  # [B, 8, 256]
            seq     = seq + self.seq_pos_embed
            seq     = self.mha_list(seq)
            return self.fc(seq[:, 0])                                       # [B, 1]