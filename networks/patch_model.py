from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.clip import CLIPResNet
from networks.resnet import resnet50
from networks.layers import ThreeLevelFusion
from networks.cooi import COOI


class Patch5Model(nn.Module):
    def __init__(self, unfreeze_last_clip_layer: bool = False, backbone: Literal['clip', 'resnet'] = 'clip'):
        super(Patch5Model, self).__init__()
        if backbone == 'clip':
            self.backbone = CLIPResNet(model_name="RN50", frozen=True, unfreeze_last_layer=unfreeze_last_clip_layer)
        else:
            self.backbone = resnet50(pretrained=True)
        self.mid_dims = 128
        self.COOI = COOI()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.mha_list = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc1 = nn.Linear(2048, 128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ThreeFusion = ThreeLevelFusion(self.mid_dims)

    def forward(
        self, input_img: torch.Tensor, cropped_img: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, _, _ = cropped_img.shape  # [batch_size, 3, 224, 224]
        shallow_global_maps, mid_global_maps, high_global_maps = self.backbone(
            cropped_img
        )  # shallow: [B, 64, 112, 112], middle: [B, 512, 28, 28], high: [B, 2048, 7, 7]
        fused_global_maps = self.ThreeFusion(
            shallow_global_maps, mid_global_maps, high_global_maps
        )  # [B, 128, 7, 7]

        # global embedding: [B, 128, 1, 1] -> [B, 128] -> [B, 1, 128]
        global_embedding = self.avgpool(fused_global_maps)
        global_embedding = global_embedding.flatten(1)
        global_embedding = self.ac(global_embedding)  # [16,128]
        global_embedding = global_embedding.view(-1, 1, self.mid_dims)  # [16, 1, 128]

        input_loc, _ = self.COOI.get_coordinates(
            fused_global_maps.detach(), scale
        )

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
                    )  # [N, 4, 3, 224, 224]

        window_imgs = window_imgs.reshape(
            batch_size * proposal_size, 3, 224, 224
        ).to(fused_global_maps.device)  # [B * proposal_size, 3, 224, 224]
        _, _, local_maps = self.backbone(
            window_imgs.detach()
        )  # [B * proposal_size, 2048, 7, 7]
        local_embedding = self.avgpool(local_maps).flatten(1)  # [B * proposal_size, 2048]
        local_embedding = self.ac(self.fc1(local_embedding))  # [B * proposal_size, 128]
        local_embedding = local_embedding.view(-1, proposal_size, 128)

        all_embeddings = torch.cat(
            (local_embedding, global_embedding), 1
        )  # [B, proposal_size + 1, 128]
        all_embeddings = self.mha_list(all_embeddings)
        all_logits = self.fc(all_embeddings[:, -1])

        return all_logits
