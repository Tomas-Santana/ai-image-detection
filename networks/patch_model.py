import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.clip import CLIPResNet
from networks.layers import MultiLevelFusion
from networks.cooi import COOI


class Patch5Model(nn.Module):
    def __init__(self):
        super(Patch5Model, self).__init__()
        self.clip = CLIPResNet(model_name="RN50", frozen=True)  # debug
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
        self.MultiFusion = MultiLevelFusion(self.mid_dims)

    def forward(
        self, input_img: torch.Tensor, cropped_img: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        x = cropped_img
        batch_size, p, _, _ = x.shape  # [batch_size, 3, 224, 224]
        shallow_global_maps, high_global_maps = self.clip(
            x
        )  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        fused_global_maps = self.MultiFusion(shallow_global_maps, high_global_maps)
        B, C, H, W = fused_global_maps.size()  # [16,128,7,7]

        ## global embeddings
        global_embedding = self.avgpool(fused_global_maps)  # [16, 128,1,1]
        global_embedding = global_embedding.view(
            global_embedding.size(0), -1
        )  # [16,128]
        global_embedding = self.ac(global_embedding)  # [16,128]
        global_embedding = global_embedding.view(-1, 1, self.mid_dims)  # [16, 1, 128]

        input_loc, fps_loc = self.COOI.get_coordinates(
            fused_global_maps.detach(), scale
        )

        _, proposal_size, _ = input_loc.size()

        window_imgs = torch.zeros([batch_size, proposal_size, 3, 224, 224]).to(
            fused_global_maps.device
        )  # [N, 4, 3, 224, 224]

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
        )  # [N*4, 3, 224, 224]
        _, local_maps = self.clip(
            window_imgs.detach()
        )  # [batchsize*self.proposalN, 2048]
        local_embedding = self.avgpool(local_maps)
        local_embedding = local_embedding.view(
            local_embedding.size(0), -1
        )  # [16*4,2048,1,1]

        local_embedding = self.ac(
            self.fc1(local_embedding)
        )  # [batchsize*self.proposalN, 128]
        local_embedding = local_embedding.view(-1, proposal_size, 128)

        all_embeddings = torch.cat(
            (local_embedding, global_embedding), 1
        )  # [1, 1+self.proposalN, 128]
        all_embeddings = self.mha_list(all_embeddings)
        all_logits = self.fc(all_embeddings[:, -1])

        return all_logits
