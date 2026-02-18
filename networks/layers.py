import torch
import torch.nn as nn


class SA_layer(nn.Module):
    def __init__(self, dim: int = 128, head_size: int = 4):
        super(SA_layer, self).__init__()
        self.mha = nn.MultiheadAttention(dim, head_size)
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ac = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, len_size, fea_dim]
        batch_size, len_size, fea_dim = x.shape
        x = torch.transpose(x, 1, 0)
        y, _ = self.mha(x, x, x)
        x = self.ln1(x + y)
        x = torch.transpose(x, 1, 0)
        x = x.reshape(batch_size * len_size, fea_dim)
        x = x + self.fc2(self.ac(self.fc1(x)))
        x = x.reshape(batch_size, len_size, fea_dim)
        x = self.ln2(x)
        return x


class MultiLevelFusion(nn.Module):
    def __init__(self, mid_dim: int = 128):
        super(MultiLevelFusion, self).__init__()
        self.mid_dim = mid_dim
        self.project_high = nn.Linear(2048, mid_dim)
        self.project_shallow = nn.Linear(64, mid_dim)
        self.project_middle = nn.Linear(512, mid_dim)

        self.mha_list = nn.Sequential(
            SA_layer(mid_dim, 4), SA_layer(mid_dim, 4), SA_layer(mid_dim, 4)
        )

    def forward(
        self, shallow_layers: torch.Tensor, high_layers: torch.Tensor
    ) -> torch.Tensor:
        # shallow_layers: [Bs, 64, 112, 112]
        # high_layers: [Bh, 2048, 7, 7]
        Bs, Cs, Hs, Ws = shallow_layers.size()
        Bh, Ch, Hh, Wh = high_layers.size()

        shallow_layers = shallow_layers.view(Bs, Cs, -1)  # [16,64,112*112]
        shallow_layers = shallow_layers.transpose(1, 2)  # [16,112*112,64]
        shallow_vecs = self.project_shallow(
            shallow_layers.reshape(-1, Cs)
        )  # [16*112*112, 128]
        shallow_vecs = shallow_vecs.view(Bs, -1, self.mid_dim)  # [16,112*112,128]
        shallow_vecs = shallow_vecs.transpose(1, 2)  # [16,128,112*112]
        shallow_vecs = shallow_vecs.view(Bs, self.mid_dim, Hs, Ws)  # [16,128,112,112]
        shallow_patches = (
            shallow_vecs.unfold(3, 16, 16).unfold(2, 16, 16).permute(0, 1, 2, 3, 5, 4)
        )  # [16,128,7,7,16,16]
        shallow_patches = shallow_patches.reshape(
            Bs, self.mid_dim, 49, 256
        )  # [16,128,49,256]

        high_layers = high_layers.view(Bh, Ch, -1)  # [16,2048,7*7]
        high_layers = high_layers.transpose(1, 2)  # [16,7*7,2048]
        high_vecs = self.project_high(high_layers.reshape(-1, Ch))  # [16*7*7,128]
        high_vecs = high_vecs.view(Bh, -1, self.mid_dim)  # [16, 7*7, 128]
        high_vecs = high_vecs.transpose(1, 2)  # [16, 128, 7*7]
        high_patches = high_vecs.view(Bh, self.mid_dim, -1, 1)  # [16, 128, 49, 1]

        all_patches = torch.cat((high_patches, shallow_patches), 3)  # [16,128,49,257]
        all_patches = all_patches.transpose(1, 2)  # [16,49,128,273]
        all_patches = all_patches.reshape(Bh * 49, self.mid_dim, 257)
        all_patches = all_patches.transpose(1, 2)  # [16*49,273,128]
        all_embedding = self.mha_list(all_patches)  # [16*49, 273, 128]
        all_embedding = all_embedding[:, -1]  # [16*49,128]
        fused_feature_maps = all_embedding.reshape(Bh, -1, self.mid_dim)  # [16,49,128]
        fused_feature_maps = fused_feature_maps.transpose(1, 2)  # [16,128,49]
        fused_feature_maps = fused_feature_maps.reshape(
            Bh, self.mid_dim, Hh, Wh
        )  # [16,128,7,7]
        return fused_feature_maps  # [16,128,7,7]
