import torch
import torch.nn as nn


class MultiLevelFusion(nn.Module):
    def __init__(self, mid_dim: int = 128):
        super(MultiLevelFusion, self).__init__()
        self.mid_dim = mid_dim
        self.project_high = nn.Linear(2048, mid_dim)
        self.project_shallow = nn.Linear(64, mid_dim)
        self.project_middle = nn.Linear(512, mid_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mid_dim,
            nhead=4,
            dim_feedforward=mid_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.mha_list = nn.TransformerEncoder(encoder_layer, num_layers=3)

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
    
class ThreeLevelFusion(nn.Module):
    def __init__(self, mid_dim: int = 128):
        super(ThreeLevelFusion, self).__init__()
        self.mid_dim = mid_dim
        self.project_high = nn.Linear(2048, mid_dim)
        self.project_shallow = nn.Linear(64, mid_dim)
        self.project_middle = nn.Linear(512, mid_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mid_dim,
            nhead=4,
            dim_feedforward=mid_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.mha_list = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(
        self,
        shallow_layers: torch.Tensor,
        middle_layers: torch.Tensor,
        high_layers: torch.Tensor,
    ) -> torch.Tensor:
        # shallow_layers: [Bs, 64, 112, 112]
        # middle_layers: [Bm, 512, 28, 28]
        # high_layers: [Bh, 2048, 7, 7]

        Bs, Cs, Hs, Ws = shallow_layers.size()
        Bm, Cm, Hm, Wm = middle_layers.size()
        Bh, Ch, Hh, Wh = high_layers.size()

        shallow_layers = shallow_layers.view(Bs, Cs, -1)  # [B, 64, 112*112]
        shallow_layers = shallow_layers.transpose(1, 2)  # [B, 112*112, 64]
        shallow_vecs = self.project_shallow(
            shallow_layers.reshape(-1, Cs)
        )  # [B*112*112, mid_dim]
        shallow_vecs = shallow_vecs.view(Bs, -1, self.mid_dim)  # [B, 112*112, mid_dim]
        shallow_vecs = shallow_vecs.transpose(1, 2)  # [B, mid_dim, 112*112]
        shallow_vecs = shallow_vecs.view(Bs, self.mid_dim, Hs, Ws)  # [B, mid_dim, 112, 112]
        shallow_patches = (
            shallow_vecs.unfold(3, 16, 16).unfold(2, 16, 16).permute(0, 1, 2, 3, 5, 4)
        )  # [B, mid_dim, 7, 7, 16, 16]
        shallow_patches = shallow_patches.reshape(
            Bs, self.mid_dim, 49, 256
        )  # [B, mid_dim, 49, 256]

        middle_layers = middle_layers.view(Bm, Cm, -1)  # [B, 512, 28*28]
        middle_layers = middle_layers.transpose(1, 2)  # [B, 28*28, 512]
        middle_vecs = self.project_middle(
            middle_layers.reshape(-1, Cm)
        )  # [B*28*28, mid_dim]
        middle_vecs = middle_vecs.view(Bm, -1, self.mid_dim)  # [B, 28*28, mid_dim]
        middle_vecs = middle_vecs.transpose(1, 2)  # [B, mid_dim, 28*28]
        middle_vecs = middle_vecs.view(Bm, self.mid_dim, Hm, Wm)  # [B, mid_dim, 28, 28]
        middle_patches = (
            middle_vecs.unfold(3, 4, 4).unfold(2, 4, 4).permute(0, 1, 2, 3, 5, 4)
        )  # [B, mid_dim, 7, 7, 4, 4]
        middle_patches = middle_patches.reshape(
            Bm, self.mid_dim, 49, 16
        )  # [B, mid_dim, 49, 16]

        high_layers = high_layers.view(Bh, Ch, -1)  # [B, 2048, 7*7]
        high_layers = high_layers.transpose(1, 2)  # [B, 7*7, 2048]
        high_vecs = self.project_high(high_layers.reshape(-1, Ch))  # [B*7*7, mid_dim]
        high_vecs = high_vecs.view(Bh, -1, self.mid_dim)  # [B, 7*7, mid_dim]
        high_vecs = high_vecs.transpose(1, 2)  # [B, mid_dim, 7*7]
        high_patches = high_vecs.view(Bh, self.mid_dim, -1, 1)  # [B, mid_dim, 49, 1]

        all_patches = torch.cat(
            (high_patches, middle_patches, shallow_patches), 3
        )  # [B, mid_dim, 49, 273]
        all_patches = all_patches.transpose(1, 2)  # [B, 49, 273, mid_dim]
        all_patches = all_patches.reshape(Bh * 49, 273, self.mid_dim)  # [B*49, 273, mid_dim]
        all_embedding = self.mha_list(all_patches)  # [B*49, 273, mid_dim]
        all_embedding = all_embedding[:, -1]  # [B*49, mid_dim]
        fused_feature_maps = all_embedding.reshape(Bh, -1, self.mid_dim)  # [B, 49, mid_dim]
        fused_feature_maps = fused_feature_maps.transpose(1, 2)  # [B, mid_dim, 49]
        fused_feature_maps = fused_feature_maps.reshape(
            Bh, self.mid_dim, Hh, Wh
        )  # [B, mid_dim, 7, 7]
        return fused_feature_maps  # [B, mid_dim, 7, 7]

class ViTLevelFusion(nn.Module):
    def __init__(self, mid_dim: int = 256):
        super(ViTLevelFusion, self).__init__()
        self.mid_dim = mid_dim
        # ViT-B-16 embedding dim is 768. 3 layers concatenated = 2304
        self.project = nn.Linear(768 * 3, mid_dim)
        
        # FIX: Spatial Positional Encoding for the 14x14 grid (196 tokens)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 196, mid_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mid_dim,
            nhead=4,
            dim_feedforward=mid_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.mha_list = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(self, early: torch.Tensor, mid: torch.Tensor, late: torch.Tensor) -> torch.Tensor:
        B, C, H, W = early.shape
        
        # Concatenate channels: [B, 2304, 14, 14]
        x = torch.cat((early, mid, late), dim=1)
        
        # Flatten to sequence for attention: [B, 196, 2304]
        x = x.view(B, C * 3, -1).transpose(1, 2)
        
        # Project down to mid_dim: [B, 196, 256]
        x = self.project(x)
        
        # Inject Spatial Awareness
        x = x + self.spatial_pos_embed
        
        # Self-attention over the 196 spatial tokens
        x = self.mha_list(x)
        
        # Reshape back to spatial feature map: [B, 256, 14, 14]
        x = x.transpose(1, 2).reshape(B, self.mid_dim, H, W)
        return x

class FuseFormer(nn.Module):
    """Fusiona los 12 CLS tokens de todos los bloques ViT (GFF, sec. 3.4).
    
    Prepende un token CLS_fuse aprendible, aplica Transformer, y usa
    el output de CLS_fuse como representación global final.
    """
    def __init__(self, d_model: int = 768, n_layers: int = 2, out_dim: int = 256):
        super().__init__()
        self.cls_fuse = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 2,   # 1536
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, out_dim)
        self.act  = nn.ReLU()

    def forward(self, all_cls: list) -> torch.Tensor:
        """
        all_cls: lista de 12 tensores, cada uno [B, 768]
        retorna: [B, out_dim]
        """
        assert all_cls[0].dim() == 2, f"Se esperaba [B, 768] pero llegó {all_cls[0].shape}"
        B = all_cls[0].shape[0]
        cls_seq  = torch.stack(all_cls, dim=1)             # [B, 12, 768]
        cls_fuse = self.cls_fuse.expand(B, -1, -1)         # [B, 1, 768]
        seq      = torch.cat([cls_fuse, cls_seq], dim=1)   # [B, 13, 768]
        out      = self.transformer(seq)                   # [B, 13, 768]
        return self.act(self.proj(out[:, 0, :]))            # [B, out_dim]

