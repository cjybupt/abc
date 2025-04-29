import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianKernelGenerator(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.sigmas = nn.Parameter(torch.tensor(scales), requires_grad=True)

    def forward(self, distance_matrix):
        A_list = []
        for sigma in self.sigmas:
            A_k = torch.exp(-distance_matrix**2 / (2*sigma**2))
            A_list.append(A_k)
        return A_list


class GeoGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.1):
        super().__init__()
        self.thresholds = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.attn_fc = nn.Linear(2*out_dim, 1)
        self.W = nn.Linear(in_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A_geo, distance_matrix=None):
        """
        X: (B, N, F_in)
        A_geo: (N, N)
        distance_matrix: (N, N)
        """
        B, N, _ = X.size()
        h = self.W(X)  # (B, N, F_out)

        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, F_out)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, F_out)
        h_concat = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, 2F_out)

        e = self.leaky_relu(self.attn_fc(h_concat)).squeeze(-1)  # (B, N, N)

        if distance_matrix is not None:
            distance_matrix = distance_matrix.clone()
            distance_matrix.fill_diagonal_(1.0)
            e = e + (1 / (distance_matrix.unsqueeze(0) + 1e-5))  # (1, N, N) -> (B, N, N)

        # Masking with adjacency
        A_geo = A_geo.to(X.device)
        mask = torch.sigmoid(10 * (A_geo - self.thresholds))  # (N, N), values in [0,1]
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # Apply mask by multiplying
        e = e * mask  # (B, N, N)

        # print(self.thresholds)

        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, h)  # (B, N, F_out)
        return h_prime



class MultiScaleGeoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_scales=2):
        super().__init__()
        self.kernel_gen = GaussianKernelGenerator(scales=[0.2, 1, 3][:n_scales])
        self.geo_gats = nn.ModuleList([
            GeoGATLayer(in_dim, hidden_dim) for _ in range(n_scales)
        ])
        self.fusion = nn.Linear(hidden_dim * n_scales, hidden_dim)

    def forward(self, X_bt, distance_matrix):
        A_list = self.kernel_gen(distance_matrix)  # [N, N] list

        H_list = [
            self.geo_gats[i](X_bt, A_list[i], distance_matrix)
            for i in range(len(A_list))
        ]  # 每个都是 [B*T, N, hidden_dim]

        H_concat = torch.cat(H_list, dim=-1)  # [B*T, N, hidden_dim * n_scales]
        H_fused = self.fusion(H_concat)  # [B*T, N, hidden_dim]
        return H_fused


class GeoSpatialBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_scales=2):
        super().__init__()
        self.encoder = MultiScaleGeoEncoder(in_dim, hidden_dim, n_scales)
        self.geo_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, all_node_feature, distance_matrix):
        """
        all_node_feature: [B, T, N, F]
        distance_matrix: [N, N]
        return: [B, T, N, out_dim]
        """
        B, T, N, F = all_node_feature.shape
        output_seq = []

        for t in range(T):
            X_t = all_node_feature[:, t, :, :]  # [B, N, F]
            X_t = X_t.contiguous().view(B, N, F)

            # [B, N, hidden_dim]
            H_geo = self.encoder(X_t, distance_matrix)  # 支持 B 维批处理
            geo_out_t = self.geo_mlp(H_geo)  # [B, N, out_dim]
            output_seq.append(geo_out_t.unsqueeze(1))  # 插入时间维度

        geo_out = torch.cat(output_seq, dim=1)  # [B, T, N, out_dim]
        return geo_out


