#!/usr/bin/env python3
"""融合模型共享模块 (单一事实来源)。
FusionModel V2: 特征分支独立 MLP + 序列分支投影 + 分类头。
提供 FocalLoss 供可选。
"""
import torch, torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    """V2: BiLSTM 序列分支(128维表示 -> 64维投影) + 62维特征分支(独立 MLP -> 64维) -> 拼接分类头。"""
    def __init__(self, n_seq=6, hidden=64, n_layers=2, feat_dim=62, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.seq_proj = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU())
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def lstm_out(self, x, L):
        B, T, _ = x.shape
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        return out

    def seq_repr(self, x, L):
        B, T, _ = x.shape
        A = self.lstm_out(x, L)
        fwd = A[torch.arange(B), L - 1, :self.hidden]
        bwd = A[:, 0, self.hidden:]
        last = torch.cat([fwd, bwd], dim=1)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < L.unsqueeze(1).to(x.device)
        maxp = A.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return last + maxp

    def forward(self, x, L, f):
        sr = self.seq_proj(self.seq_repr(x, L))   # (B,64)
        ff = self.feat_mlp(f)                      # (B,64)
        return self.head(torch.cat([sr, ff], dim=1))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
