import torch
import torch.nn as nn
import torch.nn.functional as F


class PerJetMLP(nn.Module):
    """Shared-weight MLP applied independently to each jet."""

    def __init__(self, in_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_jets, in_dim) -> (batch, n_jets, embed_dim)"""
        return self.net(x)


class AttentionPool(nn.Module):
    """Learned attention-weighted pooling over a variable-length set of jets."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, jets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        jets:  (B, n_jets, D)
        mask:  (B, n_jets) True = padded / unavailable
        Returns: (B, D)
        """
        logits = self.score(jets).squeeze(-1)            # (B, n_jets)
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))
        weights = F.softmax(logits, dim=-1)               # (B, n_jets)
        return (weights.unsqueeze(-1) * jets).sum(dim=1)  # (B, D)


class UnifiedScorer(nn.Module):
    """Single MLP that scores all3 methods from pooled jet representations.

    Input: 3 method attn-pooled embeddings + 1 global pooled embedding
           + event features + availability flags  →  3 raw logits.
    """

    def __init__(self, embed_dim: int, event_feat_dim: int,
                 mlp_hidden: tuple, dropout: float = 0.1):
        super().__init__()
        in_dim = 4 * embed_dim + event_feat_dim
        layers = []
        prev = in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.head = nn.Sequential(*layers)

    def forward(self, method_embs: torch.Tensor, global_emb: torch.Tensor,
                event: torch.Tensor) -> torch.Tensor:
        """
        method_embs: (B, 3, D) — attn-pooled [resolved, semi, merged]
        global_emb:  (B, D)    — mean-pooled over ALL jets
        event:       (B, event_feat_dim)
        Returns:     (B, 3) raw logits
        """
        B = method_embs.shape[0]
        flat_methods = method_embs.reshape(B, -1)
        x = torch.cat([flat_methods, global_emb, event], dim=-1)
        return self.head(x)


class HHSelector(nn.Module):
    """Unified topology scorer using attention-pooled method representations
    and a global event representation.

    Feeds ALL jets in the event to the encoders, then creates:
      - Per-method representations via attention pooling over algorithm-assigned jets
      - A global representation via mean pooling over ALL jets
      - A single scorer that compares all methods jointly
    """

    def __init__(
        self,
        ak4_feat_dim: int = 7,
        ak8_feat_dim: int = 7,
        event_feat_dim: int = 10,
        embed_dim: int = 128,
        n_heads: int = 4,
        mlp_hidden: tuple = (256, 128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ak4_encoder = PerJetMLP(ak4_feat_dim, embed_dim, dropout)
        self.ak8_encoder = PerJetMLP(ak8_feat_dim, embed_dim, dropout)

        # Per-method attention pooling
        self.res_attn = AttentionPool(embed_dim)
        self.semi_ak8_attn = AttentionPool(embed_dim)
        self.semi_ak4_attn = AttentionPool(embed_dim)
        self.mrg_attn = AttentionPool(embed_dim)

        # Unified scorer
        self.scorer = UnifiedScorer(embed_dim, event_feat_dim, mlp_hidden, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _gather(self, embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather jet embeddings by integer index, clamping out-of-range to 0.

        embeddings: (B, max_jets, D)
        indices:    (B, n_method)
        Returns:    (B, n_method, D)
        """
        safe_idx = indices.clamp(min=0).clamp(max=embeddings.shape[1] - 1)
        idx_exp = safe_idx.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
        return embeddings.gather(1, idx_exp)

    def forward(
        self,
        ak4: torch.Tensor,
        ak4_mask: torch.Tensor,
        ak8: torch.Tensor,
        ak8_mask: torch.Tensor,
        event: torch.Tensor,
        resolved_idx: torch.Tensor,
        semi_idx: torch.Tensor,
        merged_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        ak4:          (B, max_ak4, ak4_feat_dim)
        ak4_mask:     (B, max_ak4)  True = padded
        ak8:          (B, max_ak8, ak8_feat_dim)
        ak8_mask:     (B, max_ak8)  True = padded
        event:        (B, event_feat_dim)
        resolved_idx: (B, 4) int — [h1b1, h1b2, h2b1, h2b2] indices into ak4
        semi_idx:     (B, 3) int — [fatjet(ak8), resb1(ak4), resb2(ak4)]
        merged_idx:   (B, 2) int — [H1(ak8), H2(ak8)]

        Returns: (B, 3) raw logits — [resolved, semi, merged]
        """
        # ── encode ALL jets ────────────────────────────────────────────────
        ak4_emb = self.ak4_encoder(ak4)   # (B, max_ak4, D)
        ak8_emb = self.ak8_encoder(ak8)   # (B, max_ak8, D)

        # Zero out padded positions
        ak4_emb = ak4_emb * (~ak4_mask).unsqueeze(-1).to(ak4_emb.dtype)
        ak8_emb = ak8_emb * (~ak8_mask).unsqueeze(-1).to(ak8_emb.dtype)

        # ── global representation (mean of ALL jets) ───────────────────────
        ak4_valid = (~ak4_mask).float().unsqueeze(-1)  # (B, max_ak4, 1)
        ak8_valid = (~ak8_mask).float().unsqueeze(-1)  # (B, max_ak8, 1)

        ak4_count = ak4_valid.sum(dim=1).clamp(min=1)   # (B, 1)
        ak8_count = ak8_valid.sum(dim=1).clamp(min=1)   # (B, 1)

        global_emb = (
            (ak4_emb * ak4_valid).sum(dim=1) / ak4_count
            + (ak8_emb * ak8_valid).sum(dim=1) / ak8_count
        ) / 2  # (B, D)

        # ── per-method attention-pooled representations ────────────────────

        # Resolved: 4 AK4 jets
        res_jets = self._gather(ak4_emb, resolved_idx)   # (B, 4, D)
        res_pooled = self.res_attn(res_jets)              # (B, D)

        # Semi-resolved: 1 AK8 + 2 AK4
        semi_ak8 = self._gather(ak8_emb, semi_idx[:, :1])  # (B, 1, D)
        semi_ak4 = self._gather(ak4_emb, semi_idx[:, 1:])  # (B, 2, D)
        semi_pooled = (
            self.semi_ak8_attn(semi_ak8) + self.semi_ak4_attn(semi_ak4)
        )  # (B, D)

        # Merged: 2 AK8 jets
        mrg_jets = self._gather(ak8_emb, merged_idx)     # (B, 2, D)
        mrg_pooled = self.mrg_attn(mrg_jets)              # (B, D)

        # ── unified scoring ────────────────────────────────────────────────
        method_embs = torch.stack(
            [res_pooled, semi_pooled, mrg_pooled], dim=1
        )  # (B, 3, D)

        return self.scorer(method_embs, global_emb, event)  # (B, 3) logits
