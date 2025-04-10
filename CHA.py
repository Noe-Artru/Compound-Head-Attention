import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

def scaled_dot_product_CHA(Q, K, V, is_causal=False):
    """
    Based on https://medium.com/@maxshapp/grouped-query-attention-gqa-explained-with-code-e56ee2a1df5a

    Q, K, V: 4 dimensional, corresponding to batch_size, sequence_length, num_heads, embed_dim
    mask: 3 dimensional, corresponding to batch_size, sequence_length, sequence_length
    All batch, embed_dim values should be the same
    Query sequence length should be a multiple of key sequence length
    Sequence length and num_heads should be the same for K and V
    """
    if not Q.ndim == K.ndim == V.ndim == 4:
        raise ValueError(
            f"Expected query, key, and value to be 4-dimensional, but got shapes "
            f"{Q.shape}, {K.shape}, and {V.shape}."
        )
    
    num_groups = Q.shape[2] // K.shape[2]

    # For efficient computation:
    Q = rearrange(Q, "b n h d -> b h n d")
    K = rearrange(K, "b s h d -> b h s d")
    V = rearrange(V, "b s h d -> b h s d")

    bq, hq, nq, dq = Q.shape
    bk, hk, nk, dk = K.shape
    bv, hv, nv, dv = V.shape

    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            "Expected query, key, and value to have the same batch size (dim=0) and "
            f"embedding dimension (dim=3), but got query: {Q.shape}, "
            f"key: {K.shape}, and value: {V.shape}."
        )
    elif (hk != hv) or (nk != nv):
        raise ValueError(
            "Expected key and value to have the same size in dimensions 1 and 2, but "
            f"got key: {K.shape} and value: {V.shape}."
        )
    elif hq % hk != 0:
        raise ValueError(
            "Expected query heads to be a multiple of key/value heads, but got "
            f"query: {Q.shape} and key/value: {K.shape}."
        )

    # Spliting queries into groups
    Q = rearrange(Q, "b (h g) n d -> b g h n d", g=num_groups)
    
    # Computing attention
    scores = einsum(Q, K, "b g h n d, b h s d -> b g h n s")
    scale = Q.shape[-1] ** -0.5

    if is_causal:
        mask = torch.ones((scores.shape[0], scores.shape[3], scores.shape[4]), dtype=torch.bool).tril_().to(scores.device)
        mask = rearrange(mask, "b n s -> b () () n s")
        scores.masked_fill_(~mask, torch.finfo(scores.dtype).min) # instead of float('-inf') to avoid type issues

    attention = F.softmax(scores / scale, dim=-1)

    # Final output
    out = einsum(attention, V, "b g h n s, b h s d -> b g h n d")
    out = rearrange(out, "b g h n d -> b n (h g) d")
    return out

class CompoundHeadAttention(nn.Module):
    def __init__(self, embed_dim, query_heads, kv_heads, is_causal=False, layer_norm=False, device=None, dtype=None):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.is_causal = is_causal
        self.layer_norm = layer_norm
        kv_embed_dim = embed_dim // query_heads * kv_heads

        # Projection matrices
        self.Q_proj = nn.Linear(embed_dim, kv_embed_dim, device=device, dtype=dtype)
        self.K_proj = nn.Linear(embed_dim, kv_embed_dim, device=device, dtype=dtype)
        self.V_proj = nn.Linear(embed_dim, kv_embed_dim, device=device, dtype=dtype)
        self.G = [
            nn.Linear(embed_dim // query_heads, embed_dim // kv_heads, device=device, dtype=dtype) 
            for _ in range(self.kv_heads)
        ]
        self.norm = None
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6, device=device, dtype=dtype)

        self.FC = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    def forward(self, q, k, v):
        # Input shape of x: (b n d)
        Q = self.Q_proj(q)
        K = self.K_proj(k)
        V = self.V_proj(v)

        # Main mechanism of CHA:
        Q = rearrange(Q, "b n (h d) -> b n h d", h=self.kv_heads) 
        Q = torch.stack([self.G[i](Q[:, :, i, :]) for i in range(self.kv_heads)], dim=2)
        Q = rearrange(Q, "b n h (g d) -> b n (h g) d", g=self.query_heads//self.kv_heads)

        K = rearrange(K, "b n (h d) -> b n h d", h=self.kv_heads)
        V = rearrange(V, "b n (h d) -> b n h d", h=self.kv_heads)

        x = scaled_dot_product_CHA(Q, K, V, is_causal=self.is_causal)
        x = rearrange(x, "b n h d -> b n (h d)")

        if(self.layer_norm):
            x = self.norm(x)

        x = self.FC(x)
        return x