import torch
import torch.nn as nn
import torch.nn.functional as F
from GQA import GroupedQueryAttention
from torchscale.component.xpos_relative_position import XPOS

class GQAEncoderLayer(nn.Module):
    def __init__(self, embed_dim, query_heads, kv_heads, feedforward_dim, layer_norm_eps = 1e-5, device=None, dtype=None):
        super().__init__()

        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)

        # GQA Block
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.GQA_attention = GroupedQueryAttention(embed_dim, query_heads, kv_heads, is_causal=False, layer_norm=True, device=device, dtype=dtype)

        # FF Block
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.lin1 = nn.Linear(embed_dim, feedforward_dim, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(feedforward_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.lin2 = nn.Linear(feedforward_dim, embed_dim, device=device, dtype=dtype)
    
    def _GQA_block(self, x_in):
        x = self.norm1(x_in)
        x = self.GQA_attention(x, x, x)
        x = self.dropout(x)
        return x
    
    def _FF_block(self, x_in):
        x = self.norm2(x_in)
        x = self.activation(self.lin1(x))
        x = self.norm3(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x
    
    def forward(self, x_in):
        x = x_in
        x = x + self._GQA_block(x)
        x = x + self._FF_block(x)
        return x
    
class GQADecoderLayer(nn.Module):
    def __init__(self, embed_dim, query_heads, kv_heads, feedforward_dim, layer_norm_eps = 1e-5, device=None, dtype=None):
        super().__init__()

        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)

        # GQA Block
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.GQA_attention = GroupedQueryAttention(embed_dim, query_heads, kv_heads, is_causal=True, layer_norm=False, device=device, dtype=dtype)

        # GQA Cross Block
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.GQA_attention_decoder = GroupedQueryAttention(embed_dim, query_heads, kv_heads, is_causal=True, layer_norm=True, device=device, dtype=dtype)

        # FF Block
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.lin1 = nn.Linear(embed_dim, feedforward_dim, device=device, dtype=dtype)
        self.norm4 = nn.LayerNorm(feedforward_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.lin2 = nn.Linear(feedforward_dim, embed_dim, device=device, dtype=dtype)
    
    def _GQA_block(self, x_in):
        x = self.norm1(x_in)
        x = self.GQA_attention(x, x, x)
        x = self.dropout(x)
        return x
    
    def _GQA_cross_block_(self, x_in, memory):
        x = self.norm2(x_in)
        x = self.GQA_attention_decoder(x, memory, memory)
        x = self.dropout(x)
        return x

    def _FF_block(self, x_in):
        x = self.norm3(x_in)
        x = self.activation(self.lin1(x))
        x = self.norm4(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x
    
    def forward(self, x_in, memory):
        x = x_in
        x = x + self._GQA_block(x)
        x = x + self._GQA_cross_block_(x, memory)
        x = x + self._FF_block(x)
        return x
    

class GQATransformer(nn.Module):
    def __init__(self, embed_dim, query_heads, kv_heads, num_encoder_layers, num_decoder_layers, feedforward_dim, layer_norm_eps = 1e-5, device=None, dtype=None):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=GQAEncoderLayer(embed_dim=embed_dim, query_heads=query_heads, kv_heads=kv_heads, feedforward_dim=feedforward_dim, layer_norm_eps=layer_norm_eps, device=device, dtype=dtype),
            num_layers=num_encoder_layers,
            mask_check=False,
            enable_nested_tensor=False
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=GQADecoderLayer(embed_dim=embed_dim, query_heads=query_heads, kv_heads=kv_heads, feedforward_dim=feedforward_dim, layer_norm_eps=layer_norm_eps, device=device, dtype=dtype),
            num_layers=num_decoder_layers
        )
        
    def forward(self, x_encoder, x_decoder):
        for layer in self.encoder.layers:
            x_encoder = layer(x_encoder)
        mem = x_encoder

        for layer in self.decoder.layers:
            x_decoder = layer(x_decoder, mem)

        return x_decoder
    
class GQATransformerNextWord(nn.Module):
    def __init__(self, num_tokens_enc, num_tokens_dec, embed_dim, query_heads, kv_heads, num_encoder_layers, num_decoder_layers, feedforward_dim, layer_norm_eps = 1e-5, device=None, dtype=None):
        super().__init__()
        self.token_embed_enc = nn.Embedding(num_tokens_enc, embed_dim).to(device=device)
        self.token_embed_dec = nn.Embedding(num_tokens_dec, embed_dim).to(device=device)
        self.pos_embed = XPOS(embed_dim)
        self.transformer = GQATransformer(embed_dim, query_heads, kv_heads, num_encoder_layers, num_decoder_layers, feedforward_dim, layer_norm_eps, device=device, dtype=dtype)
        
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.output_layer = nn.Linear(embed_dim, num_tokens_dec, device=device, dtype=dtype)
    
    def forward(self, src, tgt):
        src_enc = self.token_embed_enc(src)
        x_enc = src_enc + self.pos_embed(src_enc)
        tgt_dec = self.token_embed_dec(tgt)
        x_dec = tgt_dec + self.pos_embed(tgt_dec)
        x = self.transformer(x_enc, x_dec)
        x = self.norm(x)
        x = self.output_layer(x)
        return x
    

if __name__ == "__main__":
    num_tokens = 200
    x = torch.randint(0, num_tokens - 1, size=(2,512))
    model = GQATransformerNextWord(num_tokens_enc=num_tokens, num_tokens_dec=num_tokens, embed_dim=64, query_heads=8, kv_heads=2, num_encoder_layers=3, num_decoder_layers=3, feedforward_dim=128)
    
    with torch.no_grad():
        out = model(x, x)
    print(out.shape)