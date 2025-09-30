import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding2,DataEmbedding_inverted
import matplotlib.pyplot as plt
import seaborn as sns




class HawkesAttention(nn.Module):
    def __init__(self, new_d_model, n_new_heads, time_dim, tmlp_width, tmlp_depth, activation,dropout=0.1):
        super().__init__()
        assert new_d_model % n_new_heads == 0, "d_model must be divisible by n_heads"
        self.new_d_model = new_d_model          # total embedding dimension
        self.n_new_heads = n_new_heads             # number of attention heads
        self.head_dim = new_d_model // n_new_heads    # dimension per head
        self.time_dim = time_dim                  # time embedding dimension(auto,no need to set)
        self.scale = math.sqrt(self.head_dim)      
        self.dropout= nn.Dropout(dropout)          

        if activation.lower() == "relu":
            Act = nn.ReLU
        elif activation.lower() == "gelu":
            Act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        # Linear projections for Q, K, V (shared across heads)
        self.W_Q = nn.Linear(new_d_model, new_d_model, bias=False)
        self.W_K = nn.Linear(new_d_model, new_d_model, bias=False)
        self.W_V = nn.Linear(new_d_model, new_d_model, bias=False)

        # Factory to build a single-head φ MLP (maps time_dim -> 1 scalar)
        def build_phi_single():
            layers = []
            in_d = time_dim
            # hidden layers
            for _ in range(tmlp_depth):
                layers.append(nn.Linear(in_d, tmlp_width))
                layers.append(Act())
                layers.append(nn.Dropout(dropout))
                in_d = tmlp_width
            # final output layer produces one scalar for this head
            layers.append(nn.Linear(in_d, 1))
            layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        # Create per-head φ_Q, φ_K, φ_V MLP lists
        self.phi_Q = nn.ModuleList([build_phi_single() for _ in range(n_new_heads)])
        self.phi_K = nn.ModuleList([build_phi_single() for _ in range(n_new_heads)])
        self.phi_V = nn.ModuleList([build_phi_single() for _ in range(n_new_heads)])

    def forward(self, Q, K, V, t_Q, t_K, hawkes_self_attn_mask=True):
        """
        Q: [B, Lq, d_model]  Query token embeddings
        K: [B, Lk, d_model]  Key token embeddings
        V: [B, Lk, d_model]  Value token embeddings
        t_Q: [B, Lq, time_dim]  Query time embeddings
        t_K: [B, Lk, time_dim]  Key time embeddings
        t_V: [B, Lk, time_dim]  Value time embeddings
        causal_mask: [Lq, Lk]  Causal mask for self-attention (optional)
        returns out: [B, Lq, d_model]
        """
        B, Lq, _ = Q.shape
        _, Lk, _ = K.shape

        # Q0/K0/V0: [B, Lq/Lk, d_model] -> [B, Lq/Lk, H, head_dim] -> [B, H, Lq/Lk, head_dim]
        Q0 = self.W_Q(Q).view(B, Lq, self.n_new_heads, self.head_dim).transpose(1, 2)
        K0 = self.W_K(K).view(B, Lk, self.n_new_heads, self.head_dim).transpose(1, 2)
        V0 = self.W_V(V).view(B, Lk, self.n_new_heads, self.head_dim).transpose(1, 2)

        # Δt_{j,i}: [B, Lq, Lk, time_dim]
        tQ = t_Q.unsqueeze(2)  # [B, Lq, 1, time_dim]
        tK = t_K.unsqueeze(1)  # [B, 1, Lk, time_dim]
        delta = tQ - tK        # [B, Lq, Lk, time_dim]

        # For each head h, scalars [B, Lq, Lk] -> [B, H, Lq, Lk]
        phiQ = torch.stack([self.phi_Q[h](delta).squeeze(-1) for h in range(self.n_new_heads)], dim=1)
        phiK = torch.stack([self.phi_K[h](delta).squeeze(-1) for h in range(self.n_new_heads)], dim=1)
        phiV = torch.stack([self.phi_V[h](delta).squeeze(-1) for h in range(self.n_new_heads)], dim=1)

        #  [B, H, Lq, Lk, head_dim]
        Q_mod = phiQ.unsqueeze(-1) * Q0.unsqueeze(3)
        K_mod = phiK.unsqueeze(-1) * K0.unsqueeze(2)
        V_mod = phiV.unsqueeze(-1) * V0.unsqueeze(2)

        # [B,H,T,L]
        scores = (Q_mod * K_mod).sum(-1) / self.scale

        # masking for self attention
        if hawkes_self_attn_mask:
            # Causal mask for hawkes self-attention: [Lk, Lq]
            # scores: [B, H, Lq, Lk]
            # causal_mask: [Lk, Lq] -> [1, 1, Lk, Lq]
            # Mask out future positions in self-attention
            # scores: [B, H, Lq, Lk] -> [B, H, Lq, Lk]
            causal_mask = torch.triu(torch.ones(Lq, Lk, dtype=torch.bool, device=scores.device),
                            diagonal=1)     
            mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, float('-inf'))


        attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        # Compute output: [B, H, Lq, head_dim]
        out = (attn.unsqueeze(-1) * V_mod).sum(-2)

        # Merge heads: [B, Lq, d_model]
        out = out.transpose(1, 2).reshape(B, Lq, self.new_d_model)
        return out



class HawkesEncoderLayer(nn.Module):
    """
    This encoder structure comes from the iTransformer paper, with vanilla attention replaced by HawkesAttention
    x' = x_enc + Dropout(att(x_enc, t_enc, t_dec))
    x'' = LayerNorm(x')
    y = Dropout(act(Conv1(x''))) → Dropout(Conv2(y))
     out = LayerNorm(x'' + y)
    """
    def __init__(self, attention,new_d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * new_d_model
        self.attn = attention
        self.norm1 = nn.LayerNorm(new_d_model)
        self.norm2 = nn.LayerNorm(new_d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(new_d_model, d_ff, kernel_size= 1)
        self.conv2 = nn.Conv1d(d_ff, new_d_model, kernel_size= 1)
        self.activation = F.relu if activation.lower()=="relu" else F.gelu

    def forward(self, Q, K, V, t_Q, t_K, hawkes_self_attn_mask=True):
        h = self.attn(Q, K, V, t_Q, t_K, hawkes_self_attn_mask)
        x2 = Q + self.dropout(h)
        x2 = self.norm1(x2)
        y = x2.transpose(1,2)
        y = self.activation(self.conv1(y))
        y = self.dropout(y)
        y = self.conv2(y)                
        y = self.dropout(y).transpose(1,2)      
        return self.norm2(x2 + y)

class Projector(nn.Module):
    def __init__(self, seq_len, pred_len, new_d_model, enc_in):
        super(Projector, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.new_d_model = new_d_model
        self.enc_in = enc_in

        # [B, seq_len, new_d_model] -> [B, pred_len, new_d_model]
        self.map_to_pred_len = nn.Linear(seq_len, pred_len)

        # [B, pred_len, new_d_model] -> [B, pred_len, enc_in]
        self.map_to_enc_in = nn.Linear(new_d_model, enc_in)

    def forward(self, x):
        # x: [B, seq_len, new_d_model]
        x = x.permute(0, 2, 1)  # [B, new_d_model, seq_len]
        x = self.map_to_pred_len(x)  # [B, new_d_model, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, new_d_model]
        x = self.map_to_enc_in(x)  # [B, pred_len, enc_in]
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.class_strategy = configs.class_strategy
        self.d_model=configs.d_model
        self.new_d_model=configs.new_d_model
        self.embed=configs.embed
        self.freq=configs.freq
        self.dropout=configs.dropout
        self.channel_independence = configs.channel_independence

        self.enc_embedding = DataEmbedding2(
            configs.enc_in,          
            configs.new_d_model,           
            embed_type=configs.embed,  
            freq=configs.freq,       
            dropout=configs.dropout  
        )
        
        self.eff_seq_len = self.seq_len

        self.invert_embedding = DataEmbedding_inverted(
            self.eff_seq_len,
            self.d_model,
            self.embed,
            self.freq,
            self.dropout
        )


        self.new_layers=nn.ModuleList([HawkesEncoderLayer(
            attention=HawkesAttention(configs.new_d_model,configs.n_new_heads,configs.time_dim,configs.tmlp_width,configs.tmlp_depth,configs.activation,
                                       configs.dropout
            ),new_d_model=configs.new_d_model,d_ff=configs.d_ff,dropout=configs.dropout,activation=configs.activation)
            for _ in range(configs.num_new_layers)])
        
        self.new_layers_norm= nn.LayerNorm(configs.new_d_model)
        # encoder
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))


        self.projector = nn.Linear(configs.d_model, configs.pred_len)
        
        self.new_projector = Projector(
            seq_len=self.eff_seq_len,
            pred_len=self.pred_len,
            new_d_model=self.new_d_model,
            enc_in=configs.enc_in
        )

        self.time_proj = nn.Linear(configs.time_dim, configs.new_d_model)



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means

            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev


        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, None)
        x_dec = self.enc_embedding(x_dec, None)
        x_dec_tokens = self.time_proj(x_mark_dec)
            
        for layer in self.new_layers:
            enc_out = layer(enc_out, enc_out, enc_out, x_mark_enc,x_mark_enc, hawkes_self_attn_mask=True)
        enc_out = self.new_layers_norm(enc_out)

        dec_out = self.new_projector(enc_out)

        # denormalization
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.channel_independence:
            outs = []
            D = x_enc.shape[-1]
            for i in range(D):
                x_enc_i = x_enc[..., i:i+1]
                x_dec_i = x_dec[..., i:i+1]
                out_i = self.forecast(x_enc_i, x_mark_enc, x_dec_i, x_mark_dec)
                outs.append(out_i)
            out=torch.cat(outs, dim=-1)

            return out
        else:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
