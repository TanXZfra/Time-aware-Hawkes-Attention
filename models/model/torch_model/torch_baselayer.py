import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # small change here -- we use "1" for masked element
        scores = scores.masked_fill(mask > 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class ScaledSoftplus(nn.Module):
    '''
    Use different beta for mark-specific intensities
    '''
    def __init__(self, num_marks, threshold=20.):
        super(ScaledSoftplus, self).__init__()
        self.threshold = threshold
        self.log_beta = nn.Parameter(torch.zeros(num_marks), requires_grad=True)  # [num_marks]

    def forward(self, x):
        '''
        :param x: [..., num_marks]
        '''
        beta = self.log_beta.exp()
        beta_x = beta * x
        return torch.where(
            beta_x <= self.threshold,
            torch.log1p(beta_x.clamp(max=math.log(1e5)).exp()) / beta,
            x,  # if above threshold, then the transform is effectively linear
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear

        if output_linear:
            self.linears = nn.ModuleList(
                [nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask, output_weight=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            if output_weight:
                return self.linears[-1](x), attn_weight
            else:
                return self.linears[-1](x)
        else:
            if output_weight:
                return x, attn_weight
            else:
                return x
            

##first choice
class HawkesAttention4(nn.Module):
    """
    Inputs:
      q:    (B, L, d_model) query embeddings (event embeddings projected)
      k:    (B, L, d_model) key embeddings
      v:    (B, L, d_model) value embeddings
      t_in: (B, L)          timestamps (scalar)
      c:    (B, L)          event type in [1..num_types]
    Output:
      out:  (B, L, d_model) Hidden representations h(t) for each token
    """
    def __init__(self, num_types, n_head, d_model, d_k, d_v,
                 phi_width, phi_depth, dropout=0.1, normalize_before=True):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.scale = math.sqrt(d_k) # or d_k**0.0.5
        self.normalize_before = normalize_before
        self.dropout = nn.Dropout(dropout)
        self.num_types = num_types
        self.phi_collector=None

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        def build_phi():
            layers = []
            dim = 1 
            for _ in range(phi_depth):
                linear = nn.Linear(dim, phi_width)
                nn.init.xavier_uniform_(linear.weight) 
                layers += [linear, nn.GELU(), nn.Dropout(dropout)]
                dim = phi_width

            final_linear = nn.Linear(dim, 1)
            nn.init.xavier_uniform_(final_linear.weight)
            layers.append(final_linear)

            return nn.Sequential(*layers)

        self.phi_dict = nn.ModuleDict({
            str(c): nn.ModuleList([build_phi() for _ in range(n_head)])
            for c in range(0, num_types)
        })

    def forward(self, q, k, v, t_in, c, mask=None):
        # print("HAWKES!!!")
        B, L, _ = q.size()
        H=self.n_head
        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        qh = self.w_qs(q).view(B, L, H, self.d_k).transpose(1,2)  # (B,H,L,d_k)
        kh = self.w_ks(k).view(B, L, H, self.d_k).transpose(1,2)
        vh = self.w_vs(v).view(B, L, H, self.d_v).transpose(1,2)

        t_query = t_in.unsqueeze(2)
        t_key = t_in.unsqueeze(1)
        delta = t_query - t_key
        max_delta_t = delta.max().item()
        
        # build index tensor (B, L)
        c_index = c.long()
        
        # padding mask (B, L)
        padding_mask = (c == self.num_types)
        
        # index tensor for key and query
        c_i = c_index.unsqueeze(2).expand(-1, -1, L)  # (B, L, L) query
        c_j = c_index.unsqueeze(1).expand(-1, L, -1)  # (B, L, L) eky
        
        # exclude padding
        valid_mask = (~padding_mask.unsqueeze(2)) & (~padding_mask.unsqueeze(1))

        phiQ = torch.zeros(B, H, L, L, device=q.device)
        phiK = torch.zeros(B, H, L, L, device=q.device)
        
        for h in range(self.n_head):
            
            unique_types = torch.unique(c_index)
            unique_types = unique_types[unique_types != self.num_types]  # exclude padding type
            
            # only compute at the entries applicable to this type_val
            for type_val in unique_types:
                type_str = str(type_val.item())

                phi_net = self.phi_dict[type_str][h]

                type_mask_i = (c_i == type_val) & valid_mask
                type_mask_j = (c_j == type_val) & valid_mask
                
                if type_mask_i.any():
                    # get all delta t entries having type_val event as query and input to phi_type_val
                    delta_i = delta[type_mask_i].unsqueeze(-1)
                    phi_i = phi_net(delta_i).squeeze(-1)
                    
                    phiQ_h = torch.zeros(B, L, L, device=q.device)
                    phiQ_h[type_mask_i] = phi_i
                    phiQ[:, h] += phiQ_h
                
                if type_mask_j.any():
                    # similary to key
                    delta_j = delta[type_mask_j].unsqueeze(-1)
                    phi_j = phi_net(delta_j).squeeze(-1)

                    phiK_h = torch.zeros(B, L, L, device=q.device)
                    phiK_h[type_mask_j] = phi_j
                    phiK[:, h] += phiK_h

        # q_mod[i,j] = qh[...,j,:] * phiQ[...,i,j]
        q_mod = qh.unsqueeze(3) * phiQ.unsqueeze(-1)        # (B,H,L,L,d_k)
        k_mod = kh.unsqueeze(2) * phiK.unsqueeze(-1)        # (B,H,L,L,d_k)
        v_mod = vh.unsqueeze(2) * phiK.unsqueeze(-1)        # (B,H,L,L,d_v)

        scores = (q_mod * k_mod).sum(-1) / self.scale  # (B, H, L, L)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))
        #print(mask)

        attn = F.softmax(scores, dim=-1)
        #visualize_attention(attn, title="Attention Heatmap_masked")
        attn = self.dropout(attn)

        # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        out_heads = torch.einsum('b h i j, b h i j d -> b h i d', attn, v_mod)

        # merge and final projection, residual
        out = out_heads.transpose(1,2).contiguous().view(B, L, -1)  # (B,L,H*d_v)
        out = self.dropout(self.fc(out))                           # (B,L,d_model)
        out = out + residual
        if not self.normalize_before:
            out = self.layer_norm(out)



        if self.phi_collector is not None:

            delta_positive = delta[delta > 0].detach().cpu().numpy()
            t_positive = t_in[t_in > 0].detach().cpu().numpy()
            
            if 'delta_positive' not in self.phi_collector:
                self.phi_collector['delta_positive'] = delta_positive
            else:
                self.phi_collector['delta_positive'] = np.concatenate(
                    [self.phi_collector['delta_positive'], delta_positive]
                )

            if 't_positive' not in self.phi_collector:
                self.phi_collector['t_positive'] = t_positive
            else:
                self.phi_collector['t_positive'] = np.concatenate(
                    [self.phi_collector['t_positive'], t_positive]
                )

            for ty_str, phi_nets in self.phi_dict.items():
                ty = int(ty_str)
                for h in range(H):
                    if (ty, h) not in self.phi_collector:
                        self.phi_collector[(ty, h)] = {'phi_net': phi_nets[h], 'max_delta_t': max_delta_t}
                    else:
                        self.phi_collector[(ty, h)]['max_delta_t'] = max(self.phi_collector[(ty, h)]['max_delta_t'], max_delta_t)


        return out


class SublayerConnection(nn.Module):
    # used for residual connection
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            x = self.self_attn(x, x, x, mask)
            if self.feed_forward is not None:
                return self.feed_forward(x)
            else:
                return x


class TimePositionalEncoding(nn.Module):
    """Temporal encoding in THP, ICML 2020
    """

    def __init__(self, d_model, max_len=5000, device='cpu'):
        super().__init__()
        i = torch.arange(0, d_model, 1, device=device)
        div_term = (2 * (i // 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        """Compute time positional encoding defined in Equation (2) in THP model.

        Args:
            x (tensor): time_seqs, [batch_size, seq_len]

        Returns:
            temporal encoding vector, [batch_size, seq_len, model_dim]

        """
        result = x.unsqueeze(-1) * self.div_term
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result


class TimeShiftedPositionalEncoding(nn.Module):
    """Time shifted positional encoding in SAHP, ICML 2020
    """

    def __init__(self, d_model, max_len=5000, device='cpu'):
        super().__init__()
        # [max_len, 1]
        position = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        # [model_dim //2 ]
        div_term = (torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model)).exp()

        self.layer_time_delta = nn.Linear(1, d_model // 2, bias=False)

        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

    def forward(self, x, interval):
        """

        Args:
            x: time_seq, [batch_size, seq_len]
            interval: time_delta_seq, [batch_size, seq_len]

        Returns:
            Time shifted positional encoding defined in Equation (8) in SAHP model

        """
        phi = self.layer_time_delta(interval.unsqueeze(-1))
        aa = len(x.size())
        if aa > 1:
            length = x.size(1)
        else:
            length = x.size(0)

        arc = (self.position[:length] * self.div_term).unsqueeze(0)

        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


class GELU(nn.Module):
    """GeLu activation function
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Identity(nn.Module):

    def forward(self, inputs):
        return inputs


def activation_layer(act_name):
    """Construct activation layers
    Args:
        act_name: str or nn.Module, name of activation function
    Return:
        act_layer: activation layer

    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == 'gelu':
            act_layer = GELU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class DNN(nn.Module):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
        The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
        For instance, for a 2D input with shape ``(batch_size, input_dim)``,
        the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_size**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_size, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_size) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_size = [inputs_dim] + list(hidden_size)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_size[i], hidden_size[i + 1]) for i in range(len(hidden_size) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_size[i + 1]) for i in range(len(hidden_size) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for i in range(len(hidden_size) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input
    

class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out
