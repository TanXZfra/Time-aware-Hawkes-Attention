import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.model.torch_model.torch_baselayer import  MultiHeadAttention, HawkesAttention4, TimePositionalEncoding, ScaledSoftplus, EncoderLayer
from models.model.torch_model.torch_basemodel import TorchBaseModel


class HawkesTHP(TorchBaseModel):

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(HawkesTHP, self).__init__(model_config)
        self.d_inner = model_config.hidden_size
        self.d_model = model_config.d_model
        self.use_norm = model_config.use_ln

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        self.num_event_types = model_config.num_event_types
        self.d_k = model_config.d_k
        self.d_v = model_config.d_v
        self.d_rnn = model_config.d_rnn
        self.phi_width = model_config.phi_width 
        self.phi_depth = model_config.phi_depth
        self.pad = model_config.pad_token_id
        self.rnn_ornot = model_config.rnn

        # self.layer_temporal_encoding = TimePositionalEncoding(self.d_model, device=self.device)

        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        # convert hidden vectors into event-type-sized vector
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)   # learnable mark-specific beta

        # # Add MLP layer
        # # Equation (5)
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model * 2),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model * 2, self.d_model)
        # )

        # self.stack_layers = nn.ModuleList(
        #     [EncoderLayer(
        #         self.d_model,
        #         MultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
        #                            output_linear=False),
        #         use_residual=False,
        #         feed_forward=self.feed_forward,
        #         dropout=self.dropout
        #     ) for _ in range(self.n_layers)])


        # self.stack_layers = nn.ModuleList(
        #     [EncoderLayer(
        #         self.d_model,
        #         HawkesAttention4(self.num_event_types, self.n_head, self.d_model, self.d_k, self.d_v,
        #                          self.phi_width, self.phi_depth, self.dropout),
        #         use_residual=False,
        #         feed_forward=self.feed_forward,
        #         dropout=self.dropout
        #     ) for _ in range(self.n_layers)])


        # self.stack_layers = nn.ModuleList([
        #     EncoderLayer(model_config,self.d_model, d_inner, n_head, d_k, d_v,num_types,phi_width,phi_depth, 
        #                  dropout=dropout, normalize_before=False,original_THP=original_THP)
        #     for _ in range(self.n_layers)])

        self.encoder = Encoder(
            model_config,
            num_types=self.num_event_types,
            d_model=self.d_model,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            phi_width=self.phi_width,  # width of φ networks
            phi_depth=self.phi_depth,  # depth of φ networks
            dropout=self.dropout
        )
        self.rnn = RNN_layers(self.d_model, self.d_rnn)
        print(f"d_inner",{self.d_inner},"d_model",{self.d_model},"n_layers",{self.n_layers},"n_head",{self.n_head},"d_k",{self.d_k},"d_v",{self.d_v},"phi_width",{self.phi_width},"phi_depth",{self.phi_depth})

    def forward(self, time_seqs, type_seqs, attention_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]

        non_pad_mask = get_non_pad_mask(type_seqs, pad=self.pad)

        enc_output = self.encoder(type_seqs, time_seqs, non_pad_mask)

        if self.rnn_ornot:
            # apply RNN layers if specified
            enc_output = self.rnn(enc_output, non_pad_mask)

        return enc_output
    

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])

        # [batch_size, seq_len, num_event_types]
        # update time decay based on Equation (6)
        # [1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_event_types]
        intensity_states = factor_intensity_decay * time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(
            enc_out) + factor_intensity_base

        lambda_at_event = self.softplus(intensity_states)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample dtimes
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        type_seq=type_seqs[:, 1:])

        # compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]

        # [batch_size, seq_len, num_samples, 1]
        sample_dtimes = sample_dtimes[..., None]

        # [1, 1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = factor_intensity_decay * sample_dtimes + self.layer_intensity_hidden(
            event_states) + factor_intensity_base

        return intensity_states

    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            time_delta_seqs,
                                            type_seqs,
                                            sample_dtimes,
                                            **kwargs):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        attention_mask = kwargs.get('attention_mask', None)
        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        # [batch_size, seq_len, num_samples]
        enc_out = self.forward(time_seqs, type_seqs, attention_mask)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)
        return lambdas


def get_non_pad_mask(seq,pad):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(pad).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q,pad):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self,opt, d_model, d_inner, n_head, d_k, d_v,num_types,phi_width,phi_depth, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()

        self.self_attn = HawkesAttention4(
            num_types= num_types,
            n_head= n_head,
            d_model= d_model,
            d_k= d_k,
            d_v= d_v,
            phi_width= phi_width,
            phi_depth= phi_depth,
            dropout= dropout,
            normalize_before= True
        )
            
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, event_time, event_type, non_pad_mask=None, self_attn_mask=None):

        enc_output = self.self_attn(
            q=enc_input,
            k=enc_input,
            v=enc_input,
            t_in=event_time,      #(B,L)
            c=event_type,          # (B,L)
            mask=self_attn_mask
        )
            
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output
    
class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
    
class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,opt,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v,phi_width, phi_depth, dropout):
        super().__init__()

        self.d_model = d_model
        self.pad = opt.pad_token_id

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cpu'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=self.pad)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(opt,d_model, d_inner, n_head, d_k, d_v,num_types,phi_width,phi_depth, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type, pad=self.pad)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)


        enc_output = self.event_emb(event_type)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                event_time=event_time,  # (B,L)
                event_type=event_type,  # (B,L)
                non_pad_mask=non_pad_mask,
                self_attn_mask=slf_attn_mask)
        return enc_output
    

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
        # print("HawkesRNN!!!!!!!")
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out