# #ETH-GBERT
# import math
# import inspect
# import torch
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F


# # for huggingface transformers 0.6.2;
# from pytorch_pretrained_bert.modeling import (
#     BertEmbeddings,
#     BertEncoder,
#     BertModel,
#     BertPooler,
# )


# class VocabGraphConvolution(nn.Module):
#     def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
#         super().__init__()
#         self.voc_dim = voc_dim
#         self.num_adj = num_adj
#         self.hid_dim = hid_dim
#         self.out_dim = out_dim

#         for i in range(self.num_adj):
#             setattr(
#                 self, "W%d_vh" % i, nn.Parameter(torch.randn(voc_dim, hid_dim))
#             )

#         self.fc_hc = nn.Linear(hid_dim, out_dim)
#         self.act_func = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)

#         self.reset_parameters()

#     def reset_parameters(self):
#         for n, p in self.named_parameters():
#             if (
#                     n.startswith("W")
#                     or n.startswith("a")
#                     or n in ("W", "a", "dense")
#             ):
#                 init.kaiming_uniform_(p, a=math.sqrt(5))

#     def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
#         for i in range(self.num_adj):
#             # H_vh = vocab_adj_list[i].mm(getattr(self, "W%d_vh" % i))
#             if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
#                 raise TypeError("Expected a PyTorch sparse tensor")
#             H_vh = torch.sparse.mm(vocab_adj_list[i].float(), getattr(self, "W%d_vh" % i))

#             # H_vh=self.dropout(F.elu(H_vh))
#             H_vh = self.dropout(H_vh)
#             H_dh = X_dv.matmul(H_vh)

#             if add_linear_mapping_term:
#                 H_linear = X_dv.matmul(getattr(self, "W%d_vh" % i))
#                 H_linear = self.dropout(H_linear)
#                 H_dh += H_linear

#             if i == 0:
#                 fused_H = H_dh
#             else:
#                 fused_H += H_dh

#         out = self.fc_hc(fused_H)
#         return out

# def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
#     """
#     Implement DiffSoftmax for using soft or hard labels during training.
#     - tau: Temperature parameter that controls the smoothness of the softmax output
#     - hard: Whether to use hard labels
#     """
#     y_soft = (logits / tau).softmax(dim)
#     if hard:
#         # Straight through.
#         index = y_soft.max(dim, keepdim=True)[1]
#         y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
#         ret = y_hard - y_soft.detach() + y_soft
#     else:
#         # Reparametrization trick.
#         ret = y_soft
#     return ret


# class DynamicFusionLayer(nn.Module):
#     def __init__(self, hidden_dim, tau=1.0, hard_gate=False):
#         super(DynamicFusionLayer, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.tau = tau
#         self.hard_gate = hard_gate

#         self.gate_network = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             # nn.Dropout(p=0.5),
#             nn.Linear(hidden_dim, 3),
#             # nn.Softmax(dim=-1),
#         )

#         self.fusion_weight = nn.Parameter(torch.tensor(0.5))

#     def forward(self, bert_embeddings, gcn_enhanced_embeddings):
#         concat_embeddings = torch.cat([bert_embeddings, gcn_enhanced_embeddings], dim=-1)

#         gate_logits = self.gate_network(concat_embeddings)
#         gate_values = DiffSoftmax(gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1)

#         gate_bert_only = gate_values[:, :, 0].unsqueeze(-1)
#         gate_gcn_enhanced = gate_values[:, :, 1].unsqueeze(-1)
#         gate_gcn_bert_weighted = gate_values[:, :, 2].unsqueeze(-1)

#         embeddings_bert_only = bert_embeddings
#         embeddings_gcn_enhanced = gcn_enhanced_embeddings
#         embeddings_gcn_bert_weighted = self.fusion_weight * bert_embeddings + (1 - self.fusion_weight) * gcn_enhanced_embeddings

#         fused_embeddings = (
#                 gate_bert_only * embeddings_bert_only +
#                 gate_gcn_enhanced * embeddings_gcn_enhanced +
#                 gate_gcn_bert_weighted * embeddings_gcn_bert_weighted
#         )

#         return fused_embeddings


# class ETH_GBertEmbeddings(BertEmbeddings):
#     def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim):
#         super(ETH_GBertEmbeddings, self).__init__(config)
#         assert gcn_embedding_dim >= 0
#         self.gcn_embedding_dim = gcn_embedding_dim
#         self.vocab_gcn = VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim)

#         self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

#     def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None):
#         words_embeddings = self.word_embeddings(input_ids)

#         vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
#         gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)

#         gcn_words_embeddings = words_embeddings.clone()
#         for i in range(self.gcn_embedding_dim):
#             tmp_pos = (attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
#                        ) + torch.arange(0, input_ids.shape[0]).to(input_ids.device) * input_ids.shape[1]
#             gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = gcn_vocab_out[:, :, i]

#         new_words_embeddings = self.dynamic_fusion_layer(words_embeddings, gcn_words_embeddings)

#         seq_length = input_ids.size(1)
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)

#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)

#         embeddings = new_words_embeddings + position_embeddings + token_type_embeddings

#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings


# class ETH_GBertModel(BertModel):
#     def __init__(
#             self,
#             config,
#             gcn_adj_dim,
#             gcn_adj_num,
#             gcn_embedding_dim,
#             num_labels,
#             output_attentions=False,
#             keep_multihead_output=False,
#     ):
#         super().__init__(config)
#         self.embeddings = ETH_GBertEmbeddings(
#             config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim
#         )
#         self.encoder = BertEncoder(config)
#         self.pooler = BertPooler(config)
#         self.num_labels = num_labels
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.num_labels)
#         self.output_attentions = config.output_attentions if hasattr(config, 'output_attentions') else False
#         self.keep_multihead_output = config.keep_multihead_output if hasattr(config, 'keep_multihead_output') else False
#         self.will_collect_cls_states = False
#         self.all_cls_states = []
#         self.apply(self.init_bert_weights)

#     def forward(
#             self,
#             vocab_adj_list,
#             gcn_swop_eye,
#             input_ids,
#             token_type_ids=None,
#             attention_mask=None,
#             output_all_encoded_layers=False,
#             head_mask=None,
#     ):
#         vocab_adj_list = [adj * 0 for adj in vocab_adj_list]

#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)

#         embedding_output = self.embeddings(
#             vocab_adj_list,
#             gcn_swop_eye,
#             input_ids,
#             token_type_ids,
#             attention_mask,
#         )

#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#         extended_attention_mask = extended_attention_mask.to(
#             dtype=next(self.parameters()).dtype
#         )  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         if head_mask is not None:
#             if head_mask.dim() == 1:
#                 head_mask = (
#                     head_mask.unsqueeze(0)
#                     .unsqueeze(0)
#                     .unsqueeze(-1)
#                     .unsqueeze(-1)
#                 )
#                 head_mask = head_mask.expand_as(
#                     self.config.num_hidden_layers, -1, -1, -1, -1
#                 )
#             elif head_mask.dim() == 2:
#                 head_mask = (
#                     head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
#                 )  # We can specify head_mask for each layer
#             head_mask = head_mask.to(
#                 dtype=next(self.parameters()).dtype
#             )  # switch to fload if need + fp16 compatibility
#         else:

#             head_mask = [None] * self.config.num_hidden_layers

#         encoder_args = {

#         }
#         if 'head_mask' in inspect.signature(self.encoder.forward).parameters:
#             encoder_args['head_mask'] = head_mask

#         if self.output_attentions:
#             output_all_encoded_layers = True

#         encoded_layers = self.encoder(
#             embedding_output,
#             extended_attention_mask,
#             output_all_encoded_layers=output_all_encoded_layers,
#             **encoder_args
#             # head_mask=head_mask,
#         )
#         if self.output_attentions:
#             all_attentions, encoded_layers = encoded_layers

#         pooled_output = self.pooler(encoded_layers[-1])
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         if self.output_attentions:
#             return all_attentions, logits

#         return logits




#ETH-GSetBert

import math
import inspect
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from env_config import env_config

# for huggingface transformers 0.6.2;
from pytorch_pretrained_bert.modeling import (
    BertEmbeddings,
    BertEncoder,
    BertModel,
    BertPooler,
)


class VocabGraphConvolution(nn.Module):
    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        for i in range(self.num_adj):
            setattr(
                self, "W%d_vh" % i, nn.Parameter(torch.randn(voc_dim, hid_dim))
            )

        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if (
                    n.startswith("W")
                    or n.startswith("a")
                    or n in ("W", "a", "dense")
            ):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            # H_vh = vocab_adj_list[i].mm(getattr(self, "W%d_vh" % i))
            if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
                raise TypeError("Expected a PyTorch sparse tensor")
            H_vh = torch.sparse.mm(vocab_adj_list[i].float(), getattr(self, "W%d_vh" % i))

            # H_vh=self.dropout(F.elu(H_vh))
            H_vh = self.dropout(H_vh)
            H_dh = X_dv.matmul(H_vh)

            if add_linear_mapping_term:
                H_linear = X_dv.matmul(getattr(self, "W%d_vh" % i))
                H_linear = self.dropout(H_linear)
                H_dh += H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out = self.fc_hc(fused_H)
        return out

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    """
    Implement DiffSoftmax for using soft or hard labels during training.
    - tau: Temperature parameter that controls the smoothness of the softmax output
    - hard: Whether to use hard labels
    """
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class DynamicFusionLayer(nn.Module):
    def __init__(self, hidden_dim, tau=1.0, hard_gate=False):
        super(DynamicFusionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.hard_gate = hard_gate

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 3),
            # nn.Softmax(dim=-1),
        )

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, bert_embeddings, gcn_enhanced_embeddings):
        concat_embeddings = torch.cat([bert_embeddings, gcn_enhanced_embeddings], dim=-1)

        gate_logits = self.gate_network(concat_embeddings)
        gate_values = DiffSoftmax(gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1)

        gate_bert_only = gate_values[:, :, 0].unsqueeze(-1)
        gate_gcn_enhanced = gate_values[:, :, 1].unsqueeze(-1)
        gate_gcn_bert_weighted = gate_values[:, :, 2].unsqueeze(-1)

        embeddings_bert_only = bert_embeddings
        embeddings_gcn_enhanced = gcn_enhanced_embeddings
        embeddings_gcn_bert_weighted = self.fusion_weight * bert_embeddings + (1 - self.fusion_weight) * gcn_enhanced_embeddings

        fused_embeddings = (
                gate_bert_only * embeddings_bert_only +
                gate_gcn_enhanced * embeddings_gcn_enhanced +
                gate_gcn_bert_weighted * embeddings_gcn_bert_weighted
        )

        return fused_embeddings


class ETH_GBertEmbeddings(BertEmbeddings):
    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim):
        super(ETH_GBertEmbeddings, self).__init__(config)
        assert gcn_embedding_dim >= 0
        self.gcn_embedding_dim = gcn_embedding_dim
        self.vocab_gcn = VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim)

        self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None):
        words_embeddings = self.word_embeddings(input_ids)

        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)

        gcn_words_embeddings = words_embeddings.clone()
        for i in range(self.gcn_embedding_dim):
            tmp_pos = (attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
                       ) + torch.arange(0, input_ids.shape[0]).to(input_ids.device) * input_ids.shape[1]
            gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = gcn_vocab_out[:, :, i]

        new_words_embeddings = self.dynamic_fusion_layer(words_embeddings, gcn_words_embeddings)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = new_words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class _MultiHeadAttention(nn.Module):
    """Minimal MHA for SetTransformer blocks (no bias masking beyond pad mask)."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, key_mask: torch.Tensor = None):
        """
        Q: [B, Lq, D], K,V: [B, Lk, D]
        key_mask: [B, Lk] (True for valid positions)
        """
        B, Lq, D = Q.shape
        Lk = K.size(1)
        q = self.W_q(Q).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)  # [B,h,Lq,d_k]
        k = self.W_k(K).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)  # [B,h,Lk,d_k]
        v = self.W_v(V).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)  # [B,h,Lk,d_k]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)               # [B,h,Lq,Lk]
        if key_mask is not None:
            # key_mask True = keep; False = mask-out
            neg_inf = torch.finfo(scores.dtype).min
            mask = (~key_mask).unsqueeze(1).unsqueeze(1)                        # [B,1,1,Lk]
            scores = scores.masked_fill(mask, neg_inf)
        attn = F.softmax(scores, dim=-1)
        ctx = attn @ v                                                          # [B,h,Lq,d_k]
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lq, self.d_model)        # [B,Lq,D]
        return self.W_o(ctx)


class _MAB(nn.Module):
    """Multihead Attention Block (Lee et al., SetTransformer)."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = _MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, X, Y, key_mask: torch.Tensor = None):
        h = self.mha(X, Y, Y, key_mask)                 # attention
        X = self.ln1(X + self.drop(h))                  # res + norm
        z = self.ffn(X)
        X = self.ln2(X + self.drop(z))                  # res + norm
        return X


class _SAB(nn.Module):
    """Self-Attention Block = MAB(X, X)."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mab = _MAB(d_model, num_heads, dropout)

    def forward(self, X, key_mask: torch.Tensor = None):
        return self.mab(X, X, key_mask)


class _PMA(nn.Module):
    """Pooling by Multihead Attention with 1 seed (returns [B, 1, D])."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, n_seeds: int = 1):
        super().__init__()
        assert n_seeds == 1, "This implementation assumes n_seeds=1"
        self.S = nn.Parameter(torch.randn(n_seeds, d_model))
        self.mab = _MAB(d_model, num_heads, dropout)

    def forward(self, X, key_mask: torch.Tensor = None):
        B = X.size(0)
        S = self.S.unsqueeze(0).expand(B, -1, -1)       # [B,1,D]
        return self.mab(S, X, key_mask)                  # [B,1,D]


class SetEncoder(nn.Module):
    """
    SetTransformer encoder for a set of transactions per account.
    Input:
        set_feats: FloatTensor [B, Lmax, d_in]
        set_mask : BoolTensor  [B, Lmax]  (True = valid)
    Output:
        eS: FloatTensor [B, d_model]
    """
    def __init__(self, d_in: int, d_model: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.sabs = nn.ModuleList([_SAB(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.pma = _PMA(d_model, num_heads, dropout, n_seeds=1)

    def forward(self, set_feats: torch.Tensor, set_mask: torch.Tensor):
        # set_feats: [B,L,d_in], set_mask: [B,L] (bool)
        h = self.proj(set_feats)                         # [B,L,D]
        for sab in self.sabs:
            h = sab(h, key_mask=set_mask)               # mask applies on keys
        Z = self.pma(h, key_mask=set_mask)              # [B,1,D]
        return Z.squeeze(1)                             # [B,D]


class TriModalFusionGate(nn.Module):
    """
    Three-way dynamic fusion at pooled level:
      O1: BERT-only      -> eB
      O2: SET-only       -> eS (projected to d_bert)
      O3: Weighted mix   -> w*eB + (1-w)*eS
    Gate logits -> DiffSoftmax(tau, hard) -> g in simplex (3)
    Output fused = g1*O1 + g2*O2 + g3*O3
    """
    def __init__(self, d_bert: int, d_set: int, hidden: int = 256, tau: float = 1.0, hard_gate: bool = False,
                 alpha_init: float = 0.5):
        super().__init__()
        self.tau = tau
        self.hard_gate = hard_gate
        self.align_set = nn.Linear(d_set, d_bert) if d_set != d_bert else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(d_bert + d_set, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)  # logits for {BERT-only, SET-only, WeightedMix}
        )
        # α parameterizes w via sigmoid(w_alpha) in O3
        self.w_alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, eB: torch.Tensor, eS: torch.Tensor):
        """
        eB: [B, d_bert], eS: [B, d_set]
        returns: fused [B, d_bert], gates [B,3]
        """
        eS_proj = self.align_set(eS)                    # [B, d_bert]
        x = torch.cat([eB, eS], dim=-1)                 # gating sees raw concat
        logits = self.mlp(x)                            # [B,3]
        gates = DiffSoftmax(logits, tau=self.tau, hard=self.hard_gate, dim=-1)  # [B,3]

        O1 = eB
        O2 = eS_proj
        w = torch.sigmoid(self.w_alpha)
        O3 = w * eB + (1.0 - w) * eS_proj

        fused = gates[:, 0:1] * O1 + gates[:, 1:2] * O2 + gates[:, 2:3] * O3
        return fused, gates
    
class ETH_GBertModel(BertModel):
    def __init__(
            self,
            config,
            gcn_adj_dim,
            gcn_adj_num,
            gcn_embedding_dim,
            num_labels,
            output_attentions=False,
            keep_multihead_output=False,
    ):
        super().__init__(config)
        self.embeddings = ETH_GBertEmbeddings(
            config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim
        )
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.output_attentions = config.output_attentions if hasattr(config, 'output_attentions') else False
        self.keep_multihead_output = config.keep_multihead_output if hasattr(config, 'keep_multihead_output') else False
        self.will_collect_cls_states = False
        self.all_cls_states = []
        # SetEncoder (SAB + PMA)
        self.set_encoder = SetEncoder(
            d_in=env_config.SET_D_IN,
            d_model=env_config.SET_D_MODEL,
            num_heads=env_config.SET_HEADS,
            num_layers=env_config.SET_LAYERS,
            dropout=env_config.SET_DROPOUT,
        )
        # Không cần align ở đây; TriModalFusionGate sẽ tự align eS -> d_bert
        self.set_align = nn.Identity()


        # Gating 3 nhánh ở mức pooled
        self.tri_fusion = TriModalFusionGate(
            d_bert=config.hidden_size,
            d_set=env_config.SET_D_MODEL,
            hidden=env_config.FUSION_HIDDEN,
            tau=env_config.GUMBEL_TAU,
            hard_gate=env_config.GUMBEL_HARD,
            alpha_init=env_config.FUSION_ALPHA_INIT,
        )
        # Tuỳ chọn: nơi lưu gates để debug/log (không đổi API trả về)
        self.latest_fusion_gates = None
        self.apply(self.init_bert_weights)

    def forward(
            self,
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            output_all_encoded_layers=False,
            head_mask=None,
            return_branches: bool = False,
            *,
            set_feats=None,    # FloatTensor [B, Lmax, d_in] hoặc None
            set_mask=None,     # BoolTensor  [B, Lmax]     hoặc None
    ):

        vocab_adj_list = [adj * 0 for adj in vocab_adj_list]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedding_output = self.embeddings(
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            token_type_ids,
            attention_mask,
        )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                head_mask = head_mask.expand_as(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:

            head_mask = [None] * self.config.num_hidden_layers

        encoder_args = {

        }
        if 'head_mask' in inspect.signature(self.encoder.forward).parameters:
            encoder_args['head_mask'] = head_mask

        if self.output_attentions:
            output_all_encoded_layers = True

        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            **encoder_args
            # head_mask=head_mask,
        )
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        pooled_output = self.pooler(encoded_layers[-1])

        # (1) LƯU EMBEDDING NHÁNH BERT (trước fusion)
        eB = pooled_output  # [B, d_bert]

        logits_B = None
        logits_S = None

        # (2) NẾU CÓ SET FEATS -> TÍNH eS, CHIẾU VỀ d_bert, VÀ FUSION
        if (set_feats is not None) and (set_mask is not None):
            # eS: embedding của SetTransformer
            eS = self.set_encoder(set_feats, set_mask)  # [B, d_set]

            # eS_proj: chiếu Set embedding về đúng dim BERT để dùng chung classifier
            # align_set nằm bên trong tri_fusion (Linear hoặc Identity)
            eS_proj = self.tri_fusion.align_set(eS)     # [B, d_bert]

            # fused: embedding cuối sau tri-modal gate
            fused, gates = self.tri_fusion(eB, eS)      # fused [B, d_bert]
            self.latest_fusion_gates = gates.detach()

            # (3) LOGITS RIÊNG CHO TỪNG NHÁNH (dùng cho distillation)
            logits_B = self.classifier(self.dropout(eB))        # [B, C]
            logits_S = self.classifier(self.dropout(eS_proj))   # [B, C]

            # dùng fused cho logits cuối (inference)
            pooled_output = fused

        # (4) LOGITS CHÍNH (FUSED) - dùng cho CE + inference
        logits = self.classifier(self.dropout(pooled_output))   # [B, C]

        if self.output_attentions:
            return all_attentions, logits

        # (5) TRẢ OUTPUT:
        # - nếu return_branches=False: giữ y hệt behavior cũ (return tensor logits)
        # - nếu return_branches=True : return dict có đủ 3 logits
        if return_branches:
            return {"logits": logits, "logits_B": logits_B, "logits_S": logits_S}

        return logits
