"""
A recreation of BERT
original paper: https://arxiv.org/pdf/1810.04805.pdf
transformers implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
more info : https://github.com/labmlai/annotated_deep_learning_paper_implementations
and https://github.com/ajhalthor/Transformer-Neural-Network/blob/main/transformer.py
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import NamedTuple


@dataclass
class BertConfig(NamedTuple): 
    number_attention_layers = 12
    vocab_size = 32_000
    d_model = 384 # hidden dimension (for bert it was 768)
    maxlen = 512
    n_segments = 2
    dropout_prob = 0.4
    hidden_dim = 2048
    n_heads = 8


def scaled_dot_product(Q, K, V, mask=None):
    """Q, K and V are of shape batch x seqlen x d_model/embedding_seq_length"""
    # Q, K, V = 30 x 8 x 512 x 48
    d_k = Q.shape[-1] 
    matmul = torch.matmul(Q, K.transpose(-1, -2))/ np.sqrt(d_k) # 30 x 8 x 512 x 512 | ***Quadratic Memory+Compute*** 
    if mask is not None:
        matmul = matmul.permute(1 , 0, 2, 3) + mask
        matmul = matmul.permute(1, 0, 2, 3)
    attn = nn.functional.softmax(matmul, dim=1) # 30 x 8 x 512 x 512
    values = torch.matmul(attn, V) # 30 x 8 x 512 x 64
    return values, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, bert_config: BertConfig = BertConfig) -> None:
        super(MultiHeadAttention, self).__init__()
        self.bert_config = bert_config
        self.head_dimension = self.bert_config.d_model // self.bert_config.n_heads # 384 /8 = 48
        self.qkv_layer = nn.Linear(self.bert_config.d_model, 3* self.bert_config.d_model) # 384 x 3*384
        self.linear_layer = nn.Linear(self.bert_config.d_model, self.bert_config.d_model) # 384 x 384

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size() # 30 x 512 x 384
        qkv = self.qkv_layer(x) # 30 x 512 x 3*384
        qkv = qkv.reshape(batch_size, seq_len, self.bert_config.n_heads, 3*self.head_dimension) # 30 x 512 x 8 x 3*48
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 512 x 3*48
        Q, K, V = qkv.chunk(3, dim=-1) # each chunk will be -> 30 x 8 x 512 x 48 | batch_size x num_heads x maxseqlen x head_dimension
        values, attn = scaled_dot_product(Q, K, V, mask)
        values = values.reshape(batch_size, seq_len, self.bert_config.n_heads*self.head_dimension) # values will have the same dim as x --> 30 x 512 x 384 
        out = self.linear_layer(values)
        return out

class LayerNorm(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out


class FeedForward(nn.Module):
    def __init__(self, bert_config: BertConfig = BertConfig) -> None:
        super(FeedForward, self).__init__()
        self.bert_config = bert_config
        self.layer1 = nn.Linear(self.bert_config.d_model, self.bert_config.hidden_dim) # 384 x 2048
        self.layer2 = nn.Linear(self.bert_config.hidden_dim, self.bert_config.d_model) # 2048 x 384
        self.dropout = nn.Dropout(self.bert_config.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = 30 x 512 x 384
        x = self.layer1(x) # 30 x 512 x 2048 
        x = self.relu(x) # 30 x 512 x 2048
        x = self.dropout(x) # 30 x 512 x 2048
        x = self.layer2(x) # 30 x 512 x 384
        return x # 30 x 512 x 384


class EncoderBlock(nn.Module):
    def __init__(self, bert_config: BertConfig = BertConfig) -> None:
        super(EncoderBlock, self).__init__()
        self.bert_config = bert_config
        self.mha = MultiHeadAttention(self.bert_config)
        self.linear = FeedForward(self.bert_config)
        self.layernorm = nn.LayerNorm(self.bert_config.d_model)
        self.dropout = nn.Dropout(self.bert_config.dropout_prob)

    def forward(self, x, mask=None):
        h = self.mha(x, mask)
        h = self.dropout(h)
        x = self.layernorm(x + h)
 
        h = self.linear(x)
        h = self.dropout(h)
        x = self.layernorm(x + h)
        return x
        

class BertInput(nn.Module):
    def __init__(self, bert_config: BertConfig = BertConfig, pos_emb=True, seg_emb=True) -> None:
        super(BertInput, self).__init__()
        self.bert_config = bert_config
        self.positional_embedding = nn.Embedding(self.bert_config.maxlen, self.bert_config.d_model) if pos_emb else None # 512 x 384
        self.token_embedding = nn.Embedding(self.bert_config.vocab_size, self.bert_config.d_model) # 32_000 x 384
        self.segment_embedding = nn.Embedding(self.bert_config.n_segments, self.bert_config.d_model) if seg_emb else None # 2 x 384
        # self.norm = LayerNorm(self.bert_config.d_model)
        self.layernorm = nn.LayerNorm(self.bert_config.d_model)
        self.drop = nn.Dropout(self.bert_config.dropout_prob)

    def forward(self, x, segment):
        h = self.token_embedding(x)
        if self.positional_embedding is not None:
            seg_length = x.shape[1]
            pos = torch.arange(seg_length, dtype=torch.long, device=x.device) # (seq_length, )
            pos = pos.unsqueeze(0).expand_as(x) # (seq_length,) -> (batch_size , seq_length)
            h = h + self.positional_embedding(pos)
        if self.segment_embedding is not None:
            h = h + self.segment_embedding(segment)
        # return self.drop(self.norm(h))
        return self.drop(self.layernorm(h))

    

class BERT(nn.Module):
    """Just the transformer encoder"""
    def __init__(self, bert_config: BertConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bert_config = bert_config
        self.bert_input = BertInput(self.bert_config, True, False)
        self.blocks = nn.ModuleList(
            EncoderBlock(self.bert_config) for _ in range(self.bert_config.number_attention_layers)
        )

    def forward(self, input_ids, segment=None, attn_masks=None):
        h = self.bert_input(input_ids, segment)
        for block in self.blocks:
            h = block(h, attn_masks)
        return h



if __name__=="__main__":
    bert_model = BERT(BertConfig)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sentences = tokenizer(["Hello world what is up"]*30, return_tensors="pt")
    y = bert_model.forward(input_ids=sentences['input_ids'])
    __import__("IPython").embed()
    print(y.shape)