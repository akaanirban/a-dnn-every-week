"""
This code is heavily borrowing from the following repos: 
1. https://github.com/facebookresearch/llama-recipes
2. https://github.com/Lightning-AI/lit-llama
3. https://github.com/tairov/llama2.py
4. https://github.com/hkproj/pytorch-llama.git
"""

import einops
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from typing_extensions import Self

@dataclass
class LLaMAConfig:
    vocab_size : int = 32_000
    n_layers : int = 32
    n_heads : int = 32
    n_embed : int = 4_096
    max_seq_length : int = 4_096
    n_kv_heads : Optional[int] = None
    dropout : float = 0.1
    eps : float = 1e-7

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])
    
llama_configs = {
    "7B": dict(n_layers=32, n_heads=32, hidden_dim=4096),
    "13B": dict(n_layers=40, n_heads=40, hidden_dim=5120),
    "65B": dict(n_layers=80, n_heads=64, hidden_dim=8192),
    "sample": dict(n_layers=4, n_heads=4, n_embed=512, max_seq_length=512)
}

MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]

class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.config = config
        # define the embedding layer --> b x seq_len -> b x seq_len x n_embed
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.n_embed)
        # define the transformers block
        self.transformer = nn.ModuleList()
        for _ in range(self.config.n_layers):
            self.transformer.append(TransformerBlock(self.config))
        # define the LM Head --> b x seq_len x n_embed x vocab_size
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias=False)
        self.layer_norm = RMSNorm(self.config.n_embed)
        #extra code for the rope cache
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[MaskCache] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, input_ids: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
    # def forward(self,input_ids: torch.Tensor,positions: torch.Tensor,):
    # def forward(self,input_ids: torch.Tensor,cache: RotatingBufferCache,seqlens: List[int],) -> torch.Tensor:
    # def forward(self, tokens: torch.Tensor, start_pos: int):
    # def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size , seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_length , f"Cannot forward sequence of length larger than {seq_len}, max sequence length : {self.config.max_seq_length}"

        # if the self.rope_cache is not set, build it the first time
        if self.rope_cache is None: 
            self.rope_cache = self.build_rope_cache(input_ids)
        # if the self.mask_cache is not set, build it for the first time
        if self.mask_cache is None: 
            self.mask_cache = self.build_mask_cache(input_ids)

        rope = self.rope_cache[:seq_len]
        mask = self.mask_cache[:, :, :seq_len, :seq_len]
        
        # if input_pos is not None:
        #     rope = self.rope_cache.index_select(0, input_pos)
        #     mask = self.mask_cache.index_select(2, input_pos)
        #     mask = mask[:, :, :, :max_seq_length]
        # else:
        #     rope = self.rope_cache[:T]
        #     mask = self.mask_cache[:, :, :T, :T]

        # forward pass the model
        x = self.embedding(input_ids) # token embeddings --> b x seq_len x n_embed
        if input_pos is None: 
            # this is the case when we do not want to use the cache
            for block in self.transformer:
                x, _ = block(x, rope, mask)
        else:
            for i, block in enumerate(self.transformer):
                x, self.kv_caches[i] = block(x, rope, mask, input_pos, self.kv_caches[i])

        x = self.layer_norm(x)
        logits = self.lm_head(x) # logits --> b x seq_len x vocab_size
        return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAConfig.from_name(name))
    
    def build_rope_cache(self, input_ids: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.max_seq_length,
            n_elem=self.config.n_embed//self.config.n_heads,
            dtype=input_ids.dtype,
            device=input_ids.device)
    
    def build_mask_cache(self, input_ids: torch.Tensor) -> MaskCache:
        ones = torch.ones(
            (self.config.max_seq_length, self.config.max_seq_length),
            device=input_ids.device,
            dtype=torch.bool
            )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0) # unsqueeze for batch and seq_len dimesion --> 1 x 1 x max_seq_length x max_seq_length

    def reset_cache(self) -> None:
        self.kv_caches.clear() 


class SelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None: 
        super().__init__()
        assert config.n_embed % config.n_heads == 0, f"Hidden dim cannot be equally distributed among {config.n_heads} heads"
        self.head_dim = config.n_embed // config.n_heads
        # define key, query and attention projection for all the heads at once
        self.attn_wts = nn.Linear(config.n_embed, 3* self.head_dim * config.n_heads, bias=False) # contains wK, wQ and wV
        # define output projection weights
        self.proj = nn.Linear(self.head_dim * config.n_heads, config.n_embed, bias=False)
        self.config = config

    def forward(self, x: torch.Tensor, rope: RoPECache, mask: MaskCache, input_pos: Optional[torch.Tensor]=None, kv_cache: Optional[KVCache]=None) -> Tuple[torch.Tensor, Optional[KVCache]]:
        b, seq_len, n_embed = x.shape # batch size, sequence length, embedding dimensionality (n_embd) | seq_len = 1
        # calculate the query, key values of all the heads in the batch and move head forward in the batch dim
        q, k, v = self.attn_wts(x).split(self.config.n_embed, dim=2) # (B, seq_len, 3*n_embed) -> (B, seq_len, n_embed), (B, seq_len, n_embed), (B, seq_len, n_embed)
        
        k = k.view(b, seq_len, self.config.n_heads, self.head_dim) # (B, seq_len=1, n_embed) -> (B, seq_len=1, n_heads, head_Dim)
        q = q.view(b, seq_len, self.config.n_heads, self.head_dim) # (B, seq_len=1, n_embed) -> (B, seq_len=1, n_heads, head_Dim)
        v = v.view(b, seq_len, self.config.n_heads, self.head_dim) # (B, seq_len=1, n_embed) -> (B, seq_len=1, n_heads, head_Dim)

        q = apply_rope(q, rope_cache=rope) # (B, 1, n_heads, head_Dim) --> (B, 1, n_heads, head_Dim)
        k = apply_rope(k, rope_cache=rope) # (B, 1, kv_head, head_Dim) --> (B, 1, kv_head, head_Dim)

        k = k.transpose(1,2) #(B, seq_len=1, n_heads, head_Dim) --> (B, n_heads, seq_len=1, head_Dim)
        q = q.transpose(1,2) #(B, seq_len=1, n_heads, head_Dim) --> (B, n_heads, seq_len=1, head_Dim)
        v = v.transpose(1,2) #(B, seq_len=1, n_heads, head_Dim) --> (B, n_heads, seq_len=1, head_Dim)
        # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/appnotes/transformers-neuronx/generative-llm-inference-with-neuron.html
        if kv_cache is not None: 
            cache_k , cache_v = kv_cache
            # check if we have reached the token limit
            if input_pos[-1] >= self.config.max_seq_length:
                input_pos = torch.tensor(self.config.max_seq_length-1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            # replace the entries in the cache
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v
        
        # causal self attention
        # causal self-attention; Self-attend: (B, n_heads, seq_len,head_dim) x (B, n_heads,head_dim, T) -> (B, n_heads, seq_len, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(mask[:,:,:seq_len,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, n_heads, seq_len, T) x (B, n_heads, seq_len,head_dim) -> (B, n_heads, seq_len,head_dim)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1,2).contiguous().view(b, seq_len, n_embed)

        #output projection
        y = self.proj(y)
        return y, kv_cache


class FeedForward(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embed # llama configuration
        n_hidden = int(2 * hidden_dim /3)
        n_hidden = find_multiple(n_hidden, 256)

        self.fc1 = nn.Linear(config.n_embed, n_hidden, bias=False)
        self.fc2 = nn.Linear(config.n_embed, n_hidden, bias=False)
        self.proj = nn.Linear(n_hidden, config.n_embed, bias=False)

    def forward(self, x: torch.Tensor):
         # b x seq_len x dim --> b x seq_len x hidden_dim
         swish = F.silu(self.fc1(x))
         # b x seq_len x dim --> b x seq_len x hidden_dim
         x_V = self.fc2(x)
         # (b x seq_len x hidden_dim) * (b x seq_len x hidden_dim) --> (b x seq_len x hidden_dim)
         x = swish * x_V
         # (b x seq_len x hidden_dim) --> (b x seq_len x dim)
         x = self.proj(x)
         
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x*x, dim=self.dim, keepdim=True) # norm of x
        x_normed = x * torch.rsqrt(norm_x + self.eps) # reciprocal of the square root
        return self.scale * x_normed

class TransformerBlock(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None: 
        super().__init__()
        self.rms_norm_1 = RMSNorm(config.n_embed)
        self.rms_norm_2 = RMSNorm(config.n_embed)
        self.attn = SelfAttention(config)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, rope: RoPECache, mask: MaskCache, input_pos: Optional[torch.Tensor]=None, kv_cache: Optional[KVCache]=None) -> Tuple[torch.Tensor, Optional[KVCache]]:
        h, new_kvcache = self.attn(self.rms_norm_1(x), rope, mask, input_pos, kv_cache)
        x = x+h
        x = x + self.mlp(self.rms_norm_2(x))
        return x, new_kvcache

def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.

    Returns tensor of shape --> seq_len x n_elem /2 x 2
    """
    # paper suggests to use \theta_i = 10000^-{2*(i-1)/d}, i \in [1,2,3,...,d/2]
    theta = 1. / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device).float()/ n_elem ))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache

def apply_rope(x: torch.Tensor, rope_cache: RoPECache) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def find_multiple(n: int, k: int) -> int:
    # find the nearest multiple of k to n
    if n % k == 0:
        return n
    return n + k - (n % k)


if __name__=="__main__":
    llama = LLaMA(LLaMAConfig.from_name("sample"))
    print(llama)