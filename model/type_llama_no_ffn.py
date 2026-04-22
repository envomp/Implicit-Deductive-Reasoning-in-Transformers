# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional

try:
    from flash_attn import flash_attn_func

    print("Flash Attention 2 found and imported.")
    _flash_attn_available = True
except ImportError:
    print("Flash Attention 2 not found. Install with 'pip install flash-attn --no-build-isolation'")
    print("Falling back to standard attention.")
    _flash_attn_available = False


@dataclass_json
@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    output_size: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    ffn_enabled: bool = False
    universal_transformer: bool = False
    lnorm_implementation: str = "RMSNorm"
    dropout_rate: float = 0
    type_embeddings: int = 10
    max_seq_len: int = 1024
    rope_theta: int = 10000
    use_scaled_rope: bool = True
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    generated_type_indices: list = None
    pad_token_id: int = -1
    bidirectional_stop_tokens: list = None
    isolated_subsequence_tokens: list = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight + self.bias


def apply_scaling(freqs: torch.Tensor):
    # RoPE scaling (values obtained from grid search)
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x.unsqueeze(3).expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class KVCache(nn.Module):
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos: int, xk: torch.Tensor, xv: torch.Tensor, indices: Optional[torch.Tensor] = None):
        seqlen = xk.size(1)
        if indices is None:
            self.cache_k[:, start_pos: start_pos + seqlen] = xk
            self.cache_v[:, start_pos: start_pos + seqlen] = xv
            return self.cache_k[:, : start_pos + seqlen], self.cache_v[:, : start_pos + seqlen]
        else:
            self.cache_k[indices, start_pos: start_pos + seqlen] = xk
            self.cache_v[indices, start_pos: start_pos + seqlen] = xv
            return self.cache_k[indices, : start_pos + seqlen], self.cache_v[indices, : start_pos + seqlen]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dim = args.dim
        self.use_flash_attn = args.use_flash_attention
        self.dropout_rate = args.dropout_rate
        self.dropout = nn.Dropout(args.dropout_rate)
        self.cache = None

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor | bool] = None,
                start_pos: int = None, active_indices: Optional[torch.Tensor] = None, return_attn_map=False):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.cache is not None:
            xk, xv = self.cache.update(start_pos, xk, xv, active_indices)

        if _flash_attn_available and self.use_flash_attn:
            dropout_p = self.dropout_rate if self.training else 0.0
            output = flash_attn_func(
                xq, xk, xv,
                dropout_p=dropout_p,
                causal=mask is not None,
            )
            output = output.contiguous().view(bsz, seqlen, -1)
            return self.wo(output)
        else:
            # repeat k/v heads if n_kv_heads < n_heads (GQA)
            xk = repeat_kv(xk, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            xv = repeat_kv(xv, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))

            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            proj = self.wo(output)
            if return_attn_map:
                return self.dropout(proj), scores
            else:
                return self.dropout(proj)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def get_lnorm(args):
    if args.lnorm_implementation == "RMSNorm":
        return RMSNorm(args.dim, eps=1e-05)
    elif args.lnorm_implementation == "LayerNorm":
        pass
        return LayerNorm(args.dim, eps=1e-05)
    else:
        raise RuntimeError("unknown: " + args.lnorm_implementation)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.layer_id = layer_id
        self.attention_norm = get_lnorm(args)
        self.ffn_enabled = args.ffn_enabled
        if self.ffn_enabled:
            self.feed_forward = FeedForward(args)
            self.ffn_norm = get_lnorm(args)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor | bool] = None,
                start_pos: int = None, active_indices: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), freqs_cis=freqs_cis, mask=mask, start_pos=start_pos, active_indices=active_indices)
        if self.ffn_enabled:
            h = h + self.feed_forward(self.ffn_norm(h))
        return h


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.flash_attn = params.use_flash_attention
        self.gradient_checkpointing = params.gradient_checkpointing
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0 / params.dim ** 0.5)

        self.type_embeddings = torch.nn.Parameter(torch.randn(params.type_embeddings, params.dim))
        self.generated_type_indices = torch.tensor([params.generated_type_indices])
        self.bidirectional_stop_tokens = params.bidirectional_stop_tokens
        self.isolated_subsequence_tokens = params.isolated_subsequence_tokens
        self.pad_token_id = params.pad_token_id

        self.layers = torch.nn.ModuleList()
        if params.universal_transformer:
            universal = TransformerBlock(0, params)
            for layer_id in range(params.n_layers):
                self.layers.append(universal)
        else:
            for layer_id in range(params.n_layers):
                self.layers.append(TransformerBlock(layer_id, params))

        self.norm = get_lnorm(params)
        self.output = nn.Linear(params.dim, params.output_size if params.output_size else params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len * 2, params.rope_theta, params.use_scaled_rope)

    def uninstall_cache(self):
        for block in self.layers:
            block.attention.cache = None

    def install_cache(self, bsz, total_len):
        for block in self.layers:
            layer_dtype = block.attention.wq.weight.dtype
            layer_device = block.attention.wq.weight.device
            block.attention.cache = KVCache(
                batch_size=bsz,
                seq_length=total_len,
                n_kv_heads=self.params.n_kv_heads,
                head_dim=self.params.dim // self.params.n_heads,
                dtype=layer_dtype,
                device=layer_device,
            )

    def forward(self, input_ids: torch.Tensor, start_pos: int, type_embeddings=None, active_indices: Optional[torch.Tensor] = None):
        _bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        if type_embeddings is not None:
            h = h + get_type_embeddings(self.type_embeddings, h, type_embeddings)
        freqs_cis = self.freqs_cis.to(h.device)[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            if start_pos == 0:  # prefill (start_pos == 0)
                mask = create_isolated_attention_mask(input_ids=input_ids,
                                                      isolated_subsequence_tokens=self.isolated_subsequence_tokens,
                                                      bidirectional_stop_tokens=self.bidirectional_stop_tokens,
                                                      pad_token_id=self.pad_token_id, dtype=h.dtype, device=input_ids.device)
            else:  # Standard causal mask for generation
                mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
                mask = torch.triu(mask, diagonal=1)

            # Handle KV cache
            # If start_pos == 0, zero_pad is empty and this does nothing.
            # If start_pos > 0, this pads the mask to account for cached tokens.
            zero_pad = torch.zeros((_bsz, 1, seqlen, start_pos), device=input_ids.device)
            mask = torch.cat([zero_pad, mask], dim=3).type_as(h)

        for layer in self.layers:
            h = layer(h, freqs_cis=freqs_cis, mask=mask, start_pos=start_pos, active_indices=active_indices)
        return self.output_proj(h)

    def forward_train(self, input_ids: torch.Tensor, type_embeddings=None, output_hidden_states=False):
        _bsz, seqlen = input_ids.shape
        self.uninstall_cache()
        h = self.tok_embeddings(input_ids)
        if type_embeddings is not None:
            h = h + get_type_embeddings(self.type_embeddings, h, type_embeddings)
        freqs_cis = self.freqs_cis.to(h.device)[:seqlen]
        mask = create_isolated_attention_mask(input_ids=input_ids,
                                              isolated_subsequence_tokens=self.isolated_subsequence_tokens,
                                              bidirectional_stop_tokens=self.bidirectional_stop_tokens,
                                              pad_token_id=self.pad_token_id, dtype=h.dtype, device=input_ids.device)
        all_hidden_states = []
        for layer in self.layers:
            if self.gradient_checkpointing:
                h = checkpoint(layer, h, freqs_cis, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, mask)
            all_hidden_states.append(h)

        if output_hidden_states:
            return torch.stack(all_hidden_states, dim=1).to(h.device)
        else:
            return self.output_proj(h)

    def output_proj(self, h):
        return self.output(self.norm(h)).float()

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, stop_tokens: List[int],
                 type_embeddings: torch.Tensor = None, max_new_tokens: int = 32, kv_cache=True,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0, return_log_probs: bool = False):
        """
        Generates sequences of tokens autoregressively.
        Manages the KV cache lifecycle. Can optionally return log probabilities for use in RL.
        """
        bsz, prompt_len = input_ids.shape
        if kv_cache:
            self.install_cache(bsz, self.params.max_seq_len)

        prefill_logits = self.forward(input_ids, start_pos=0, type_embeddings=type_embeddings)
        logits = prefill_logits[:, -1, :]

        generated_tokens = input_ids
        all_log_probs = []
        is_finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)

        for i in range(max_new_tokens):
            if i > 0:
                unfinished_sents = ~is_finished
                if not unfinished_sents.any():
                    break

                next_logits = torch.zeros_like(logits)
                unfinished_indices = unfinished_sents.nonzero(as_tuple=True)[0]
                tokens_to_process = next_token[unfinished_indices]
                current_pos = prompt_len + i - 1
                new_logits = self.forward(tokens_to_process, start_pos=current_pos,
                                          type_embeddings=self.generated_type_indices.to(input_ids.device),
                                          active_indices=unfinished_indices)[:, -1, :]
                next_logits[unfinished_indices] = new_logits
                logits = next_logits

            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            if top_k > 0 or top_p > 0.0:
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            tokens_to_append = next_token.clone()
            tokens_to_append[is_finished.unsqueeze(-1)] = self.pad_token_id
            generated_tokens = torch.cat([generated_tokens, tokens_to_append], dim=1)

            if return_log_probs:
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_prob = torch.gather(log_probs, 1, next_token).squeeze(-1)
                token_log_prob[is_finished] = 0.0
                all_log_probs.append(token_log_prob)

            for stop_id in stop_tokens:
                is_finished = is_finished | (next_token.squeeze(-1) == stop_id)

        self.uninstall_cache()
        if return_log_probs:
            return generated_tokens, torch.stack(all_log_probs, dim=1)
        else:
            return generated_tokens


def get_type_embeddings(type_embeddings: torch.Tensor, tok_embeddings: torch.Tensor, type_indices: torch.Tensor):
    batch_size, seq_len, dim = tok_embeddings.shape
    if type_indices.ndim == 2:
        type_indices = type_indices.unsqueeze(0).expand(batch_size, -1, -1)
    _, _, types_len = type_indices.shape
    mask = type_indices != 0

    gathered_type_embeddings = type_embeddings.index_select(0, type_indices.reshape(-1))
    gathered_type_embeddings = gathered_type_embeddings.view(batch_size, seq_len, types_len, dim)
    masked_type_embeddings = gathered_type_embeddings * mask.unsqueeze(-1)
    summed_type_embeddings = torch.sum(masked_type_embeddings, dim=2)
    return summed_type_embeddings


def create_isolated_attention_mask(input_ids: torch.Tensor,
                                   isolated_subsequence_tokens: list[int] | None,
                                   bidirectional_stop_tokens: list[int] | None,
                                   pad_token_id: int = -1,
                                   dtype: torch.dtype = torch.float32,
                                   device: str | torch.device = 'cuda'):
    """
    Creates a batched attention mask with support for isolated subsequences and
    a bidirectional prefix.

    Args:
        input_ids: The batch of input token IDs. Shape: (bs, seqlen)
        isolated_subsequence_tokens: A list of token IDs that start an isolated
            subsequence. Tokens in one isolated subsequence cannot attend to
            tokens in another isolated subsequence.
        bidirectional_stop_tokens: A list of token IDs. The *first* occurrence
            of any of these tokens marks the end of a bidirectional prefix.
            Tokens *before* this token can attend to each other fully.
            Tokens *at and after* this token revert to the standard
            causal/isolated logic.
        pad_token_id: pad token id to ignore.
        dtype: The desired dtype for the mask.
        device: The desired device for the mask.

    Returns:
        A 4D attention mask. Shape: (bs, 1, seq_len, seq_len)
    """
    bs, seq_len = input_ids.shape
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)

    if bidirectional_stop_tokens:
        stop_tokens_tensor = torch.tensor(bidirectional_stop_tokens, device=device)
        is_stop_token = torch.isin(input_ids, stop_tokens_tensor)
        stop_indices = is_stop_token.float().argmax(dim=1)
        no_stop_found = ~is_stop_token.any(dim=1)
        stop_indices = stop_indices.masked_fill(no_stop_found, seq_len)
        positions = torch.arange(seq_len, device=device)
        is_in_prefix = positions.unsqueeze(0) < stop_indices.unsqueeze(1)
        prefix_mask = is_in_prefix.unsqueeze(1).unsqueeze(2) & is_in_prefix.unsqueeze(1).unsqueeze(3)
        mask = mask.masked_fill(prefix_mask, 0.0)  # (bs, 1, seq_len, seq_len)

    if isolated_subsequence_tokens:
        iso_tokens_tensor = torch.tensor(isolated_subsequence_tokens, device=device)
        is_iso_start = torch.isin(input_ids, iso_tokens_tensor)
        segment_ids = is_iso_start.long().cumsum(dim=1)
        q_seg = segment_ids.unsqueeze(1).unsqueeze(2)  # (bs, 1, seq_len, seq_len)
        k_seg = segment_ids.unsqueeze(1).unsqueeze(3)  # (bs, 1, seq_len, seq_len)
        isolation_block = (q_seg > 0) & (k_seg > 0) & (q_seg != k_seg)
        mask = mask.masked_fill(isolation_block, float("-inf"))

    is_pad = (input_ids == pad_token_id)
    mask = mask.masked_fill(is_pad.unsqueeze(1).unsqueeze(2), float("-inf"))
    mask = mask.masked_fill(is_pad.unsqueeze(1).unsqueeze(3), float("-inf"))
    diag = torch.eye(seq_len, device=device, dtype=torch.bool)
    mask.masked_fill_(diag, 0.0)
    return mask


# ---- TESTS ----

def test_isolated_subsequence_mask():
    print("Running test_isolated_subsequence_mask...")
    device = 'cpu'
    dtype = torch.float32
    inf = float("-inf")

    # input_ids: [1, 2, 3, 4, 200, 5, 6, 7, 201, 8, 9] (batch size 1)
    # Segments:
    # - 0 (prefix): indices 0-3
    # - 1 (iso):    indices 4-7
    # - 2 (iso):    indices 8-10
    input_ids = torch.tensor([[1, 2, 3, 4, 200, 5, 6, 7, 201, 8, 9]], device=device)
    generated_mask = create_isolated_attention_mask(
        input_ids=input_ids,
        isolated_subsequence_tokens=[200, 201],
        bidirectional_stop_tokens=None,
        dtype=dtype,
        device=device
    )

    # Manually create the expected mask
    expected_mask_2d = torch.tensor([
        [0.0, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
        [0.0, 0.0, inf, inf, inf, inf, inf, inf, inf, inf, inf],
        [0.0, 0.0, 0.0, inf, inf, inf, inf, inf, inf, inf, inf],
        [0.0, 0.0, 0.0, 0.0, inf, inf, inf, inf, inf, inf, inf],
        [0.0, 0.0, 0.0, 0.0, 0.0, inf, inf, inf, inf, inf, inf],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inf, inf, inf, inf, inf],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inf, inf, inf, inf],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inf, inf, inf],
        [0.0, 0.0, 0.0, 0.0, inf, inf, inf, inf, 0.0, inf, inf],
        [0.0, 0.0, 0.0, 0.0, inf, inf, inf, inf, 0.0, 0.0, inf],
        [0.0, 0.0, 0.0, 0.0, inf, inf, inf, inf, 0.0, 0.0, 0.0]
    ], device=device, dtype=dtype)
    expected_mask = expected_mask_2d.unsqueeze(0).unsqueeze(0)
    assert torch.equal(generated_mask, expected_mask), "Isolated mask test failed."
    print("...passed.")


def test_standard_causal_mask():
    print("Running test_standard_causal_mask...")
    device = 'cpu'
    dtype = torch.float32
    inf = float("-inf")

    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    generated_mask = create_isolated_attention_mask(
        input_ids=input_ids,
        isolated_subsequence_tokens=None,
        bidirectional_stop_tokens=None,
        dtype=dtype,
        device=device
    )

    expected_mask_2d = torch.tensor([
        [0.0, inf, inf, inf],
        [0.0, 0.0, inf, inf],
        [0.0, 0.0, 0.0, inf],
        [0.0, 0.0, 0.0, 0.0],
    ], device=device, dtype=dtype)
    expected_mask = expected_mask_2d.unsqueeze(0).unsqueeze(0)
    assert torch.equal(generated_mask, expected_mask), "Standard causal mask test failed."
    print("...passed.")


def test_ignore_padding():
    print("Running test_ignore_padding...")
    device = 'cpu'
    dtype = torch.float32
    inf = float("-inf")

    input_ids = torch.tensor([[-1, 1, 2, 200, 4, 201, -1]], device=device)
    generated_mask = create_isolated_attention_mask(
        input_ids=input_ids,
        isolated_subsequence_tokens=[200, 201],
        bidirectional_stop_tokens=[200, 201],
        dtype=dtype,
        device=device
    )

    expected_mask_2d = torch.tensor([
        [0.0, inf, inf, inf, inf, inf, inf],
        [inf, 0.0, 0.0, inf, inf, inf, inf],
        [inf, 0.0, 0.0, inf, inf, inf, inf],
        [inf, 0.0, 0.0, 0.0, inf, inf, inf],
        [inf, 0.0, 0.0, 0.0, 0.0, inf, inf],
        [inf, 0.0, 0.0, inf, inf, 0.0, inf],
        [inf, inf, inf, inf, inf, inf, 0.0],
    ], device=device, dtype=dtype)
    expected_mask = expected_mask_2d.unsqueeze(0).unsqueeze(0)
    assert torch.equal(generated_mask, expected_mask), "padding test failed."
    print("...passed.")


def test_bidirectional_simple():
    print("Running test_bidirectional_simple...")
    device = 'cpu'
    dtype = torch.float32
    inf = float("-inf")

    # Token 300 is the stop token.
    # Region 1 (bidirectional): tokens 1, 2
    # Region 2 (causal): tokens 200, 4, 5
    input_ids = torch.tensor([[1, 2, 200, 4, 5]], device=device)
    generated_mask = create_isolated_attention_mask(
        input_ids=input_ids,
        isolated_subsequence_tokens=None,
        bidirectional_stop_tokens=[200],
        dtype=dtype,
        device=device
    )

    expected_mask_2d = torch.tensor([
        [0.0, 0.0, inf, inf, inf],  # k=1 (region 1, bidirectional)
        [0.0, 0.0, inf, inf, inf],  # k=2 (region 1, bidirectional)
        [0.0, 0.0, 0.0, inf, inf],  # k=300 (region 2, causal)
        [0.0, 0.0, 0.0, 0.0, inf],  # k=4 (region 2, causal)
        [0.0, 0.0, 0.0, 0.0, 0.0],  # k=5 (region 2, causal)
    ], device=device, dtype=dtype)
    expected_mask = expected_mask_2d.unsqueeze(0).unsqueeze(0)
    assert torch.equal(generated_mask, expected_mask), "Bidirectional simple test failed."
    print("...passed.")


def test_bidirectional_only():
    print("Running test_bidirectional_only...")
    device = 'cpu'
    dtype = torch.float32
    inf = float("-inf")

    # Token 300 is the stop token.
    # Region 1 (bidirectional): tokens 1, 2
    input_ids = torch.tensor([[-1, 1, 2, -1]], device=device)
    generated_mask = create_isolated_attention_mask(
        input_ids=input_ids,
        isolated_subsequence_tokens=None,
        bidirectional_stop_tokens=[200],
        dtype=dtype,
        device=device
    )

    expected_mask_2d = torch.tensor([
        [0.0, inf, inf, inf],
        [inf, 0.0, 0.0, inf],
        [inf, 0.0, 0.0, inf],
        [inf, inf, inf, 0.0],
    ], device=device, dtype=dtype)
    expected_mask = expected_mask_2d.unsqueeze(0).unsqueeze(0)
    assert torch.equal(generated_mask, expected_mask), "Bidirectional simple test failed."
    print("...passed.")


def test_bidirectional_with_isolation():
    print("Running test_bidirectional_with_isolation...")
    device = 'cpu'
    dtype = torch.float32
    inf = float("-inf")

    # Region 1 (bidirectional): (tokens 1, 2, 3)
    # Region 2 (causal/isolated): (tokens 200, 5, 6, 201, 8)
    # - isolated segment 1: (tokens 200, 5, 6)
    # - isolated segment 2: (tokens 201, 8)
    input_ids = torch.tensor([[1, 2, 3, 200, 5, 6, 201, 8]], device=device)
    generated_mask = create_isolated_attention_mask(
        input_ids=input_ids,
        isolated_subsequence_tokens=[200, 201],
        bidirectional_stop_tokens=[200, 201],
        dtype=dtype,
        device=device
    )

    expected_mask_2d = torch.tensor([
        [0.0, 0.0, 0.0, inf, inf, inf, inf, inf],  # 1 (bidirectional)
        [0.0, 0.0, 0.0, inf, inf, inf, inf, inf],  # 2 (bidirectional)
        [0.0, 0.0, 0.0, inf, inf, inf, inf, inf],  # 3 (bidirectional)
        [0.0, 0.0, 0.0, 0.0, inf, inf, inf, inf],  # 200 (isolated segment 1)
        [0.0, 0.0, 0.0, 0.0, 0.0, inf, inf, inf],  # 5
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inf, inf],  # 6
        [0.0, 0.0, 0.0, inf, inf, inf, 0.0, inf],  # 201 (isolated segment 2)
        [0.0, 0.0, 0.0, inf, inf, inf, 0.0, 0.0]  # 8
    ], device=device, dtype=dtype)
    expected_mask = expected_mask_2d.unsqueeze(0).unsqueeze(0)
    assert torch.equal(generated_mask, expected_mask), "Bidirectional with isolation test failed."
    print("...passed.")

def test_lnorm_implementation():
    print("Running test_lnorm_implementation...")
    dim = 512
    x = torch.randn(2, 10, dim)
    custom_ln = LayerNorm(dim)
    official_ln = torch.nn.LayerNorm(dim)
    with torch.no_grad():
        nn.init.normal_(custom_ln.weight)
        nn.init.normal_(custom_ln.bias)
        official_ln.weight.copy_(custom_ln.weight)
        official_ln.bias.copy_(custom_ln.bias)
    assert torch.allclose(custom_ln(x), official_ln(x), atol=1e-4)
    print("...passed.")

if __name__ == '__main__':
    test_standard_causal_mask()
    test_ignore_padding()
    test_isolated_subsequence_mask()
    test_bidirectional_simple()
    test_bidirectional_only()
    test_bidirectional_with_isolation()
    test_lnorm_implementation()
    print("\nAll tests passed!")
