"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from dataclasses import dataclass
from typing import Any, Generator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanochat.common import COMPUTE_DTYPE, get_dist_info, print0
from nanochat.flash_attention import flash_attn
from nanochat.optim import DistMuonAdamW, MuonAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def norm(x: Tensor) -> Tensor:
    """Applies RMSNorm to the input tensor without learnable parameters.

    Note: This runs in bf16 when activations are bf16, which seems fine
    empirically.

    Args:
        x: Input tensor of arbitrary shape. Normalization is applied over the
            last dimension.

    Returns:
        Normalized tensor of the same shape and dtype as the input.
    """
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """Linear layer that casts weights to match input dtype in forward.

    Replaces ``torch.amp.autocast``: master weights stay fp32 for optimizer
    precision, but matmuls run in the activation dtype (typically bf16 from
    embeddings).
    """

    def forward(self, x: Tensor) -> Tensor:
        """Applies the linear transformation with dynamic weight casting.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Checks if a GPT layer should have Value Embedding.

    Value Embeddings are applied to alternating layers, with the last layer
    always included.

    Args:
        layer_idx: Index of the layer.
        n_layer: Total number of layers.

    Returns:
        True if the layer should have Value Embedding.
    """
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Applies rotary positional embeddings to the input tensor.

    Args:
        x: Input tensor of shape (B, T, H, D).
        cos: Cosine component of rotary embeddings.
        sin: Sine component of rotary embeddings.

    Returns:
        Tensor with rotary embeddings applied.
    """
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with GQA, sliding windows, and value residual.

    Supports Group-Query Attention (GQA) where ``n_kv_head < n_head``,
    Flash Attention 3 on Hopper+ GPUs with SDPA fallback, and optional
    sliding window attention via the ``window_size`` parameter.

    Attributes:
        layer_idx: Index of this layer within the transformer stack.
        n_head: Number of query attention heads.
        n_kv_head: Number of key/value attention heads (for GQA).
        n_embd: Model embedding dimension.
        head_dim: Dimension of each attention head.
        c_q: Query projection.
        c_k: Key projection.
        c_v: Value projection.
        c_proj: Output projection.
        ve_gate_channels: Number of input channels used for value-embedding gate.
        ve_gate: Learnable gate for value residual (None if layer has no VE).
    """

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        """Initializes the causal self-attention module.

        Args:
            config: Model configuration dataclass.
            layer_idx: Index of this layer in the transformer stack.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        # Each head operates on a slice of the embedding; head_dim = n_embd / n_head.
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # GQA constraint: multiple query heads share each key/value head,
        # so n_head must be a multiple of n_kv_head.
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Q, K, V projections: each maps the full embedding to per-head
        # subspaces. Think of these as learned linear maps that ask "what am
        # I looking for?" (Q), "what do I contain?" (K), and "what do I
        # offer if matched?" (V).
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # Output projection: maps concatenated head outputs back to n_embd.
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # Value residual gate: on alternating layers, a learned gate mixes
        # in a "value embedding" (a direct lookup from the token ID), giving
        # the model a shortcut path for value information that bypasses the
        # learned V projection. The gate uses only the first few channels of
        # the input as features, producing one scalar per KV head.
        self.ve_gate_channels: int = 32
        self.ve_gate: Optional[Linear] = (
            Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(
        self,
        x: Tensor,
        ve: Optional[Tensor],
        cos_sin: tuple[Tensor, Tensor],
        window_size: tuple[int, int],
        kv_cache: Optional[object],
    ) -> Tensor:
        """Computes causal self-attention with optional value residual.

        Args:
            x: Input tensor of shape ``(B, T, n_embd)``.
            ve: Value embedding tensor of shape ``(B, T, n_kv_head * head_dim)``
                or ``None`` if this layer has no value embedding.
            cos_sin: Tuple of ``(cos, sin)`` rotary embedding tensors, each of
                shape ``(1, T, 1, head_dim // 2)``.
            window_size: ``(left, right)`` tuple for sliding window attention.
                ``(-1, 0)`` for full context, ``(N, 0)`` for sliding window of
                size N.
            kv_cache: KV cache object for inference, or ``None`` for training.

        Returns:
            Output tensor of shape ``(B, T, n_embd)``.
        """
        B, T, _ = x.size()  # B=batch, T=sequence length

        # --- Step 1: Project input into Q, K, V subspaces ---
        # Each token's embedding (dim=n_embd) is linearly projected into
        # multiple "heads", each of dimension head_dim. Reshaping to
        # (B, T, H, D) splits the flat projection into separate heads.
        # Flash Attention expects this (B, T, H, D) layout directly.
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # --- Step 2: Value residual (ResFormer) ---
        # On alternating layers, mix in a "value embedding" looked up
        # directly from the token ID. This gives the model a shortcut:
        # useful value information can flow through without being
        # transformed by the V projection. The gate is input-dependent
        # (sigmoid -> range [0, 3]) and per-head, so the model learns
        # how much of the raw embedding to blend in at each position.
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(
                self.ve_gate(x[..., :self.ve_gate_channels])
            )  # (B, T, n_kv_head), range (0, 3)
            # Broadcast gate over head_dim: each head gets one scalar gate
            v = v + gate.unsqueeze(-1) * ve

        # --- Step 3: Rotary positional embeddings (RoPE) ---
        # Standard attention is position-agnostic (permutation-equivariant).
        # RoPE encodes position by rotating Q and K vectors in 2D subspaces
        # of the head dimension. The dot product Q·K then depends on the
        # *relative* distance between positions (rotation difference),
        # giving the model a sense of token ordering without adding
        # explicit positional vectors to the input.
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # --- Step 4: QK normalization + attention sharpening ---
        # Normalize Q and K to unit RMS before computing attention scores.
        # This stabilizes training by preventing the Q·K dot products from
        # growing with head_dim (similar motivation to 1/sqrt(d) scaling in
        # vanilla attention, but more robust). The 1.15x scaling afterward
        # makes attention distributions peakier (more confident), which was
        # found empirically to help.
        q, k = norm(q), norm(k)
        q = q * 1.15
        k = k * 1.15

        # --- Step 5: Compute attention weights and aggregate values ---
        # The core attention operation: softmax(Q @ K^T / sqrt(d)) @ V.
        # "Causal" means each token can only attend to itself and earlier
        # tokens (the upper triangle of the attention matrix is masked to
        # -inf before softmax). window_size limits how far back a token
        # can look, trading off context range for compute.
        # Flash Attention computes this without materializing the full
        # T×T attention matrix, making it O(T) in memory instead of O(T²).
        if kv_cache is None:
            # Training: process all positions at once
            y = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=window_size
            )
        else:
            # Inference: reuse previously computed K, V from a cache so we
            # only compute attention for new tokens, not the full history.
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # The last layer is responsible for advancing the cache position
            # so all layers see consistent positions within one forward pass.
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # --- Step 6: Merge heads and project back ---
        # Concatenate all head outputs back into a single vector of dim
        # n_embd, then apply a learned linear projection. This lets the
        # model combine information gathered by different heads.
        y = y.contiguous().view(B, T, -1)  # (B, T, n_embd)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Feed-forward MLP with squared ReLU activation.

    Uses the expansion ratio of 4x: ``n_embd -> 4 * n_embd -> n_embd``.

    Attributes:
        c_fc: Up-projection from ``n_embd`` to ``4 * n_embd``.
        c_proj: Down-projection from ``4 * n_embd`` back to ``n_embd``.
    """

    def __init__(self, config: GPTConfig) -> None:
        """Initializes the MLP.

        Args:
            config: Model configuration dataclass.
        """
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the MLP: up-project, squared ReLU, down-project.

        Args:
            x: Input tensor of shape ``(B, T, n_embd)``.

        Returns:
            Output tensor of shape ``(B, T, n_embd)``.
        """
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm residual attention and MLP.

    Applies RMSNorm before both the attention and MLP sub-layers, with
    additive residual connections around each.

    Attributes:
        attn: Causal self-attention sub-layer.
        mlp: Feed-forward MLP sub-layer.
    """

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        """Initializes the transformer block.

        Args:
            config: Model configuration dataclass.
            layer_idx: Index of this block in the transformer stack.
        """
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        ve: Optional[Tensor],
        cos_sin: tuple[Tensor, Tensor],
        window_size: tuple[int, int],
        kv_cache: Optional[object],
    ) -> Tensor:
        """Applies attention and MLP with pre-norm residual connections.

        Args:
            x: Input tensor of shape ``(B, T, n_embd)``.
            ve: Value embedding tensor or ``None`` (passed to attention).
            cos_sin: Rotary embedding ``(cos, sin)`` tuple (passed to attention).
            window_size: Sliding window ``(left, right)`` tuple (passed to
                attention).
            kv_cache: KV cache for inference or ``None`` (passed to attention).

        Returns:
            Output tensor of shape ``(B, T, n_embd)``.
        """
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """GPT language model with Flash Attention and sliding window support.

    Key architectural features:
        - Rotary positional embeddings (RoPE) instead of learned positions.
        - QK normalization with learned attention sharpening.
        - Untied token embedding and language model head weights.
        - Squared ReLU activation in the MLP.
        - RMSNorm (no learnable parameters) after embedding and before output.
        - Group-Query Attention (GQA) for efficient inference.
        - Per-layer learnable residual scaling (``resid_lambdas``) and skip
          connections back to the initial embedding (``x0_lambdas``).
        - Value embeddings (ResFormer-style) on alternating layers.
        - Logit softcapping to bound output logits.

    Attributes:
        config: Model configuration.
        window_sizes: Per-layer ``(left, right)`` window size tuples.
        transformer: ModuleDict containing ``wte`` (token embedding) and
            ``h`` (list of transformer blocks).
        lm_head: Language model head projection.
        resid_lambdas: Per-layer residual stream scaling factors.
        x0_lambdas: Per-layer initial-embedding skip connection weights.
        value_embeds: Value embeddings for alternating layers.
        cos: Precomputed cosine rotary embeddings (non-persistent buffer).
        sin: Precomputed sine rotary embeddings (non-persistent buffer).
    """

    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64) -> None:
        """Initializes the GPT model.

        NOTE: This __init__ function runs in meta device context, so calculations
        inside here are shapes and dtypes only, no actual data. We actually
        initialize all data (parameters, buffers, etc.) in init_weights() instead.

        Args:
            config: Model configuration.
            pad_vocab_size_to: Pad vocabulary size to this multiple for efficiency.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1)
            // pad_vocab_size_to * pad_vocab_size_to
        )
        if padded_vocab_size != config.vocab_size:
            print0(
                f"Padding vocab_size from {config.vocab_size} to "
                f"{padded_vocab_size} for efficiency"
            )
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([
                Block(config, layer_idx)
                for layer_idx in range(config.n_layer)
            ]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        # fake init, real init in init_weights()
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        # 10X over-compute should be enough, TODO make nicer?
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # persistent=False means it's not saved to the checkpoint
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initializes the full model weights.

        Weight initialization scheme:
            wte (embedding):     normal, std=1.0
            lm_head:             normal, std=0.001
            for each block:
                attn.c_q:        uniform, std=1/sqrt(n_embd)
                attn.c_k:        uniform, std=1/sqrt(n_embd)
                attn.c_v:        uniform, std=1/sqrt(n_embd)
                attn.c_proj:     zeros
                mlp.c_fc:        uniform, std=1/sqrt(n_embd)
                mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.5, s * 0.5)  # 0.5x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int = 100000,
        device: Optional[torch.device] = None,
    ) -> tuple[Tensor, Tensor]:
        """Precomputes rotary positional embeddings (RoPE).

        Args:
            seq_len: Maximum sequence length to precompute for.
            head_dim: Dimension of each attention head.
            base: Base frequency for the rotary embeddings (theta).
            device: Device to create tensors on. Defaults to the device of
                the token embedding weights.

        Returns:
            Tuple of ``(cos, sin)`` tensors, each of shape
            ``(1, seq_len, 1, head_dim // 2)``, ready for broadcasting over
            batch and head dimensions.
        """
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(
            0, head_dim, 2, dtype=torch.float32, device=device
        )
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        # Add batch and head dims for broadcasting: (1, T, 1, head_dim // 2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config: GPTConfig) -> list[tuple[int, int]]:
        """Computes per-layer window sizes for sliding window attention.

        Args:
            config: Model configuration.

        Returns:
            List of (left, right) tuples for FA3's window_size parameter:
            - left: how many tokens before current position to attend to (-1 = unlimited)
            - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        # Map characters to window sizes
        long_window = config.sequence_len
        # Ceil to FA3 tile size (e.g., 2048 -> 768)
        short_window = -(-long_window // 3 // 128) * 128
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self) -> torch.device:
        """Returns the device of the model."""
        return self.transformer.wte.weight.device

    def estimate_flops(self) -> int:
        """Returns the estimated FLOPs per token for the model (forward + backward).

        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +)
        in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation:
        https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4

        On top of that, 12 * h * q * effective_seq_len accounts for key @ query
        matmul flops inside attention. With sliding windows, effective_seq_len
        varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).

        This is ~1% off from the exact formulas of Chinchilla paper:
        - Chinchilla counts the embedding layer as flops (we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax (we ignore)

        Returns:
            Estimated FLOPs per token.
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(
            ve.weight.numel() for ve in self.value_embeds.values()
        )
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self) -> dict[str, int]:
        """Returns detailed parameter counts for scaling law analysis.

        Different papers use different conventions for which parameters to
        count:
            - Kaplan et al. excluded embedding parameters.
            - Chinchilla included all parameters.

        References:
            - https://arxiv.org/abs/2203.15556 (Chinchilla paper)
            - https://arxiv.org/abs/2001.08361 (Kaplan et al. scaling laws)

        Returns:
            Dict with counts for each parameter group (``wte``,
            ``value_embeds``, ``lm_head``, ``transformer_matrices``,
            ``scalars``, ``total``), so downstream analysis can experiment
            with which combination gives the cleanest scaling laws.
        """
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(
            p.numel() for p in self.transformer.h.parameters()
        )
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), (
            "Parameter count mismatch"
        )
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        scalar_lr: float = 0.5,
    ) -> MuonAdamW | DistMuonAdamW:
        """Creates and returns the combined MuonAdamW optimizer.

        Separates parameters into groups with different optimization strategies:
            - **AdamW**: embeddings, lm_head, and per-layer scalars
              (``resid_lambdas``, ``x0_lambdas``).
            - **Muon**: transformer matrix parameters (attention and MLP
              weights), grouped by shape for efficient stacking.

        Learning rates for AdamW groups are scaled by ``1 / sqrt(n_embd / 768)``
        (muP-style scaling, tuned at d12=768).

        Args:
            unembedding_lr: Learning rate for the language model head.
            embedding_lr: Learning rate for token and value embeddings.
            matrix_lr: Learning rate for transformer matrix parameters (Muon).
            weight_decay: Cautious weight decay for Muon parameters.
            scalar_lr: Learning rate for per-layer scalar parameters.

        Returns:
            Configured optimizer (``DistMuonAdamW`` if DDP, else ``MuonAdamW``).
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        total_groups = (
            len(matrix_params) + len(embedding_params) + len(lm_head_params)
            + len(value_embeds_params) + len(resid_params) + len(x0_params)
        )
        assert len(list(self.parameters())) == total_groups

        # Scale LR for AdamW parameters by 1/sqrt(d_model) (tuned at d12=768)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters "
            f"∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
        )

        # Build param_groups with all required fields explicit
        adam_lr = dmodel_lr_scale
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(
                kind="adamw", params=lm_head_params,
                lr=unembedding_lr * adam_lr,
                betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01,
            ),
            dict(
                kind="adamw", params=embedding_params,
                lr=embedding_lr * adam_lr,
                betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001,
            ),
            dict(
                kind="adamw", params=value_embeds_params,
                lr=embedding_lr * adam_lr * 0.5,
                betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01,
            ),
            dict(
                kind="adamw", params=resid_params,
                lr=scalar_lr * 0.01,
                betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05,
            ),
            dict(
                kind="adamw", params=x0_params,
                lr=scalar_lr,
                betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0,
            ),  # higher beta1 for x0
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind="muon", params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9,
                weight_decay=weight_decay,
            ))

        factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        kv_cache: Optional[object] = None,
        loss_reduction: str = "mean",
    ) -> Tensor:
        """Runs the full GPT forward pass.

        In training mode (``targets`` provided), returns the cross-entropy loss.
        In inference mode (no ``targets``), returns softcapped logits.

        The forward pass proceeds as:
            1. Token embedding + RMSNorm.
            2. Transformer blocks with per-layer residual scaling
               (``resid_lambdas``) and skip connections to the initial
               embedding (``x0_lambdas``).
            3. Final RMSNorm.
            4. Language model head projection + logit softcapping.

        Args:
            idx: Input token IDs of shape ``(B, T)``.
            targets: Target token IDs of shape ``(B, T)`` for training.
                Use ``-1`` to mask positions from the loss. ``None`` for
                inference.
            kv_cache: KV cache object for efficient autoregressive inference,
                or ``None`` for training.
            loss_reduction: Reduction mode for cross-entropy loss
                (``"mean"`` or ``"none"``).

        Returns:
            If ``targets`` is provided: scalar loss tensor (or per-token losses
            if ``loss_reduction="none"``).
            If ``targets`` is ``None``: logit tensor of shape
            ``(B, T, vocab_size)``.
        """
        B, T = idx.size()

        # Fetch rotary embeddings for the current sequence length
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: "
            f"{T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: "
            f"{idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == COMPUTE_DTYPE, (
            f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        )
        # Offset rotary embeddings when using KV cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T]

        # Embed tokens and normalize
        x = self.transformer.wte(idx)
        # Ensure activations are in compute dtype (no-op for bf16, active
        # for fp16 code path)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual

        # Forward through transformer blocks with per-layer scaling
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = (
                self.value_embeds[str(i)](idx).to(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Language model head with logit softcapping
        softcap = 20
        logits = self.lm_head(x)  # (B, T, padded_vocab_size)
        logits = logits[..., :self.config.vocab_size]  # remove vocab padding
        logits = logits.float()  # fp32 for softcap and loss
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            # Training: compute and return cross-entropy loss
            # TODO: experiment with chunked cross-entropy?
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # Inference: return logits directly
            return logits

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
    ) -> Generator[Any, None, None]:
        """Generates tokens autoregressively.

        Naive autoregressive streaming inference. Assumes batch size is 1 and
        ids and yielded tokens are simple Python lists and ints.

        Args:
            tokens: Input token IDs.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k sampling parameter.
            seed: Random seed for sampling.

        Yields:
            Generated token IDs one at a time.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
