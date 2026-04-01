import inspect
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

flash_attn_backend: Optional[str] = None
flash_attn_error: Optional[str] = None

try:
    # flash-attn 1.x (preferred - simpler public API)
    from flash_attn.flash_attention import FlashMHA as _FlashMHA1

    _FlashMHA2 = None
    flash_attn_backend = "fa1"
    flash_attn_error = None
except (ImportError, OSError) as exc:
    _FlashMHA1 = None
    flash_attn_error = str(exc)
    try:
        # flash-attn 2.x (fallback - use modules.mha.MHA)
        from flash_attn.modules.mha import MHA as _FlashMHA2

        flash_attn_backend = "fa2"
        flash_attn_error = None
    except (ImportError, OSError) as exc:
        _FlashMHA2 = None
        flash_attn_error = str(exc)

flash_attn_available = flash_attn_backend is not None


def get_flash_attn_info() -> str:
    """
    Return a human-readable string describing the detected flash-attn backend.
    
    Returns:
        String describing the backend (e.g., "flash-attn 1.x", "flash-attn 2.x", 
        or "not installed").
        
    Note:
        FA1 is preferred when available for its simpler API.
        FA2 is used as fallback with a simplified wrapper (no rotary, MQA/GQA, etc.).
    """
    if flash_attn_backend == "fa1":
        return "flash-attn 1.x (preferred - FlashMHA with explicit FlashAttention wrapper)"
    elif flash_attn_backend == "fa2":
        return "flash-attn 2.x (fallback - MHA module with simplified wrapper interface)"
    elif flash_attn_error is not None:
        return f"flash-attn import failed: {flash_attn_error}"
    else:
        return "flash-attn not installed"


class FlashMHA(nn.Module):
    """
    Compatibility wrapper for flash-attn 1.x and 2.x MHA classes.
    
    **API Preference Strategy:**
    - **FA1 (Preferred):** Uses the simple public API from flash_attn.flash_attention.FlashMHA.
      This provides a straightforward interface for basic use cases.
    - **FA2 (Fallback):** Uses flash_attn.modules.mha.MHA but with a simplified wrapper that
      avoids FA2-specific complexity (rotary embeddings, MQA/GQA, varlen sequences, etc.)
      unless explicitly required. This ensures compatibility when only FA2 is available.

    Both backends perform optimized attention computation:
    - FA1: Explicit FlashAttention wrapper that calls flash_attn_unpadded_qkvpacked_func
      with padding/unpadding (unpad_input, pad_input) for efficient mask handling.
    - FA2: MHA module uses the same padding optimization internally. Our wrapper
      uses simplified configuration (use_flash_attn=True with key_padding_mask=None,
      falling back to dense if mask is provided) to maintain correctness and avoid
      FA2's advanced feature constraints.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        """
        Initialize FlashMHA with parameters compatible with FA1 API.
        
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            bias: Whether to use bias in projections (default: True).
            batch_first: Whether input is (batch, seq_len, hidden) (default: True).
            attention_dropout: Dropout probability in attention (default: 0.0).
            causal: Whether to use causal masking (default: False).
            device: Device for model parameters.
            dtype: Data type for model parameters.
            **kwargs: Additional backend-specific parameters (mostly for FA2 fallback).
            
        Implementation Details:
            - **FA1 path:** Directly instantiates FlashMHA from flash_attn.flash_attention.
            - **FA2 path:** Instantiates MHA from flash_attn.modules.mha with simplified
              configuration. Constructor parameters are introspected to avoid passing
              unsupported arguments (different FA2 versions vary in their accepted kwargs).
        
        Notes on key_padding_mask:
            - Uses flash-attn convention: True=keep/valid token, False=masked token.
            - This matches FA1 unpadding utilities and FA2 MHA behavior.
            - Callers that use PyTorch Transformer masks (True=pad) should invert
                the mask before calling this wrapper.
        """
        super().__init__()
        if not batch_first:
            raise ValueError("FlashMHA wrapper currently supports batch_first=True only.")

        if not flash_attn_available:
            raise ImportError(
                "flash_attn is not installed. Install flash-attn 1.x or 2.x to use "
                "fast transformer backend."
            )

        self.batch_first = batch_first
        self.backend = flash_attn_backend
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Validate dimensions
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        factory_kwargs = {"device": device, "dtype": dtype}

        if self.backend == "fa1":
            # FA1: Direct instantiation with simple API
            self._impl = _FlashMHA1(
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=bias,
                batch_first=batch_first,
                attention_dropout=attention_dropout,
                causal=causal,
                **factory_kwargs,
                **kwargs,
            )
        else:
            # FA2: Simplified wrapper - introspect constructor to avoid unsupported args
            # (signature varies across 2.x builds).
            sig_params = inspect.signature(_FlashMHA2.__init__).parameters
            impl_kwargs = {
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "causal": causal,
                **factory_kwargs,
            }

            if "qkv_proj_bias" in sig_params:
                impl_kwargs["qkv_proj_bias"] = bias
            if "out_proj_bias" in sig_params:
                impl_kwargs["out_proj_bias"] = bias
            if "bias" in sig_params:
                impl_kwargs["bias"] = bias
            if "dropout" in sig_params:
                impl_kwargs["dropout"] = attention_dropout
            if "attention_dropout" in sig_params:
                impl_kwargs["attention_dropout"] = attention_dropout
            if "batch_first" in sig_params:
                impl_kwargs["batch_first"] = batch_first
            impl_kwargs.update(kwargs)

            # FA2 MHA: handles QKV projection, attention, and output projection internally.
            self._impl = _FlashMHA2(**impl_kwargs)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with support for key padding mask.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            key_padding_mask: Optional bool mask of shape (batch, seq_len).
                             True indicates valid tokens to keep.
                             False indicates masked / padding tokens.
                             This follows flash-attn convention (FA1 and FA2).
                             If your upstream mask uses PyTorch Transformer semantics
                             (True=pad), invert it before passing to FlashMHA.
            need_weights: Whether to return attention weights (rarely supported 
                         efficiently in flash attention). Default: False.
            **kwargs: Additional backend-specific arguments.
            
        Returns:
            Tuple[output, attn_weights_or_none]:
                - output: Attention output of shape (batch, seq_len, embed_dim).
                - attn_weights_or_none: None (flash attention doesn't expose weights efficiently).
                
        Notes:
              * Both backends perform padding-aware attention internally:
              * Unpad input to remove padding tokens
              * Run flash attention on the packed sequence
              * Pad output back to original shape
            - This is much more efficient than masking directly.
        """
        orig_dtype = x.dtype

        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.bool()

        if self.backend == "fa1":
            # FA1: Use need_weights parameter if supported
            try:
                out = self._impl(
                    x,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    **kwargs,
                )
            except TypeError as e:
                # Some flash-attn 1.x builds may not expose need_weights.
                if "need_weights" in str(e):
                    out = self._impl(
                        x,
                        key_padding_mask=key_padding_mask,
                        **kwargs,
                    )
                else:
                    raise
        else:
            # FA2: Simplified wrapper - pass only supported arguments
            out = self._impl(
                x,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if isinstance(out, tuple):
            out_tensor, attn_weights = out
            if out_tensor.dtype != orig_dtype:
                out_tensor = out_tensor.to(orig_dtype)
            return out_tensor, attn_weights
        
        # Normalize output to always be (output, attn_weights_or_none)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out, None