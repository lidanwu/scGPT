"""
Test suite for flash-attn compatibility wrapper.

This validates that the FlashMHA wrapper correctly supports both
flash-attn 1.x and 2.x APIs, especially for the CUDA 12.8 + flash-attn 2.8.x
upgrade path.
"""

import pytest
import torch
import torch.nn as nn

from scgpt.model.flash_attn_compat import (
    FlashMHA,
    flash_attn_available,
    flash_attn_backend,
    get_flash_attn_info,
)


def test_flash_attn_availability():
    """Test that flash-attn availability flag is set correctly."""
    print(f"\nFlash-attn backend: {get_flash_attn_info()}")
    # If flash-attn is installed, backend should be "fa1" or "fa2"
    # If not installed, tests will be skipped via pytest.mark.skipif


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
def test_flash_mha_instantiation():
    """Test FlashMHA instantiation with standard parameters."""
    embed_dim = 256
    num_heads = 8

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.1,
        causal=False,
    )

    assert hasattr(mha, "batch_first")
    assert hasattr(mha, "backend")
    assert hasattr(mha, "embed_dim")
    assert mha.embed_dim == embed_dim
    assert mha.num_heads == num_heads


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
def test_flash_mha_dimension_validation():
    """Test that invalid embed_dim/num_heads raises error."""
    with pytest.raises(ValueError, match="must be divisible by"):
        FlashMHA(
            embed_dim=255,  # Not divisible by 8
            num_heads=8,
            batch_first=True,
        )


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
def test_flash_mha_batch_first_only():
    """Test that batch_first=False is not supported."""
    with pytest.raises(ValueError, match="batch_first=True"):
        FlashMHA(
            embed_dim=256,
            num_heads=8,
            batch_first=False,
        )


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_mha_forward_no_mask():
    """Test forward pass without padding mask on CUDA."""
    embed_dim = 256
    num_heads = 8
    batch_size = 2
    seq_len = 16
    device = "cuda"

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.1,
    )
    mha = mha.to(device)

    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    with torch.no_grad():
        out, attn_weights = mha(x)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert attn_weights is None, "Flash attention should not return weights"


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_mha_forward_with_mask():
    """Test forward pass with key padding mask on CUDA."""
    embed_dim = 256
    num_heads = 8
    batch_size = 2
    seq_len = 16
    device = "cuda"

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.1,
    )
    mha = mha.to(device)

    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    # Mark last 2 tokens as padding (True = pad, False = real)
    key_padding_mask[:, -2:] = True

    with torch.no_grad():
        out, attn_weights = mha(x, key_padding_mask=key_padding_mask)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert attn_weights is None, "Flash attention should not return weights"


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_mha_mask_effect():
    """Test that applied mask actually affects output."""
    embed_dim = 256
    num_heads = 8
    batch_size = 1
    seq_len = 8
    device = "cuda"

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,  # No dropout for determinism
    )
    mha = mha.to(device)
    mha.eval()

    x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=False)

    # Get output without mask
    with torch.no_grad():
        out_no_mask, _ = mha(x)

    # Get output with mask
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    key_padding_mask[:, -2:] = True
    with torch.no_grad():
        out_with_mask, _ = mha(x, key_padding_mask=key_padding_mask)

    # Outputs should differ (masked attention should ignore padded tokens)
    # Use a generous tolerance since different backends may have slight differences
    assert not torch.allclose(
        out_no_mask, out_with_mask, atol=1e-2
    ), "Mask did not affect output as expected"


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_mha_causal():
    """Test causal attention mode."""
    embed_dim = 256
    num_heads = 8
    batch_size = 2
    seq_len = 16
    device = "cuda"

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        causal=True,
    )
    mha = mha.to(device)

    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    with torch.no_grad():
        out, attn_weights = mha(x)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
def test_backend_detection():
    """Test that correct backend is detected."""
    # Just verify that get_flash_attn_info returns a meaningful string
    info = get_flash_attn_info()
    assert isinstance(info, str)
    assert len(info) > 0
    # Should mention either FA1, FA2, or "not installed"
    assert any(
        x in info for x in ["flash-attn 1", "flash-attn 2", "not installed"]
    ), f"Unexpected backend info: {info}"
