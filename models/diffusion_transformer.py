"""PyTorch implementation for Diffusion Transformer (DiT)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_sinusoidal_embedding(
    timestep: torch.Tensor, embedding_dim: int
) -> torch.Tensor:
    """Get sinusoidal positional embedding for timesteps.

    Args:
        timestep: Timestep tensor of shape (batch_size,) with values in [0, 1].
        embedding_dim: The dimension of the embedding.

    Returns:
        Sinusoidal embedding tensor of shape (batch_size, embedding_dim).
    """
    frequencies = torch.exp(
        torch.arange(0, embedding_dim, 2, device=timestep.device)
        * -(math.log(10000) / embedding_dim)
    )
    angles = timestep.unsqueeze(1) * frequencies
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return embedding


class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, config):
        """Initialize self-attention module.

        Args:
            config: Configuration object containing:
                - num_heads: Number of attention heads.
                - embedding_dim: Dimension of embeddings (must be divisible by num_heads).
        """
        super().__init__()
        self.config = config

        self._num_heads = config.num_heads
        self._embedding_dim = config.embedding_dim

        if self._embedding_dim % self._num_heads:
            raise ValueError(
                "Embedding dimension needs to be divisible by number of attention heads."
            )

        self._head_dim = self._embedding_dim // self._num_heads

        self._dropout = nn.Dropout(p=0.1)
        self._q_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self._k_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self._v_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self._out_proj = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention to input.

        Args:
            x: Input tensor of shape (batch_size, sequence_len, embedding_dim).

        Returns:
            Output tensor of shape (batch_size, sequence_len, embedding_dim).
        """
        B = x.size(0)

        # q: [B, S, D] --> [B, H, S, D_h]
        q = self._q_proj(x)
        q = q.view(B, -1, self._num_heads, self._head_dim).transpose(1, 2)

        # k: [B, S, D] --> [B, H, S, D_h]
        k = self._k_proj(x)
        k = k.view(B, -1, self._num_heads, self._head_dim).transpose(1, 2)

        # v: [B, S, D] --> [B, H, S, D_h]
        v = self._v_proj(x)
        v = v.view(B, -1, self._num_heads, self._head_dim).transpose(1, 2)

        # attn: [B, H, S, D_h] --> [B, H, S, S]
        attn = (q @ k.transpose(-1, -2)) / self._head_dim**0.5
        attn = F.softmax(attn, dim=-1)
        attn = self._dropout(attn)

        # out: [B, H, S, S] --> [B, H, S, D_h]
        out = attn @ v

        # out: [B, H, S, D_h] --> [B, S, D]
        out = out.transpose(1, 2).contiguous().view(B, -1, self._embedding_dim)

        # out: [B, S, D] --> [B, S, D]
        return self._out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with adaptive layer normalization and self-attention."""

    def __init__(self, config):
        """Initialize transformer block.

        Args:
            config: Configuration object containing:
                - embedding_dim: Dimension of embeddings.
                - num_heads: Number of attention heads.
        """
        super().__init__()

        self.mha = SelfAttention(config)
        self.ln_mha = nn.LayerNorm(config.embedding_dim, elementwise_affine=False)

        self.ada_ln_zero_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(config.embedding_dim, config.embedding_dim * 6)
        )

        # Zero initialize for accelerated large scale training
        nn.init.constant_(self.ada_ln_zero_mlp[-1].weight, 0.0)
        nn.init.constant_(self.ada_ln_zero_mlp[-1].bias, 0.0)

        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
        )
        self.ln_mlp = nn.LayerNorm(config.embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with residual connections.

        Args:
            x: Input tensor of shape (batch_size, sequence_len, embedding_dim).
            cond: Conditioning tensor of shape (batch_size, embedding_dim).

        Returns:
            Output tensor of shape (batch_size, sequence_len, embedding_dim).
        """
        # pre-layer norm for better gradient flow
        x_norm = self.ln_mha(x)

        # compute conditional scale and shift parameters, chunk across embedding dim
        mha_gamma, mha_beta, mha_alpha, mlp_gamma, mlp_beta, mlp_alpha = (
            self.ada_ln_zero_mlp(cond).chunk(6, dim=-1)
        )

        # scale + shift x_norm
        x_norm = x_norm * (1 + mha_gamma) + mha_beta

        # attn_out: [B, S, D] --> [B, S, D]
        mha_out = self.mha(x_norm) * (1 + mha_alpha)

        # first residual connection
        x = x + mha_out

        # pre-layer norm for better gradient flow
        x_norm = self.ln_mlp(x)

        # scale + shift x_norm
        x_norm = x_norm * (1 + mlp_gamma) + mlp_beta

        # mlp_out: [B, S, D] --> [B, S, D]
        mlp_out = self.mlp(x_norm) * (1 + mlp_alpha)

        return x + mlp_out


class PatchEmbedding(nn.Module):
    """Patch embedding layer for vision transformers."""

    def __init__(self, config):
        """Initialize patch embedding layer.

        Args:
            config: Configuration object containing:
                - input_dim: Number of input channels.
                - embedding_dim: Dimension of patch embeddings.
                - patch_size: Size of each patch (assumes square patches).
                - sequence_len: Expected sequence length (for reference).
        """
        super().__init__()

        self._input_dim = config.input_dim
        self._embedding_dim = config.embedding_dim
        self._patch_size = config.patch_size
        self._sequence_len = config.sequence_len

        self._conv = nn.Conv2d(
            self._input_dim,
            self._embedding_dim,
            kernel_size=self._patch_size,
            stride=self._patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image patches to embeddings with CLS token.

        Args:
            x: Input image tensor of shape (batch_size, input_dim, height, width).
                Height and width must be divisible by patch_size.

        Returns:
            Patch embeddings with CLS token of shape (batch_size, num_patches + 1, embedding_dim).
        """
        B = x.size(0)
        # x: [B, input_dim, H, W] --> [B, D, H', W']
        x = self._conv(x)
        # Flatten spatial dimensions: [B, D, H', W'] --> [B, D, H' * W'] --> [B, H' * W', D]
        x = x.flatten(2).transpose(1, 2)

        return x


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer (DiT) for flow matching."""

    def __init__(self, config):
        """Initialize Diffusion Transformer.

        Args:
            config: Configuration object containing:
                - embedding_dim: Dimension of embeddings.
                - sequence_len: Sequence length (number of patches).
                - patch_size: Size of each patch.
                - input_dim: Number of input channels.
                - num_transformer_blocks: Number of transformer blocks.
        """
        super().__init__()
        self.config = config
        self._embedding_dim = config.embedding_dim
        self._num_classes = config.num_classes
        self._sequence_len = config.sequence_len
        self._patch_size = config.patch_size
        self._input_dim = config.input_dim

        # We use a fixed learned positional embedding since we are working with CIFAR10
        self._pos_embedding = nn.Parameter(
            torch.randn(1, self._sequence_len, self._embedding_dim)
        )

        # Fixed class embeddings given vision datasets
        self._class_embedding = nn.Embedding(self._num_classes, self._embedding_dim)

        # Setup patch embedding layer
        self._patch_emb = PatchEmbedding(config)

        # Setup time embedding layer
        self._time_mlp = nn.Sequential(
            nn.Linear(self._embedding_dim, self._embedding_dim * 4),
            nn.GELU(),
            nn.Linear(self._embedding_dim * 4, self._embedding_dim),
        )

        # Setup transformer backbone
        self._transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_transformer_blocks)]
        )

        self._final_norm = nn.LayerNorm(config.embedding_dim)
        self._out_proj = nn.Linear(
            self._embedding_dim, self._patch_size**2 * self._input_dim
        )

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch embeddings back to image format.

        Args:
            x: Patch embeddings tensor of shape (batch_size, num_patches, patch_dim).

        Returns:
            Reconstructed image tensor of shape (batch_size, input_dim, height, width).
        """
        B, T, _ = x.size()
        P = self._patch_size
        C = self._input_dim
        H = W = int(T**0.5) * P

        x = x.reshape(B, H // P, W // P, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return x

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the diffusion transformer.

        Args:
            x: Noisy images of shape (batch_size, input_dim, height, width).
            t: Timesteps of shape (batch_size,) with values in [0, 1] for flow matching.
            y: Class label of shape (batch_size,) with values in [0, num_classes] for flow matching.

        Returns:
            Predicted velocity field of shape (batch_size, input_dim, height, width).
        """
        # Compute image and positional embeddings
        x = self._patch_emb(x)
        x = x + self._pos_embedding

        # Compute time embeddings
        t_emb = get_sinusoidal_embedding(t, self._embedding_dim)
        t_emb = self._time_mlp(t_emb)

        y_emb = self._class_embedding(y)
        cond = t_emb + y_emb

        for transformer_block in self._transformer_blocks:
            x = transformer_block(x, cond)

        out = self._out_proj(self._final_norm(x))

        return self._unpatchify(out)
