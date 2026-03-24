# -*- coding: utf-8 -*-
"""
CLAPEncoder Backbone for LibFewShot

A configurable CLAP backbone with freeze control (full/partial) and
optional spatial projection for mode-2 (non-flatten) classifiers.

The CLAP model produces 512-dimensional audio embeddings. When
is_flatten=True (mode-1), the 512-d vector is returned directly.
When is_flatten=False (mode-2), a learned linear projection reshapes
the embedding into a [B, C, H, W] spatial feature map.

Usage:
    backbone:
        name: CLAPEncoder
        kwargs:
            is_flatten: True
            freeze_mode: full
"""

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import sys

# Don't import laion_clap at module level to avoid numba/trainer conflict.
# See _import_laion_clap() docstring for the full explanation.
CLAP_AVAILABLE: Optional[object] = None  # Will be checked on first use


def _import_laion_clap() -> Union[object, bool]:
    """Lazy import of laion_clap to avoid module conflicts with numba.

    The issue: numba decorates ``builtins.print`` during initialization, but
    its module resolution can incorrectly find ``libfewshot_core.trainer``
    instead of builtins, causing ``AttributeError``.

    Solution: Temporarily remove ``libfewshot_core.*`` from ``sys.modules``
    during import and restore them afterwards.

    Returns:
        The ``laion_clap`` module on success, or ``False`` if the package
        is not installed.
    """
    global CLAP_AVAILABLE

    if CLAP_AVAILABLE is not None:
        return CLAP_AVAILABLE

    # Save and remove libfewshot_core modules to prevent numba confusion
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key.startswith('libfewshot_core'):
            saved_modules[key] = sys.modules.pop(key)

    try:
        import laion_clap
        CLAP_AVAILABLE = laion_clap
        return laion_clap
    except ImportError as e:
        CLAP_AVAILABLE = False
        print(f"Warning: laion-clap not installed. Run: pip install laion-clap")
        print(f"Error: {e}")
        return False
    finally:
        # Restore previously cached libfewshot_core modules
        sys.modules.update(saved_modules)


class CLAPEncoder(nn.Module):
    """CLAP encoder backbone with configurable freezing and spatial projection.

    Args:
        is_flatten: If True (mode-1), return flat 512-d vectors.
            If False (mode-2), project to spatial feature map [B, C, H, W].
        freeze_mode: One of 'full' (freeze entire CLAP), 'partial' (unfreeze
            last ``frozen_layers`` layer groups), or 'none' (all trainable).
        frozen_layers: Number of last layer groups to unfreeze when
            freeze_mode='partial'.
        enable_fusion: Whether to enable fusion in the CLAP model.
        spatial_channels: Channel dimension C for mode-2 spatial output.
        spatial_size: Spatial height/width for mode-2 output (H = W).
        **kwargs: Ignored extra keyword arguments for config compatibility.
    """

    def __init__(
        self,
        is_flatten: bool = True,
        freeze_mode: str = 'full',
        frozen_layers: int = 0,
        enable_fusion: bool = False,
        spatial_channels: int = 64,
        spatial_size: int = 5,
        **kwargs,
    ) -> None:
        super(CLAPEncoder, self).__init__()

        # Lazy import to avoid numba/trainer conflict (see module docstring)
        laion_clap = _import_laion_clap()
        if not laion_clap:
            raise ImportError(
                "laion-clap is required for CLAPEncoder. "
                "Install with: pip install laion-clap"
            )

        self.is_flatten: bool = is_flatten
        self.freeze_mode: str = freeze_mode
        self.frozen_layers: int = frozen_layers
        self.spatial_channels: int = spatial_channels
        self.spatial_size: int = spatial_size

        # Initialize CLAP model and load pretrained weights
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        self.clap_model.load_ckpt()

        # Feature dimension exposed to downstream classifiers (used by LibFewShot)
        if is_flatten:
            # Mode 1: flat 512-d vector for Baseline, BaselinePlus, MetaBaseline, ProtoNet
            self.feat_dim: int = 512
        else:
            # Mode 2: spatial [B, C, H, W] feature map for ADM, DN4
            self.feat_dim = spatial_channels
            self.spatial_proj: nn.Sequential = nn.Sequential(
                nn.Linear(512, spatial_channels * spatial_size * spatial_size),
                nn.ReLU(inplace=True),
            )

        # Apply freeze strategy to CLAP parameters
        self._apply_freeze()

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _get_layer_groups(self) -> list:
        """Return named children of the CLAP audio branch as layer groups.

        Returns:
            List of ``(name, module)`` tuples from the audio branch.
        """
        return list(self.clap_model.model.audio_branch.named_children())

    def _apply_freeze(self) -> None:
        """Freeze CLAP parameters according to ``freeze_mode``.

        Populates ``_frozen_group_names`` which is later used by
        :meth:`train` to keep frozen sub-modules in ``eval()`` mode,
        preserving correct BatchNorm statistics.
        """
        self._frozen_group_names: list = []

        if self.freeze_mode == 'full':
            # Freeze every parameter in the CLAP model
            for param in self.clap_model.parameters():
                param.requires_grad = False
            self._frozen_group_names = [
                name for name, _ in self._get_layer_groups()
            ]

        elif self.freeze_mode == 'partial':
            # Start by freezing everything, then selectively unfreeze
            for param in self.clap_model.parameters():
                param.requires_grad = False

            groups = self._get_layer_groups()
            # Unfreeze the last `frozen_layers` groups for fine-tuning
            groups_to_unfreeze = groups[-self.frozen_layers:] if self.frozen_layers > 0 else []
            unfrozen_names = {name for name, _ in groups_to_unfreeze}

            for name, module in groups_to_unfreeze:
                for param in module.parameters():
                    param.requires_grad = True

            self._frozen_group_names = [
                name for name, _ in groups if name not in unfrozen_names
            ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CLAPEncoder.

        Args:
            x: Pre-extracted CLAP embeddings of shape ``[B, 512]``.

        Returns:
            If ``is_flatten=True``: tensor of shape ``[B, 512]`` (pass-through).
            If ``is_flatten=False``: tensor of shape
            ``[B, spatial_channels, spatial_size, spatial_size]``
            produced by a learned linear projection + reshape.
        """
        if self.is_flatten:
            return x

        B = x.size(0)
        out = self.spatial_proj(x)
        return out.view(B, self.spatial_channels, self.spatial_size, self.spatial_size)

    # ------------------------------------------------------------------
    # Waveform extraction (for clap_data_mode='waveforms')
    # ------------------------------------------------------------------

    def extract_and_forward(
        self, audio_data_list: List, is_train: bool = True
    ) -> torch.Tensor:
        """Extract CLAP embeddings from raw waveforms and apply forward pass.

        Used when ``clap_data_mode='waveforms'`` (partial freeze mode).
        Gradients flow through unfrozen CLAP layers.

        Args:
            audio_data_list: List of numpy arrays, each shape ``[T]``
                (variable-length raw waveforms at 48kHz).
            is_train: Whether to use train mode for unfrozen layers.

        Returns:
            If ``is_flatten=True``: tensor ``[B, 512]``.
            If ``is_flatten=False``: tensor ``[B, C, H, W]``.
        """
        if is_train:
            self.train()
        else:
            self.eval()

        device = next(self.parameters()).device
        embeddings = []
        for y in audio_data_list:
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).float()
            if y.ndim > 1:
                y = y.flatten()
            y = y.to(device)

            emb = self.clap_model.get_audio_embedding_from_data(
                x=y.unsqueeze(0), use_tensor=True
            )
            embeddings.append(emb)

        embeddings = torch.cat(embeddings, dim=0)  # [B, 512]
        return self.forward(embeddings)

    # ------------------------------------------------------------------
    # Train / eval overrides
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "CLAPEncoder":
        """Override train to respect freeze settings.

        Ensures that frozen sub-modules remain in ``eval()`` mode even when
        the overall model is set to training. This is critical for keeping
        BatchNorm running-mean/variance statistics stable in frozen layers.

        Args:
            mode: If ``True``, set to training mode; if ``False``, set to eval.

        Returns:
            Self.
        """
        super().train(mode)

        if self.freeze_mode == 'full':
            # Keep the entire CLAP model in eval regardless of outer mode
            self.clap_model.eval()
        elif self.freeze_mode == 'partial':
            # Only keep frozen layer groups in eval; unfrozen groups follow mode
            frozen_set = set(self._frozen_group_names)
            for name, module in self.clap_model.model.audio_branch.named_children():
                if name in frozen_set:
                    module.eval()

        return self
