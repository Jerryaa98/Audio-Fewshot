# -*- coding: utf-8 -*-
"""
CLAP (Contrastive Language-Audio Pretraining) Backbone for LibFewShot

This backbone uses LAION-CLAP to extract audio embeddings.
The CLAP model produces 512-dimensional audio embeddings.

Note: CLAP expects audio at 48kHz sample rate. The model's 
get_audio_embedding_from_filelist method automatically handles 
resampling from different sample rates to 48kHz.

Usage:
    backbone:
        name: CLAPBackbone
        kwargs:
            enable_fusion: False
            checkpoint_path: null  # Optional: path to fine-tuned checkpoint
"""

import torch
import torch.nn as nn
import numpy as np
import sys

# Don't import laion_clap at module level to avoid numba/trainer conflict
# numba (imported by librosa, imported by laion_clap) tries to access builtins.print
# but can incorrectly resolve to libfewshot_core.trainer module
CLAP_AVAILABLE = None  # Will be checked on first use


def _import_laion_clap():
    """
    Lazy import of laion_clap to avoid module conflicts with numba.
    
    The issue: numba decorates builtins.print during initialization, but its module
    resolution can incorrectly find libfewshot_core.trainer instead of builtins,
    causing AttributeError: module 'libfewshot_core.trainer' has no attribute 'use_logger'
    
    Solution: Temporarily remove libfewshot_core.* from sys.modules during import.
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
        # Restore modules
        sys.modules.update(saved_modules)


class CLAPBackbone(nn.Module):
    """
    CLAP Backbone for audio few-shot learning.
    
    CLAP (Contrastive Language-Audio Pretraining) produces 512-dimensional
    embeddings from audio files. This backbone wraps the LAION-CLAP model
    for use in the LibFewShot framework.
    
    Args:
        enable_fusion: Whether to enable fusion in CLAP model (default: False)
        checkpoint_path: Optional path to fine-tuned CLAP checkpoint
        embedding_dim: Output embedding dimension (default: 512)
        device: Device to load model on (default: 'cuda')
    """
    
    def __init__(
        self,
        enable_fusion: bool = False,
        checkpoint_path: str = None,
        embedding_dim: int = 512,
        device: str = 'cuda',
        **kwargs
    ):
        super(CLAPBackbone, self).__init__()
        
        # Lazy import
        laion_clap = _import_laion_clap()
        if not laion_clap:
            raise ImportError(
                "laion-clap is required for CLAPBackbone. "
                "Install with: pip install laion-clap"
            )
        
        self.enable_fusion = enable_fusion
        self.checkpoint_path = checkpoint_path
        self.embedding_dim = embedding_dim
        self._device = device
        
        # Initialize CLAP model
        self.clap_model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion,
            device=device
        )
        self.clap_model.load_ckpt()  # Load default pretrained checkpoint
        
        # Load fine-tuned weights if provided
        # if checkpoint_path is not None:
        #     self._load_finetuned_weights(checkpoint_path)
        
        # CLAP parameters are trainable by default
        # Call freeze_clap() if you want to use CLAP as frozen feature extractor
        
        # Feature dimension for downstream classifiers
        self.feat_dim = embedding_dim
    
    def freeze_clap(self):
        """Freeze CLAP parameters (use as frozen feature extractor)."""
        for param in self.clap_model.parameters():
            param.requires_grad = False
        print("CLAP backbone frozen (parameters will not be updated)")
    
    def unfreeze_clap(self):
        """Unfreeze CLAP parameters (allow fine-tuning)."""
        for param in self.clap_model.parameters():
            param.requires_grad = True
        print("CLAP backbone unfrozen (parameters will be updated)")
        
    def _load_finetuned_weights(self, checkpoint_path: str):
        """Load fine-tuned CLAP weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        input('bla bla')
        
        if 'model_state_dict' in checkpoint:
            self.clap_model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.clap_model.model.load_state_dict(checkpoint)
        
        print(f"Loaded fine-tuned CLAP from {checkpoint_path}")
        if 'val_acc' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  Trained for {checkpoint['epoch']} epochs")
    
    def forward(self, x):
        """
        Forward pass to extract CLAP embeddings.
        
        Args:
            x: Input audio data. Can be:
               - List of audio file paths (strings)
               - Tensor of pre-processed audio waveforms [B, samples] or [B, 1, samples]
               - Tensor of pre-extracted embeddings [B, embedding_dim]
               
        Returns:
            Tensor of embeddings [B, embedding_dim]
        """
        # If input is already embeddings (pre-extracted), just return them
        if isinstance(x, torch.Tensor):
            if x.dim() == 2 and x.shape[-1] == self.embedding_dim:
                # Already embeddings
                return x
            elif x.dim() == 1 and x.shape[0] == self.embedding_dim:
                # Single embedding
                return x.unsqueeze(0)
            elif x.dim() == 2 or x.dim() == 3:
                # Audio waveform tensor - need to convert to numpy for CLAP
                # CLAP expects [B, samples] or processes files directly
                raise NotImplementedError(
                    "Direct waveform input not supported. "
                    "Please use pre-extracted embeddings or file paths."
                )
        
        # If input is file paths
        if isinstance(x, (list, tuple)) and all(isinstance(p, str) for p in x):
            return self._extract_from_files(x)
        
        raise ValueError(
            f"Unsupported input type: {type(x)}. "
            f"Expected: list of file paths, or pre-extracted embeddings tensor."
        )
    
    def _extract_from_files(self, audio_paths: list):
        """
        Extract CLAP embeddings from audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            Tensor of embeddings [B, embedding_dim]
        """
        with torch.no_grad():
            embeddings = self.clap_model.get_audio_embedding_from_filelist(
                x=audio_paths,
                use_tensor=True
            )
        return embeddings
    
    def extract_embeddings_batch(self, audio_paths: list):
        """
        Extract CLAP embeddings for a batch of audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            numpy array of embeddings [batch_size, embedding_dim]
        """
        with torch.no_grad():
            embeddings = self.clap_model.get_audio_embedding_from_filelist(
                x=audio_paths,
                use_tensor=True
            )
        return embeddings.cpu().numpy()
    
    def extract_single_embedding(self, audio_path: str):
        """
        Extract CLAP embedding for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            numpy array of embedding [embedding_dim]
        """
        with torch.no_grad():
            embedding = self.clap_model.get_audio_embedding_from_filelist(
                x=[audio_path],
                use_tensor=True
            )
        return embedding.cpu().numpy().squeeze()
    
    def get_text_embedding(self, texts: list):
        """
        Get text embeddings from CLAP.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of embeddings [B, embedding_dim]
        """
        with torch.no_grad():
            embeddings = self.clap_model.get_text_embedding(
                texts,
                use_tensor=True
            )
        return embeddings
    
    def extract_embeddings_from_audio_paths(self, audio_data: list, is_train: bool = False):
        """
        Extract CLAP embeddings from a batch of audio file paths.
        
        This function reads audio files from the given paths and returns their
        corresponding embeddings extracted by the CLAP model.
        
        Args:
            audio_paths: List of paths to audio files (e.g., ['/path/to/audio1.wav', '/path/to/audio2.wav'])
            
        Returns:
            torch.Tensor: Batch of embeddings with shape [batch_size, embedding_dim]
                          where embedding_dim is typically 512 for CLAP
        
        Example:
            >>> backbone = CLAPBackbone(device='cuda')
            >>> paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
            >>> embeddings = backbone.extract_embeddings_from_audio_paths(paths)
            >>> print(embeddings.shape)  # torch.Size([3, 512])
        """
        if not isinstance(audio_data, (list, tuple)):
            raise ValueError(f"audio_data must be a list or tuple, got {type(audio_data)}")
        
        if len(audio_data) == 0:
            raise ValueError("audio_data cannot be empty")
        
        # Keep model in training mode to allow gradients
        # Don't set to eval mode - we want gradients to flow
        
        # Extract embeddings with gradient tracking
        embeddings = []
        if is_train:
            self.clap_model.train()  # Ensure model is in train mode
        else:
            self.clap_model.eval()   # Eval mode if not training
        for y in audio_data:
            # Convert numpy to tensor if needed
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).float()
            
            # Ensure 1D tensor (CLAP expects 1D audio waveform)
            if y.ndim == 0:
                raise ValueError(f"Audio data cannot be 0-dimensional scalar")
            elif y.ndim > 1:
                # Flatten to 1D: (1, N) -> (N,) or (C, N) -> (C*N,)
                y = y.flatten()
            
            # Move to device and ensure requires_grad
            y = y.to(self._device)
            y.requires_grad_(True)

            # print(y)
            
            # Use use_tensor=True to get torch tensors with gradients
            emb = self.clap_model.get_audio_embedding_from_data(x=y.unsqueeze(0), use_tensor=True)
            # print(emb)
            # print(emb.shape)
            # input()
            embeddings.append(emb)
        embeddings = torch.vstack(embeddings)

        # 1. find max length
        # max_len = max(w.shape[1] for w in audio_data)

        # # 2. pad each waveform
        # batch_padded = np.array([np.pad(w, (0, max_len - w.shape[1])) for w in audio_data])

        # # 3. convert to tensor
        # batch_tensor = torch.tensor(batch_padded, dtype=torch.float32).to(self._device)  # [batch, max_len]

        # print(batch_tensor.shape)
        # input()

        # embeddings = self.clap_model.get_audio_embedding_from_data(x=batch_tensor, use_tensor=True)

        # print(embeddings.shape)
        # input()
        
        # Ensure embeddings are on the correct device
        embeddings = embeddings.to(self._device)
        
        return embeddings
    
    def train(self, mode: bool = True):
        """Override train to keep CLAP in eval mode (frozen backbone)."""
        # Keep parent class behavior for other modules
        super().train(mode)
        # But always keep CLAP model in eval mode
        self.clap_model.eval()
        return self
    
    def eval(self):
        """Set to evaluation mode."""
        return super().eval()


class CLAPEmbeddingBackbone(nn.Module):
    """
    A simpler CLAP backbone that works with pre-extracted embeddings.
    
    This is useful when you have already extracted CLAP embeddings
    and saved them as numpy files. The backbone simply passes through
    the embeddings (identity function) but provides the correct
    feature dimension information.
    
    Args:
        embedding_dim: Dimension of pre-extracted embeddings (default: 512)
    """
    
    def __init__(self, embedding_dim: int = 512, **kwargs):
        super(CLAPEmbeddingBackbone, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_dim = embedding_dim
        
    def forward(self, x):
        """
        Forward pass - identity function for pre-extracted embeddings.
        
        Args:
            x: Pre-extracted embeddings [B, embedding_dim]
            
        Returns:
            Same embeddings [B, embedding_dim]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return x


def load_clap_model(device='cuda', enable_fusion=False):
    """
    Helper function to load CLAP model from LAION.
    
    Args:
        device: Device to load model on
        enable_fusion: Whether to enable fusion
        
    Returns:
        CLAP model
    """
    laion_clap = _import_laion_clap()
    if not laion_clap:
        raise ImportError("laion-clap is required. Install with: pip install laion-clap")
    
    model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, device=device)
    model.load_ckpt()  # Load default pretrained checkpoint
    model.eval()
    return model


def load_finetuned_clap(checkpoint_path: str, device: str = 'cuda', enable_fusion: bool = False):
    """
    Load a fine-tuned CLAP model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on
        enable_fusion: Whether to enable fusion
        
    Returns:
        CLAP model with fine-tuned weights
    """
    laion_clap = _import_laion_clap()
    if not laion_clap:
        raise ImportError("laion-clap is required. Install with: pip install laion-clap")
    
    # Initialize base CLAP model
    model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, device=device)
    model.load_ckpt()  # Load base architecture
    
    # Load fine-tuned weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"Loaded fine-tuned CLAP from {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    
    return model


def extract_embeddings_from_audio_paths(model, audio_paths: list, device: str = 'cuda'):
    """
    Extract CLAP embeddings from a batch of audio file paths.
    
    This function reads audio files from the given paths and returns their
def extract_embeddings_from_audio_paths(model, audio_paths: list, device: str = 'cuda'):

    Extract CLAP embeddings from a batch of audio file paths.
    
    This function reads audio files from the given paths and returns their
    corresponding embeddings extracted by the CLAP model.
    
    Args:
        model: CLAP model (from load_clap_model or load_finetuned_clap)
        audio_paths: List of paths to audio files (e.g., ['/path/to/audio1.wav', '/path/to/audio2.wav'])
        device: Device to use for inference (default: 'cuda')
        
    Returns:
        torch.Tensor: Batch of embeddings with shape [batch_size, embedding_dim]
                      where embedding_dim is typically 512 for CLAP
    
    Example:
        >>> model = load_clap_model(device='cuda')
        >>> paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
        >>> embeddings = extract_embeddings_from_audio_paths(model, paths)
        >>> print(embeddings.shape)  # torch.Size([3, 512])
    """
    if not CLAP_AVAILABLE:
        raise ImportError("laion-clap is required. Install with: pip install laion-clap")
        raise ValueError("audio_paths cannot be empty")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Extract embeddings
    with torch.no_grad():
        embeddings = model.get_audio_embedding_from_filelist(
            x=audio_paths,
            use_tensor=True
        )
    
    # Ensure embeddings are on the correct device
    embeddings = embeddings.to(device)
    
    return embeddings


# if __name__ == "__main__":
#     # Test CLAPBackbone
#     # print("Testing CLAPBackbone...")
    


if __name__ == "__main__":
    # Test CLAPBackbone
    print("Testing CLAPBackbone...")
    
    laion_clap = _import_laion_clap()
    if laion_clap:
        # Test with pre-extracted embeddings (most common use case)
        backbone = CLAPEmbeddingBackbone(embedding_dim=512)
        
        # Simulate pre-extracted embeddings
        dummy_embeddings = torch.randn(5, 512)
        output = backbone(dummy_embeddings)
        
        print(f"Input shape: {dummy_embeddings.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Feature dimension: {backbone.feat_dim}")
        
        # Test with numpy input
        dummy_np = np.random.randn(5, 512).astype(np.float32)
        output_np = backbone(dummy_np)
        print(f"Numpy input output shape: {output_np.shape}")
        
        print("\nCLAPBackbone tests passed!")
        
        # Test extract_embeddings_from_audio_paths function
        print("\nTesting extract_embeddings_from_audio_paths...")
        print("Note: This test requires actual audio files to work.")
        print("Example usage:")
        print("  model = load_clap_model(device='cuda')")
        print("  paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']")
        print("  embeddings = extract_embeddings_from_audio_paths(model, paths)")
        print("  # Output shape: [3, 512]")
    else:
        print("CLAP not available, skipping full tests")
        
        # Test embedding backbone only
        backbone = CLAPEmbeddingBackbone(embedding_dim=512)
        dummy = torch.randn(5, 512)
        output = backbone(dummy)
        print(f"CLAPEmbeddingBackbone output shape: {output.shape}")
