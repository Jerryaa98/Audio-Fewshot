# -*- coding: utf-8 -*-
"""
Audio Spectrogram Augmentation Functions

This module provides augmentation techniques for audio spectrograms including:
- Random Cutout
- Linear Type FilterAugment
- De-normalization and re-normalization utilities
"""

import torch
import numpy as np
import random


def denormalize_spectrogram(spectrogram, mean, std):
    """
    De-normalize a spectrogram using mean and std.
    
    Args:
        spectrogram (torch.Tensor): Normalized spectrogram
        mean (float or torch.Tensor): Mean used for normalization
        std (float or torch.Tensor): Std used for normalization
    
    Returns:
        torch.Tensor: De-normalized spectrogram
    """
    if isinstance(mean, (int, float)):
        mean = torch.tensor(mean, device=spectrogram.device, dtype=spectrogram.dtype)
    if isinstance(std, (int, float)):
        std = torch.tensor(std, device=spectrogram.device, dtype=spectrogram.dtype)
    
    return spectrogram * std + mean


def normalize_spectrogram(spectrogram, mean, std):
    """
    Normalize a spectrogram using mean and std.
    
    Args:
        spectrogram (torch.Tensor): Spectrogram to normalize
        mean (float or torch.Tensor): Mean for normalization
        std (float or torch.Tensor): Std for normalization
    
    Returns:
        torch.Tensor: Normalized spectrogram
    """
    if isinstance(mean, (int, float)):
        mean = torch.tensor(mean, device=spectrogram.device, dtype=spectrogram.dtype)
    if isinstance(std, (int, float)):
        std = torch.tensor(std, device=spectrogram.device, dtype=spectrogram.dtype)
    
    return (spectrogram - mean) / std


def random_cutout(spectrogram, num_cutouts=1, cutout_size_ratio=(0.1, 0.3), fill_value=0.0):
    """
    Apply random cutout augmentation to a spectrogram.
    
    Randomly masks rectangular regions in the spectrogram.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram of shape [B, C, H, W], [C, H, W] or [H, W]
        num_cutouts (int): Number of cutout regions to apply
        cutout_size_ratio (tuple): (min_ratio, max_ratio) for cutout size relative to spectrogram size
        fill_value (float): Value to fill in the cutout regions
    
    Returns:
        torch.Tensor: Augmented spectrogram
    """
    spec = spectrogram.clone()
    
    # Handle [B, C, H, W], [C, H, W] and [H, W] shapes
    if spec.dim() == 2:
        h, w = spec.shape
        mode = '2D'
    elif spec.dim() == 3:
        c, h, w = spec.shape
        mode = '3D'
    elif spec.dim() == 4:
        b, c, h, w = spec.shape
        mode = '4D'
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {spec.shape}")
    
    for _ in range(num_cutouts):
        # Random cutout height and width
        cutout_h = int(h * random.uniform(*cutout_size_ratio))
        cutout_w = int(w * random.uniform(*cutout_size_ratio))
        
        # Random position
        top = random.randint(0, max(0, h - cutout_h))
        left = random.randint(0, max(0, w - cutout_w))
        
        # Apply cutout
        if mode == '2D':
            spec[top:top+cutout_h, left:left+cutout_w] = fill_value
        elif mode == '3D':
            spec[:, top:top+cutout_h, left:left+cutout_w] = fill_value
        elif mode == '4D':
            spec[:, :, top:top+cutout_h, left:left+cutout_w] = fill_value
    
    return spec


def background_noise_suppression(spectrogram, noise_percentile=20, suppression_strength=0.5):
    """
    Suppress background noise in a spectrogram to reduce OOD effects from environmental noise.
    
    This augmentation helps align OOD samples (with different background noise) closer to 
    the support set by identifying and suppressing low-energy background components.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram of shape [B, C, H, W], [C, H, W] or [H, W]
        noise_percentile (float): Percentile (0-100) to identify noise floor threshold
        suppression_strength (float): Strength of suppression (0 to 1), 0=no suppression, 1=complete removal
    
    Returns:
        torch.Tensor: Noise-suppressed spectrogram
    """
    spec = spectrogram.clone()
    
    # Handle different tensor dimensions
    if spec.dim() == 2:
        h, w = spec.shape
        mode = '2D'
    elif spec.dim() == 3:
        c, h, w = spec.shape
        mode = '3D'
    elif spec.dim() == 4:
        b, c, h, w = spec.shape
        mode = '4D'
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {spec.shape}")
    
    # Estimate noise floor using percentile
    if mode == '2D':
        noise_threshold = torch.quantile(spec.abs(), noise_percentile / 100.0)
        # Create soft mask: values below threshold are suppressed
        mask = torch.sigmoid((spec.abs() - noise_threshold) / (noise_threshold * 0.1 + 1e-8))
        spec = spec * (1 - suppression_strength * (1 - mask))
    
    elif mode == '3D':
        # Process each channel
        for c_idx in range(c):
            noise_threshold = torch.quantile(spec[c_idx].abs(), noise_percentile / 100.0)
            mask = torch.sigmoid((spec[c_idx].abs() - noise_threshold) / (noise_threshold * 0.1 + 1e-8))
            spec[c_idx] = spec[c_idx] * (1 - suppression_strength * (1 - mask))
    
    elif mode == '4D':
        # Process each sample and channel
        for b_idx in range(b):
            for c_idx in range(c):
                noise_threshold = torch.quantile(spec[b_idx, c_idx].abs(), noise_percentile / 100.0)
                mask = torch.sigmoid((spec[b_idx, c_idx].abs() - noise_threshold) / (noise_threshold * 0.1 + 1e-8))
                spec[b_idx, c_idx] = spec[b_idx, c_idx] * (1 - suppression_strength * (1 - mask))
    
    return spec


def temporal_median_background_subtraction(spectrogram, percentile=10):
    """
    Remove background noise by subtracting a temporal median/percentile profile.
    
    This helps make embeddings invariant to static or slowly-varying background noise
    by removing the persistent background component, leaving only the foreground signal.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram of shape [B, C, H, W], [C, H, W] or [H, W]
        percentile (float): Percentile (0-100) to estimate background along time axis
    
    Returns:
        torch.Tensor: Background-subtracted spectrogram
    """
    spec = spectrogram.clone()
    
    # Handle different tensor dimensions
    if spec.dim() == 2:
        h, w = spec.shape
        mode = '2D'
    elif spec.dim() == 3:
        c, h, w = spec.shape
        mode = '3D'
    elif spec.dim() == 4:
        b, c, h, w = spec.shape
        mode = '4D'
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {spec.shape}")
    
    def process_2d(spec_2d):
        # Estimate background as percentile along time axis for each frequency bin
        background = torch.quantile(spec_2d, percentile / 100.0, dim=1, keepdim=True)
        
        # Subtract background and apply ReLU to keep only positive values
        spec_2d = torch.clamp(spec_2d - background, min=0.0)
        
        return spec_2d
    
    if mode == '2D':
        spec = process_2d(spec)
    elif mode == '3D':
        for c_idx in range(c):
            spec[c_idx] = process_2d(spec[c_idx])
    elif mode == '4D':
        for b_idx in range(b):
            for c_idx in range(c):
                spec[b_idx, c_idx] = process_2d(spec[b_idx, c_idx])
    
    return spec


def spectral_contrast_enhancement(spectrogram, contrast_factor=1.5, clip_percentile=95):
    """
    Enhance spectral contrast to emphasize foreground signal over background.
    
    This augmentation enhances the difference between high-energy (foreground) and 
    low-energy (background) regions, making CNN embeddings more focused on foreground features.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram of shape [B, C, H, W], [C, H, W] or [H, W]
        contrast_factor (float): Factor to enhance contrast (>1 increases contrast)
        clip_percentile (float): Percentile for clipping to avoid extreme values
    
    Returns:
        torch.Tensor: Contrast-enhanced spectrogram
    """
    spec = spectrogram.clone()
    
    # Handle different tensor dimensions
    if spec.dim() == 2:
        h, w = spec.shape
        mode = '2D'
    elif spec.dim() == 3:
        c, h, w = spec.shape
        mode = '3D'
    elif spec.dim() == 4:
        b, c, h, w = spec.shape
        mode = '4D'
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {spec.shape}")
    
    def process_2d(spec_2d):
        # Calculate mean value as reference point
        mean_val = spec_2d.mean()
        
        # Enhance contrast around mean
        spec_2d = mean_val + (spec_2d - mean_val) * contrast_factor
        
        # Clip extreme values using percentile
        if clip_percentile < 100:
            max_val = torch.quantile(spec_2d.abs(), clip_percentile / 100.0)
            spec_2d = torch.clamp(spec_2d, -max_val, max_val)
        
        return spec_2d
    
    if mode == '2D':
        spec = process_2d(spec)
    elif mode == '3D':
        for c_idx in range(c):
            spec[c_idx] = process_2d(spec[c_idx])
    elif mode == '4D':
        for b_idx in range(b):
            for c_idx in range(c):
                spec[b_idx, c_idx] = process_2d(spec[b_idx, c_idx])
    
    return spec


def foreground_energy_normalization(spectrogram, top_k_percent=20):
    """
    Normalize based on foreground energy to make embeddings invariant to overall energy differences.
    
    This identifies high-energy foreground regions and normalizes the entire spectrogram
    based on foreground statistics, reducing sensitivity to background level variations.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram of shape [B, C, H, W], [C, H, W] or [H, W]
        top_k_percent (float): Percentage of highest energy values considered as foreground
    
    Returns:
        torch.Tensor: Foreground-normalized spectrogram
    """
    spec = spectrogram.clone()
    
    # Handle different tensor dimensions
    if spec.dim() == 2:
        h, w = spec.shape
        mode = '2D'
    elif spec.dim() == 3:
        c, h, w = spec.shape
        mode = '3D'
    elif spec.dim() == 4:
        b, c, h, w = spec.shape
        mode = '4D'
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {spec.shape}")
    
    def process_2d(spec_2d):
        # Identify foreground using top-k energy
        energy = spec_2d.abs()
        threshold = torch.quantile(energy, 1.0 - top_k_percent / 100.0)
        foreground_mask = energy >= threshold
        
        if foreground_mask.sum() > 0:
            # Calculate foreground statistics
            foreground_values = spec_2d[foreground_mask]
            fg_mean = foreground_values.mean()
            fg_std = foreground_values.std() + 1e-8
            
            # Normalize entire spectrogram using foreground statistics
            spec_2d = (spec_2d - fg_mean) / fg_std
        
        return spec_2d
    
    if mode == '2D':
        spec = process_2d(spec)
    elif mode == '3D':
        for c_idx in range(c):
            spec[c_idx] = process_2d(spec[c_idx])
    elif mode == '4D':
        for b_idx in range(b):
            for c_idx in range(c):
                spec[b_idx, c_idx] = process_2d(spec[b_idx, c_idx])
    
    return spec


def wiener_like_filtering(spectrogram, noise_floor_percentile=15, gain_factor=2.0):
    """
    Apply Wiener-like filtering to suppress background while preserving foreground.
    
    Estimates SNR for each time-frequency bin and applies gain based on estimated
    signal-to-noise ratio, making embeddings focus on foreground signal.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram of shape [B, C, H, W], [C, H, W] or [H, W]
        noise_floor_percentile (float): Percentile to estimate noise floor
        gain_factor (float): Amplification factor for high-SNR regions
    
    Returns:
        torch.Tensor: Filtered spectrogram
    """
    spec = spectrogram.clone()
    
    # Handle different tensor dimensions
    if spec.dim() == 2:
        h, w = spec.shape
        mode = '2D'
    elif spec.dim() == 3:
        c, h, w = spec.shape
        mode = '3D'
    elif spec.dim() == 4:
        b, c, h, w = spec.shape
        mode = '4D'
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {spec.shape}")
    
    def process_2d(spec_2d):
        # Estimate noise floor
        noise_estimate = torch.quantile(spec_2d.abs(), noise_floor_percentile / 100.0)
        
        # Calculate SNR estimate for each bin
        signal_power = spec_2d.abs()
        snr = signal_power / (noise_estimate + 1e-8)
        
        # Wiener-like gain: suppress low SNR, amplify high SNR
        gain = snr / (snr + 1.0)  # Soft gating function
        gain = gain * gain_factor  # Optional amplification
        
        # Apply gain
        spec_2d = spec_2d * gain
        
        return spec_2d
    
    if mode == '2D':
        spec = process_2d(spec)
    elif mode == '3D':
        for c_idx in range(c):
            spec[c_idx] = process_2d(spec[c_idx])
    elif mode == '4D':
        for b_idx in range(b):
            for c_idx in range(c):
                spec[b_idx, c_idx] = process_2d(spec[b_idx, c_idx])
    
    return spec


def adaptive_noise_profile_matching(spectrogram, target_noise_level=None, smoothing_window=5):
    """
    Adapt the noise profile of a spectrogram to match a target noise level.
    
    This helps OOD samples with different background noise characteristics align with 
    the support set's noise profile in a few-shot setting.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram of shape [B, C, H, W], [C, H, W] or [H, W]
        target_noise_level (float): Target noise level. If None, uses a moderate default
        smoothing_window (int): Window size for temporal smoothing of noise estimate
    
    Returns:
        torch.Tensor: Noise-profile-adapted spectrogram
    """
    spec = spectrogram.clone()
    
    # Handle different tensor dimensions
    if spec.dim() == 2:
        h, w = spec.shape
        mode = '2D'
    elif spec.dim() == 3:
        c, h, w = spec.shape
        mode = '3D'
    elif spec.dim() == 4:
        b, c, h, w = spec.shape
        mode = '4D'
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {spec.shape}")
    
    # Default target noise level if not specified
    if target_noise_level is None:
        target_noise_level = 0.1
    
    def process_2d(spec_2d):
        # Estimate noise floor per time frame (minimum along frequency axis)
        noise_estimate = spec_2d.abs().min(dim=0, keepdim=True)[0]
        
        # Smooth the noise estimate temporally
        if smoothing_window > 1 and w > smoothing_window:
            kernel = torch.ones(1, 1, smoothing_window, device=spec_2d.device) / smoothing_window
            noise_estimate = noise_estimate.unsqueeze(0).unsqueeze(0)
            # Pad to maintain size
            pad_size = smoothing_window // 2
            noise_estimate = torch.nn.functional.pad(noise_estimate, (pad_size, pad_size), mode='reflect')
            noise_estimate = torch.nn.functional.conv2d(noise_estimate, kernel)
            noise_estimate = noise_estimate.squeeze(0).squeeze(0)
        
        # Calculate scaling factor to match target noise level
        current_noise_level = noise_estimate.mean()
        if current_noise_level > 1e-8:
            noise_scale = target_noise_level / (current_noise_level + 1e-8)
            noise_scale = torch.clamp(noise_scale, 0.5, 2.0)  # Limit extreme adjustments
        else:
            noise_scale = 1.0
        
        # Apply noise scaling with soft transition
        # Identify signal vs noise using energy threshold
        energy_threshold = torch.quantile(spec_2d.abs(), 0.3)
        signal_mask = torch.sigmoid((spec_2d.abs() - energy_threshold) / (energy_threshold * 0.1 + 1e-8))
        
        # Apply different scaling to noise vs signal regions
        spec_2d = spec_2d * (signal_mask + (1 - signal_mask) * noise_scale)
        
        return spec_2d
    
    if mode == '2D':
        spec = process_2d(spec)
    elif mode == '3D':
        for c_idx in range(c):
            spec[c_idx] = process_2d(spec[c_idx])
    elif mode == '4D':
        for b_idx in range(b):
            for c_idx in range(c):
                spec[b_idx, c_idx] = process_2d(spec[b_idx, c_idx])
    
    return spec


def apply_linear_filteraugment(spectrogram, num_points=4, filter_strength=0.5):
    """
    Applies "Linear Type" FilterAugment to a spectrogram.
    
    This method creates a continuous filter with random peaks and dips 
    by linearly interpolating between a few random points.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram of shape [B, C, H, W], [C, H, W] or [H, W]
        num_points (int): Number of random points to use for interpolation
        filter_strength (float): Strength of the filter (0 to 1), controls the magnitude of peaks/dips
    
    Returns:
        torch.Tensor: Filtered spectrogram
    """
    spec = spectrogram.clone()
    
    # Handle [B, C, H, W], [C, H, W] and [H, W] shapes
    if spec.dim() == 2:
        h, w = spec.shape
        mode = '2D'
    elif spec.dim() == 3:
        c, h, w = spec.shape
        mode = '3D'
    elif spec.dim() == 4:
        b, c, h, w = spec.shape
        mode = '4D'
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {spec.shape}")
    
    # Create random points along frequency axis (height)
    freq_points = sorted(random.sample(range(h), min(num_points, h)))
    
    # Generate random filter values at these points
    # Values range from (1 - filter_strength) to (1 + filter_strength)
    filter_values = [1.0 + random.uniform(-filter_strength, filter_strength) for _ in freq_points]
    
    # Create full filter by linear interpolation
    filter_curve = np.interp(
        np.arange(h),
        freq_points,
        filter_values
    )
    
    # Convert to torch tensor
    filter_curve = torch.tensor(filter_curve, device=spec.device, dtype=spec.dtype)
    
    # Apply filter along frequency axis
    if mode == '2D':
        # Reshape filter for broadcasting: [H, 1]
        filter_curve = filter_curve.view(h, 1)
        spec = spec * filter_curve
    elif mode == '3D':
        # Reshape filter for broadcasting: [1, H, 1]
        filter_curve = filter_curve.view(1, h, 1)
        spec = spec * filter_curve
    elif mode == '4D':
        # Reshape filter for broadcasting: [1, 1, H, 1]
        filter_curve = filter_curve.view(1, 1, h, 1)
        spec = spec * filter_curve
    
    return spec


def augment_spectrogram(spectrogram, mean, std, augmentation_type='random', **kwargs):
    """
    Main augmentation function that:
    1) De-normalizes the spectrogram
    2) Applies augmentation (random cutout, Linear Type FilterAugment, or noise adaptation)
    3) Re-normalizes the spectrogram
    
    Args:
        spectrogram (torch.Tensor): Normalized input spectrogram
        mean (float or torch.Tensor): Mean used for original normalization
        std (float or torch.Tensor): Std used for original normalization
        augmentation_type (str): 'cutout', 'linear_filter', 'noise_suppression', 
                                 'noise_matching', or 'random' (randomly choose)
        **kwargs: Additional arguments for specific augmentation methods
    
    Returns:
        torch.Tensor: Augmented and re-normalized spectrogram
    """
    # Step 1: De-normalize
    spec = denormalize_spectrogram(spectrogram, mean, std)
    
    # Step 2: Apply augmentation
    if augmentation_type == 'random':
        augmentation_type = random.choice([
            'cutout', 'linear_filter', 'noise_suppression', 'noise_matching',
            'background_subtraction', 'contrast_enhancement', 'foreground_norm', 'wiener_filter'
        ])
    
    if augmentation_type == 'cutout':
        num_cutouts = kwargs.get('num_cutouts', random.randint(1, 3))
        cutout_size_ratio = kwargs.get('cutout_size_ratio', (0.1, 0.3))
        fill_value = kwargs.get('fill_value', 0.0)
        spec = random_cutout(spec, num_cutouts, cutout_size_ratio, fill_value)
    
    elif augmentation_type == 'linear_filter':
        num_points = kwargs.get('num_points', random.randint(3, 6))
        filter_strength = kwargs.get('filter_strength', random.uniform(0.3, 0.7))
        spec = apply_linear_filteraugment(spec, num_points, filter_strength)
    
    elif augmentation_type == 'noise_suppression':
        noise_percentile = kwargs.get('noise_percentile', random.uniform(15, 25))
        suppression_strength = kwargs.get('suppression_strength', random.uniform(0.4, 0.7))
        spec = background_noise_suppression(spec, noise_percentile, suppression_strength)
    
    elif augmentation_type == 'noise_matching':
        target_noise_level = kwargs.get('target_noise_level', None)
        smoothing_window = kwargs.get('smoothing_window', random.choice([3, 5, 7]))
        spec = adaptive_noise_profile_matching(spec, target_noise_level, smoothing_window)
    
    elif augmentation_type == 'background_subtraction':
        percentile = kwargs.get('percentile', random.uniform(5, 15))
        spec = temporal_median_background_subtraction(spec, percentile)
    
    elif augmentation_type == 'contrast_enhancement':
        contrast_factor = kwargs.get('contrast_factor', random.uniform(1.3, 2.0))
        clip_percentile = kwargs.get('clip_percentile', random.uniform(90, 98))
        spec = spectral_contrast_enhancement(spec, contrast_factor, clip_percentile)
    
    elif augmentation_type == 'foreground_norm':
        top_k_percent = kwargs.get('top_k_percent', random.uniform(15, 25))
        spec = foreground_energy_normalization(spec, top_k_percent)
    
    elif augmentation_type == 'wiener_filter':
        noise_floor_percentile = kwargs.get('noise_floor_percentile', random.uniform(10, 20))
        gain_factor = kwargs.get('gain_factor', random.uniform(1.5, 2.5))
        spec = wiener_like_filtering(spec, noise_floor_percentile, gain_factor)
    
    else:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type}")
    
    # Step 3: Re-normalize
    spec = normalize_spectrogram(spec, mean, std)
    
    return spec


def batch_augment_spectrogram(spectrograms, mean, std, num_augmentations=10, **kwargs):
    """
    Apply augmentation to a batch of spectrograms, generating multiple augmented versions.
    
    Args:
        spectrograms (torch.Tensor): Batch of normalized spectrograms [B, C, H, W] or single [C, H, W]
        mean (float or torch.Tensor): Mean used for normalization
        std (float or torch.Tensor): Std used for normalization
        num_augmentations (int): Number of augmented versions to generate per input
        **kwargs: Additional arguments for augmentation_spectrogram
    
    Returns:
        torch.Tensor: Augmented spectrograms [B * num_augmentations, C, H, W]
    """
    # Handle different input dimensions
    if spectrograms.dim() == 3:
        spectrograms = spectrograms.unsqueeze(0)
    elif spectrograms.dim() == 4:
        pass  # Already in correct format [B, C, H, W]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {spectrograms.shape}")
    
    batch_size = spectrograms.shape[0]
    augmented_list = []
    
    for i in range(batch_size):
        spec = spectrograms[i:i+1]  # Keep 4D shape [1, C, H, W]
        for _ in range(num_augmentations):
            aug_spec = augment_spectrogram(spec, mean, std, **kwargs)
            # Remove batch dimension if it was added
            if aug_spec.dim() == 4:
                aug_spec = aug_spec.squeeze(0)
            augmented_list.append(aug_spec)
    
    # Stack all augmented spectrograms
    result = torch.stack(augmented_list, dim=0)
    
    return result


# Convenience functions for step-based and linear filter augmentation
def apply_step_filteraugment(spectrogram, mean=0.0, std=1.0, num_augmentations=10):
    """
    Convenience function applying random augmentation type.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram
        mean (float): Normalization mean
        std (float): Normalization std
        num_augmentations (int): Number of augmented versions to generate
    
    Returns:
        torch.Tensor: Batch of augmented spectrograms
    """
    return batch_augment_spectrogram(
        spectrogram, 
        mean, 
        std, 
        num_augmentations=num_augmentations,
        augmentation_type='random'
    )


def apply_linear_filteraugment_wrapper(spectrogram, mean=0.0, std=1.0, num_augmentations=10):
    """
    Convenience function applying only linear filter augmentation.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram
        mean (float): Normalization mean
        std (float): Normalization std
        num_augmentations (int): Number of augmented versions to generate
    
    Returns:
        torch.Tensor: Batch of augmented spectrograms
    """
    return batch_augment_spectrogram(
        spectrogram, 
        mean, 
        std, 
        num_augmentations=num_augmentations,
        augmentation_type='linear_filter'
    )
