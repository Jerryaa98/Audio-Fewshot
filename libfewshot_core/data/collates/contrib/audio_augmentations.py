import numpy as np
import random

def apply_step_filteraugment(spectrogram, num_bands=1, min_freq_bands=10, max_freq_bands=20, min_db_change=-6, max_db_change=6):
    """
    Applies "Step Type" FilterAugment to a spectrogram.

    This method, described as the prototype, applies a uniform increase or
    decrease (a "step") to one or more random frequency bands.[2]

    Args:
        spectrogram (np.array): Input log-mel spectrogram (shape: [time_steps, n_mels]).
        num_bands (int): The number of frequency bands to create.
        min_freq_bands (int): The minimum width (in mels) of a frequency band.
        max_freq_bands (int): The maximum width (in mels) of a frequency band.
        min_db_change (float): The minimum decibel change to apply (can be negative).
        max_db_change (float): The maximum decibel change to apply.

    Returns:
        np.array: The augmented spectrogram.
    """
    # Copy to avoid modifying the original array
    aug_spec = spectrogram.copy()
    num_mels = aug_spec.shape[1]

    for _ in range(num_bands):
        # Determine the width of the frequency band
        band_width = random.randint(min_freq_bands, max_freq_bands)
        if band_width > num_mels:
            band_width = num_mels
        
        # Determine the start frequency bin for the band
        # Ensure the band fits within the spectrogram
        start_mel = random.randint(0, num_mels - band_width)
        end_mel = start_mel + band_width

        # Determine the random gain/loss in dB
        db_change = random.uniform(min_db_change, max_db_change)

        # Apply the uniform change to the selected frequency band
        # We add the change, as log-mel spectrograms are in a logarithmic (dB) scale
        aug_spec[:, start_mel:end_mel] = aug_spec[:, start_mel:end_mel] + db_change

    return aug_spec

def apply_linear_filteraugment(spectrogram, num_points=4, min_db_change=-8, max_db_change=8):
    """
    Applies "Linear Type" FilterAugment to a spectrogram.

    This method creates a continuous filter with random peaks and dips 
    by linearly interpolating between a few random points.[2]

    Args:
        spectrogram (np.array): Input log-mel spectrogram (shape: [time_steps, n_mels]).
        num_points (int): Number of random points to create along the frequency axis.
        min_db_change (float): The minimum decibel change for a random point.
        max_db_change (float): The maximum decibel change for a random point.

    Returns:
        np.array: The augmented spectrogram.
    """
    # Copy to avoid modifying the original array
    aug_spec = spectrogram.copy()
    time_steps, num_mels = aug_spec.shape

    if num_points < 2:
        num_points = 2

    # 1. Create random points for the filter
    # x-coordinates (frequency bins) for the points, including 0 and the max mel bin
    # Handle edge cases where num_points is small or > available bins
    if num_points <= 2 or num_mels <= 2:
        x_coords = np.array([0, num_mels - 1])
    else:
        interior_needed = num_points - 2
        max_interior = max(0, num_mels - 2)
        pick = min(interior_needed, max_interior)
        if pick > 0:
            interior = np.random.choice(range(1, num_mels - 1), pick, replace=False)
            x_coords = np.sort(np.concatenate(([0], interior, [num_mels - 1])))
        else:
            x_coords = np.array([0, num_mels - 1])

    # y-coordinates (dB change) for the points: one value per x anchor
    y_coords = np.random.uniform(min_db_change, max_db_change, size=x_coords.shape[0])
    # Ensure the filter starts and ends near 0 dB change (small endpoint values)
    y_coords[0] = random.uniform(min_db_change / 4, max_db_change / 4)
    y_coords[-1] = random.uniform(min_db_change / 4, max_db_change / 4)

    # 2. Linearly interpolate between the points to create the full filter
    # 'xp' are the x-coordinates, 'fp' are the y-coordinates
    # This creates a filter shape across all 'num_mels' frequency bins
    full_filter = np.interp(np.arange(num_mels), xp=x_coords, fp=y_coords)

    # 3. Apply the filter to the spectrogram
    # We need the filter to be shape (1, num_mels) so it can be broadcast-added
    # to the spectrogram of shape (time_steps, num_mels)
    full_filter = full_filter.reshape(1, num_mels)
    
    aug_spec = aug_spec + full_filter

    return aug_spec

# # --- Example Usage ---

# # 1. Create a dummy log-mel spectrogram
# # (e.g., 100 time steps, 128 mel frequency bins)
# dummy_spectrogram = np.random.rand(100, 128) * -20.0 + 5.0

# # 2. Apply "Step Type" FilterAugment
# # This will add 2 random bands of gain/loss
# augmented_spec_step = apply_step_filteraugment(
#     dummy_spectrogram, 
#     num_bands=2, 
#     min_freq_bands=15, 
#     max_freq_bands=30, 
#     min_db_change=-10, 
#     max_db_change=10
# )

# # 3. Apply "Linear Type" FilterAugment
# # This will create a smooth filter with 5 anchor points
# augmented_spec_linear = apply_linear_filteraugment(
#     dummy_spectrogram, 
#     num_points=5, 
#     min_db_change=-9, 
#     max_db_change=9
# )

# print(f"Original spectrogram shape: {dummy_spectrogram.shape}")
# print(f"Augmented (Step) shape: {augmented_spec_step.shape}")
# print(f"Augmented (Linear) shape: {augmented_spec_linear.shape}")