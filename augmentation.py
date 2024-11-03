import random
import torch
import torchaudio
import torchaudio.functional as F

def band_mask_augmentation(waveform, sample_rate, f0, f1, apply_probability=0.05):
    """
    Apply band mask augmentation randomly to the waveform.

    Parameters:
    - waveform: The audio waveform tensor.
    - sample_rate: The sample rate of the audio waveform.
    - f0, f1: The lower and upper bounds of the frequency range to be attenuated.
    - apply_probability: Probability of applying the band mask augmentation. Defaults to 0.5.
    """
    # Randomly decide whether to apply the band mask augmentation
    if random.random() < apply_probability:
        # Calculate the center frequency and bandwidth for the band-reject filter
        center_freq = (f0 + f1) / 2
        bandwidth = f1 - f0
        
        # Apply the band-reject filter
        filtered_waveform = F.equalizer_biquad(waveform, sample_rate, center_freq, 0, bandwidth)
        return filtered_waveform
    else:
        # If not applying the augmentation, return the original waveform
        return waveform

def remix_augmentation(clean_batch, noise_batch, p=0.05):
    if random.random() < p:
        # Remix Augmentation
        perm = torch.randperm(noise_batch.size(0))
        remixed_noise_batch = noise_batch[perm]

        # Mix clean audio with remixed noise
        remixed_data = clean_batch + remixed_noise_batch  # Adjust mixing logic as needed
    else:
        remixed_data = noise_batch
        
    return remixed_data

def apply_aug(clean_batch, noisy_batch, sr=48000, p=0.05):
    #noisy_batch = remix_augmentation(clean_batch, noisy_batch)
    #r1 = random.randint(1000, 20000)
    #r2 = random.randint(1000, 20000)
    # BandMask Augmentation (applied to the noisy mix)
    grid = [(1000, 8000), (8000, 16000), (16000, 24000), (24000, 32000), (32000, 40000), (40000, 47000)]
    for i in range(noisy_batch.size(0)):
        #f0, f1 = random.choice(grid)
        f0, f1 = 1000, 8000 #select_frequency_range()  # Implement this based on your dataset/strategy
        #f0 = min(r1, r2)
        #f1 = max(r1, r2)
        noisy_batch[i] = band_mask_augmentation(noisy_batch[i], sr, f0, f1, p)
    
    return clean_batch, noisy_batch
