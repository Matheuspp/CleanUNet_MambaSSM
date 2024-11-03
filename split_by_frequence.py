import torchaudio
import torch

def bandpass_filter_waveform(waveform, sample_rate, low_freq, high_freq):
    # Convert frequency cutoffs to normalized frequency (Nyquist frequency is 0.5)
    low_freq_normalized = low_freq / (sample_rate / 2)
    high_freq_normalized = high_freq / (sample_rate / 2)

    # Design a bandpass filter kernel
    kernel = torch.tensor([low_freq_normalized, high_freq_normalized], dtype=torch.float32)
    kernel = kernel.unsqueeze(0)  # Add batch dimension
    kernel = kernel.unsqueeze(0)  # Add input channel dimension

    # Pad the kernel to match the length of the waveform
    padding = (kernel.shape[-1] - 1) // 2
    padded_waveform = torch.nn.functional.pad(waveform, (padding, padding), mode='reflect')

    # Apply 1D convolution to perform bandpass filtering
    filtered_waveform = torch.nn.functional.conv1d(padded_waveform.unsqueeze(0), kernel)

    return filtered_waveform.squeeze(0)

def split_audio_by_frequency(audio_file):
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    print(waveform)

    # Define frequency ranges
    freq_ranges = [
        (0, 1000),    # 0 Hz to 1 kHz
        (1000, 10000),  # 1 kHz to 10 kHz
        (10000, sample_rate)  # 10 kHz to 48 kHz
    ]

    # List to store filtered waveforms
    filtered_waveforms = []

    # Apply bandpass filter to waveform for each frequency range
    for low_freq, high_freq in freq_ranges:
        filtered_waveform = bandpass_filter_waveform(waveform, sample_rate, low_freq, high_freq)
        filtered_waveforms.append(filtered_waveform)

    return filtered_waveforms


# Example usage:
audio_file = "./datasets/Valentini_3/testing_set/clean/fileid_289.wav"
filtered_waveforms = split_audio_by_frequency(audio_file)
for w in filtered_waveforms:
    print(print(w.shape))