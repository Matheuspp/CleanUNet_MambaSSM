# CleanUNet_MambaSSM
# SpeechDenoiser

In this research, we present an improved version of the CleanUNet model from the paper [Speech Denoising in the Waveform Domain with Self-Attention](https://arxiv.org/abs/2202.07790). This model performs advanced noise reduction while preserving causal relationships in the audio signal at the raw waveform level. The original source code is available here: [PyTorch Implementation of CleanUNet](https://github.com/NVIDIA/CleanUNet).

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained Models](#pre-trained-models)
- [References](#references)

## Installation

To set up the environment for this project, you need to install the necessary dependencies and ensure you have **CUDA version 11.7 or higher** installed.

**Step 1: Create and activate a new conda environment (recommended)**

```bash
conda create -n speechdenoiser python=3.8
conda activate speechdenoiser
```

**Step 2: Install Python dependencies**

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

**Note:** It's recommended to install all dependencies inside a conda environment to avoid conflicts.

## Datasets

The main dataset used in our experiments is the **Valentini dataset**, which is a valuable resource for training speech enhancement algorithms and text-to-speech (TTS) models. You can download the dataset from [this link](https://datashare.ed.ac.uk/handle/10283/2791).

**Dataset Preparation:**

1. **Download the Dataset:**

    - Navigate to the [Valentini dataset page](https://datashare.ed.ac.uk/handle/10283/2791) and download the dataset files.

2. **Extract the Dataset:**

    - Extract the contents of the dataset into your designated `datasets` directory.

3. **Process the Audio Files:**

    - Use the provided script `process_speech_dataset.py` to crop the audio clips to your desired length. This script helps in standardizing the audio files for training.

## Configuration

Experiment configurations are controlled via JSON files located in the `configs` directory. These files contain various setup parameters allowing you to customize and fine-tune your experiments.

**Example Configuration (`configs/example_config.json`):**

```json
{
    "network_config": {
        "channels_input": 1,
        "channels_output": 1,
        "channels_H": 64,
        "max_H": 768,
        "encoder_n_layers": 4,
        "kernel_size": 3,
        "stride": 2
    },
    "train_config": {
        "exp_path": "experiments/exp1",
        "log": {
            "directory": "./logs",
            "ckpt_iter": "max",
            "iters_per_ckpt": 10000,
            "iters_per_valid": 500
        },
        "optimization": {
            "n_iters": 1000000,
            "learning_rate": 0.00025,
            "batch_size_per_gpu": 32
        },
        "loss_config": {
            "ell_p": 1,
            "ell_p_lambda": 1,
            "stft_lambda": 1,
            "stft_config": {
                "sc_lambda": 0.5,
                "mag_lambda": 0.5,
                "band": "full",
                "hop_sizes": [50, 120, 240],
                "win_lengths": [240, 600, 1200],
                "fft_sizes": [512, 1024, 2048]
            }
        }
    },
    "trainset_config": {
        "root": "./datasets/processed",
        "crop_length_sec": 2,
        "sample_rate": 48000
    }
}
```

**Note:** Adjust the parameters according to your requirements. Ensure that the paths in `trainset_config` point to the correct directories where your processed dataset is located.

## Training

With the environment set up, dependencies installed, and the dataset preprocessed, you can start training the model.

**Training Command:**

```bash
python3 train.py -c configs/your_config.json
```

- Replace `your_config.json` with the path to your configuration JSON file.
- The training script will read the configurations and start the training process accordingly.


## Evaluation

Before evaluating the model's performance, generate enhanced audio samples using the trained model.

**Generating Enhanced Samples:**

```bash
python3 denoise.py -c configs/your_config.json 
```

- The `-c` flag specifies your configuration JSON file.

A directory will be created inside the predefined folder specified in the JSON configuration file. Once the enhanced samples are generated, proceed to evaluate the model.

**Evaluation Command:**

```bash
python3 python_eval.py -e "path_to_enhanced_samples" -t "path_to_clean_test_samples"
```

- Replace `"path_to_enhanced_samples"` with the directory path of the enhanced samples generated.
- Replace `"path_to_clean_test_samples"` with the directory path of the clean test samples.

The evaluation script computes metrics to assess the model's performance on the test data.



**Download Pre-trained Models:**

You can download the pre-trained models and enhanced audio samples from [this Google Drive link](https://drive.google.com/drive/folders/10uQAdmaRtwYJNMIXe0_WV3GHGIcRbadu?usp=sharing).


## References

- **Research Paper:** [Speech Denoising in the Waveform Domain with Self-Attention](https://arxiv.org/abs/2202.07790)
- **Original Codebase:** [PyTorch Implementation of CleanUNet](https://github.com/NVIDIA/CleanUNet)
- **Valentini Dataset:** [Valentini-Botinhao Dataset](https://datashare.ed.ac.uk/handle/10283/2791)

