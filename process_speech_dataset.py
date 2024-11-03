# Copyright (c) 2023 Matheus Vieira da Silva

import argparse
import pathlib
import multiprocessing
from shutil import move
import soundfile as sf
import librosa
import numpy as np

# ### change here
desired_length=3
output_dir='training/clean'
input_dir ='training/clean_pre'
# ### =======================

def rename(folder):
    """Rename files in the given folder to 'fileid_<index>.wav'."""
    files = pathlib.Path(folder).glob('*')
    for i, audio in enumerate(sorted(files)):
        move(audio, f'{folder}/fileid_{i}.wav')
        #print(audio)

def pad(acum, name, sr=48000):
    """Trim and save the remaining audio data."""
    end = sr*desired_length
    for i in range(0, acum.shape[0], end):
        if i+end > acum.shape[0]:
            remain = end - acum[i:-1].shape[0]
            extra = np.array(remain*[acum[-1]])
            y = np.concatenate([acum[i:-1], extra])
            print(f'Remaining file new shape: {y.shape}')
        else:
            y = acum[i:i+end]
        sf.write(file=f'{output_dir}/{name}.wav', data=y, samplerate=sr)

def trim(files):
    """Trim and save audio clips from a list of file paths."""
    acum, sr = librosa.load(str(files[0]), sr=48000)
    idx = 0
    end = sr*desired_length
    for audio in files[1:]:
        name = audio.split('/')[-1]
        #print('>>>',name)
        # Load the audio file
        y, sr = librosa.load(str(audio), sr=48000)
        acum = np.concatenate([acum, y])
        if acum.shape[0] > end:
            y = acum[0:end]
            sf.write(file=f'{output_dir}/{name}', data=y, samplerate=sr)
            acum = acum[end:]
    print(f'>>> Remaning audio sample: {acum.shape} -> {name}')
    return acum, name


if __name__ == '__main__':
    file = sorted(pathlib.Path(input_dir).glob('*'))
    files = [str(path) for path in file]
    remain, name = trim(files)
    pad(remain, name)
    rename(output_dir)
    print('Done!')
