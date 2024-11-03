# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import load_CleanNoisyPairDataset
from stft_loss import MultiResolutionSTFTLoss
from util import find_max_epoch, print_size
from util import LinearWarmupCosineDecay, loss_fn

from network import CleanUNet
from augmentation import apply_aug

def train(num_gpus, rank, group_name, 
          exp_path, log, optimization, loss_config):
    if rank == 0:
        print('exp_path:', exp_path)
    
    # Create tensorboard logger
    log_directory = os.path.join(log["directory"], exp_path)
    if rank == 0:
        tb = SummaryWriter(os.path.join(log_directory, 'tensorboard'))

    # Prepare checkpoint directory
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0 and not os.path.isdir(ckpt_directory):
        os.makedirs(ckpt_directory)
        os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # Load training data
    trainloader = load_CleanNoisyPairDataset(**trainset_config, 
                            subset='training',
                            batch_size=optimization["batch_size_per_gpu"], 
                            num_gpus=num_gpus)
    print('Data loaded')
    
    # Initialize model and optimizer
    net = CleanUNet(**network_config).cuda()
    print_size(net, keyword="tsfm")
    optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"])

    # Load checkpoint if available
    time0 = time.time()
    ckpt_iter = find_max_epoch(ckpt_directory) if log["ckpt_iter"] == 'max' else log["ckpt_iter"]
    if ckpt_iter >= 0:
        try:
            model_path = os.path.join(ckpt_directory, f'{ckpt_iter}.pkl')
            checkpoint = torch.load(model_path, map_location='cpu')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            time0 -= checkpoint['training_time_seconds']
            print(f'Model at iteration {ckpt_iter} has been trained for {checkpoint["training_time_seconds"]} seconds')
            print('Checkpoint model loaded successfully')
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    n_iter = ckpt_iter + 1

    # Define learning rate scheduler and STFT loss
    scheduler = LinearWarmupCosineDecay(
                    optimizer,
                    lr_max=optimization["learning_rate"],
                    n_iter=optimization["n_iters"],
                    iteration=n_iter,
                    divider=19,
                    warmup_proportion=0.05,
                    phase=('linear', 'cosine'),
                )

    mrstftloss = MultiResolutionSTFTLoss(**loss_config["stft_config"]).cuda() if loss_config["stft_lambda"] > 0 else None
    min_loss = np.inf
    p = 0.05  # Probability for augmentation

    while n_iter < optimization["n_iters"] + 1:
        for clean_audio, noisy_audio, _ in trainloader: 
            clean_audio, noisy_audio = apply_aug(clean_audio, noisy_audio, p)
            clean_audio = clean_audio.cuda()
            noisy_audio = noisy_audio.cuda()

            optimizer.zero_grad()
            X = (clean_audio, noisy_audio)
            loss, loss_dic = loss_fn(net, X, **loss_config, mrstftloss=mrstftloss)

            reduced_loss = loss.item()
            with open(f'{ckpt_directory}/losses.txt', 'a') as fp_loss:
                fp_loss.write(f"{reduced_loss}\n")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            scheduler.step()
            optimizer.step()

            if n_iter % log["iters_per_valid"] == 0:
                print(f"iteration: {n_iter} \treduced loss: {reduced_loss:.7f} \tloss: {loss.item():.7f}", flush=True)
                if rank == 0:
                    tb.add_scalar("Train/Train-Loss", loss.item(), n_iter)
                    tb.add_scalar("Train/Train-Reduced-Loss", reduced_loss, n_iter)
                    tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                    tb.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter)

            if n_iter > 5000 and ((loss < min_loss) or (n_iter % 100000 == 0)):
                min_loss = loss
                checkpoint_name = f'{n_iter}.pkl'
                torch.save({'iter': n_iter,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time() - time0)}, 
                            os.path.join(ckpt_directory, checkpoint_name))
                print(f'Model at iteration {n_iter} is saved. Loss: {loss}')

            n_iter += 1

    if rank == 0:
        tb.close()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/2.json', 
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='Rank of process for distributed training')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='Name of group for distributed training')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        data = f.read()
    config = json.loads(data)
    train_config    = config["train_config"]
    dist_config     = config["dist_config"]
    network_config  = config["network_config"]
    trainset_config = config["trainset_config"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and args.group_name == '':
        print("WARNING: Multiple GPUs detected but no distributed group set")
        print("Only running 1 GPU. Use distributed.py for multiple GPUs")
        num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, **train_config)
