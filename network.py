import torch
import torch.nn as nn
import torch.nn.functional as F

from util import weight_scaling_init
from mamba_ssm import Mamba

def padding(x, D, K, S):
    """Padding zeros to x so that denoised audio has the same length."""
    L = x.shape[-1]
    for _ in range(D):
        if L < K:
            L = 1
        else:
            L = 1 + ((L - K + S - 1) // S)

    for _ in range(D):
        L = (L - 1) * S + K

    L = int(L)
    x = F.pad(x, (0, L - x.shape[-1]))
    return x

class CleanUNet(nn.Module):
    """CleanUNet architecture."""

    def __init__(self, channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=4, kernel_size=4, stride=2,
                 tsfm_d_model=512):
        """
        Parameters:
        channels_input (int):   Input channels.
        channels_output (int):  Output channels.
        channels_H (int):       Middle channels H that controls capacity.
        max_H (int):            Maximum H.
        encoder_n_layers (int): Number of encoder/decoder layers D.
        kernel_size (int):      Kernel size K.
        stride (int):           Stride S.
        tsfm_d_model (int):     d_model of self-attention.
        """
        super(CleanUNet, self).__init__()

        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.tsfm_d_model = tsfm_d_model

        # Encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        channels_H = self.channels_H
        channels_input = channels_input
        channels_output = channels_H

        for i in range(encoder_n_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv1d(channels_input, channels_H, kernel_size, stride),
                nn.BatchNorm1d(channels_H),
                nn.ReLU(inplace=False),
                nn.Conv1d(channels_H, channels_H * 2, 1),
                nn.GLU(dim=1)
            ))
            channels_input = channels_H

            if i == 0:
                # No ReLU at the end
                self.decoder.append(nn.Sequential(
                    nn.Conv1d(channels_H, channels_H * 2, 1),
                    nn.GLU(dim=1),
                    nn.ConvTranspose1d(channels_H, 1, kernel_size, stride),
                ))
            else:
                self.decoder.insert(0, nn.Sequential(
                    nn.Conv1d(channels_H, channels_H * 2, 1),
                    nn.GLU(dim=1),
                    nn.ConvTranspose1d(channels_H, channels_output, kernel_size, stride),
                    nn.BatchNorm1d(channels_output),
                    nn.ReLU(inplace=False),
                ))
            channels_output = channels_H

            # Double H but keep below max_H
            channels_H = min(channels_H * 2, self.max_H)

        # Mamba encoder
        self.tsfm_conv1 = nn.Conv1d(channels_output, tsfm_d_model, kernel_size=1)
        self.mamba_encoder = Mamba(
            d_model=tsfm_d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.tsfm_conv2 = nn.Conv1d(tsfm_d_model, channels_output, kernel_size=1)

        # Weight scaling initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_scaling_init(layer)

    def forward(self, noisy_audio):
        # Ensure input is (B, C, L)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        B, C, L = noisy_audio.shape
        assert C == 1

        # Normalization and padding
        std = noisy_audio.std(dim=2, keepdim=True) + 1e-3
        noisy_audio /= std
        x = padding(noisy_audio, self.encoder_n_layers, self.kernel_size, self.stride)

        # Encoder
        skip_connections = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            skip_connections.append(x)
        skip_connections = skip_connections[::-1]

        # Mamba encoder
        x = self.tsfm_conv1(x)
        x = x.permute(0, 2, 1)
        x = self.mamba_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.tsfm_conv2(x)

        # Decoder
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = skip_connections[i]
            x = x + skip_i[:, :, :x.shape[-1]]
            x = upsampling_block(x)

        x = x[:, :, :L] * std
        return x


if __name__ == '__main__':
    import json
    import argparse 
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/2.json', 
                        help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    network_config = config["network_config"]

    model = CleanUNet(**network_config).cuda()
    #torch.set_grad_enabled(True)
    model.train()
    #print(model)

    from util import print_size
    print_size(model, keyword="tsfm")
    
    input_data = torch.ones([1,1,48000]).cuda()
    # ### dilated attention
    device = torch.device("cuda")
    dtype = torch.float16
    embed_dim = 512
    #input_data = torch.randn(4, 8192, 512, device=device, dtype=dtype)
    #y = torch.randn(4, 8192, 512, device=device, dtype=dtype)
    output = model(input_data)
    print(output.shape)

    y = torch.rand([1,1,48000]).cuda()
    loss = torch.nn.MSELoss()(y, output)
    #loss.requires_grad = True
    #loss.retain_grad()
    # Backward pass
    loss.backward()
    print(loss.item())
    
