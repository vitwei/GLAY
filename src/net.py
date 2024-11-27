from src.model import mymodelycbcr,DenoisingCNN
from src.mamba import VMUNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, filters=32):
        super().__init__()
        self.Unet=VMUNet(input_channels=3)
        self.denoise = DenoisingCNN(64)

    def forward(self, inputs):
        inputs_denoise = self.denoise(inputs)
        out_unet = self.Unet(inputs)
        final = out_unet + inputs_denoise
        return final