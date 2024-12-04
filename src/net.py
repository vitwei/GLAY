from src.model import mymodelycbcr,DenoisingCNN
from src.mamba import VMUNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.glnet import glnet_4g
class net(nn.Module):
    def __init__(self, filters=32):
        super().__init__()
        self.VUnet=glnet_4g()
        self.denoise = DenoisingCNN(64)

    def forward(self, inputs):
        out_unet = self.VUnet(inputs)
        final=self.denoise(out_unet)
        return final