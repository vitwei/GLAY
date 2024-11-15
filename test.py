from src.model import net
import torch
model=net()
model(torch.rand(1,3,400,400))
model(torch.rand(1,3,400,600))