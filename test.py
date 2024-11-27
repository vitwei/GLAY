from src.model import net
import torch
model=net().cuda()
a=model(torch.rand(1,3,400,600).cuda())
