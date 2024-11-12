import torch
from src.utils import PairedImageDataset,MixUp_AUG,GradualWarmupScheduler,torchPSNR,torchSSIM,network_parameters
from torch.utils.data import DataLoader
from thop import profile
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from src.utils import create_dataloaders
from src.model import mymodelycbcr,net
from tqdm import tqdm

to_pil = transforms.ToPILImage()

model_dir='/home/huangweiyan/workspace/model/cv/checkpoint'
model_restored=net()
model_restored.eval()

checkpoint=torch.load(os.path.join(model_dir,'model_bestSSIM.pth'))
model_restored.load_state_dict(checkpoint['state_dict'],strict=False)
savedir='/home/huangweiyan/workspace/model/cv/data/LOLv1/save2test'
_,test_loader=create_dataloaders(train='/home/huangweiyan/workspace/model/cv/data/LOLv1/Train',
                                    test='/home/huangweiyan/workspace/model/cv/data/LOLv1/Test',
                                    crop_size=256,augimg=True,batch_size=8)
for idx, data in enumerate(tqdm(test_loader, desc="save images")):
    target = data[0]
    input = data[1]
    name = data[2][0]
    
    # 预测并处理结果
    restored = model_restored(input).squeeze(0)
    image_pil = to_pil(restored)
    
    # 保存图像
    image_pil.save(os.path.join(savedir, name))





