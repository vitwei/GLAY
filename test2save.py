from src.model import mymodel
import torch
from src.utils import PairedImageDataset,MixUp_AUG,GradualWarmupScheduler,torchPSNR,torchSSIM,network_parameters
from torch.utils.data import DataLoader
from thop import profile
import os
from PIL import Image
from torchvision import transforms
import numpy as np

model_dir='/home/huangweiyan/workspace/model/cv/checkpoint'

img_path='/home/huangweiyan/workspace/model/cv/data/LOLv1/Test/input/1.png'
img = Image.open(img_path).convert("RGB")  # 转换为 RGB 格式
target_img_path='/home/huangweiyan/workspace/model/cv/data/LOLv1/Test/target/1.png'# 转换为 RGB 格式
target_img =  Image.open(target_img_path).convert("RGB")
# 转换为 Tensor
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img).unsqueeze(0)
target_img_tensor=  to_tensor(target_img).unsqueeze(0)

test=mymodel()
test.eval()
#test=test.cuda()
checkpoint = torch.load(os.path.join(model_dir, "model_bestPSNR.pth"),map_location=torch.device('cpu'))
test.load_state_dict(checkpoint['state_dict'],strict=False)

res=test(img_tensor)#.cuda())
res=test._rgb_to_ycbcr(res)
brightness_tensor = res[:, 0, :, :]
target=test._rgb_to_ycbcr(target_img_tensor)
target_tensor = target[:, 0, :, :]
# 去掉 batch 维度 (假设只有一个样本)
brightness_tensor = brightness_tensor.squeeze(0)

# 转换为 numpy 数组并规范化到 [0, 255]
brightness_np = brightness_tensor.detach().numpy()
brightness_np = (brightness_np - brightness_np.min()) / (brightness_np.max() - brightness_np.min())  # 归一化到 [0, 1]
brightness_np = (brightness_np * 255).astype(np.uint8)  # 缩放到 [0, 255]，并转换为 uint8

# 使用 PIL 将 numpy 数组转换为图像
brightness_img = Image.fromarray(brightness_np, mode='L')  # 使用 'L' 模式表示灰度图

# 保存图像
brightness_img.save('brightness_image.png')

#flops,params=profile(test,inputs=(torch.randn(1,3,256,256)))

#train_ds=PairedImageDataset('/home/huangweiyan/workspace/model/cv/data/Train')
#train_loader = DataLoader(dataset=train_ds, batch_size=1, shuffle=True,num_workers=0)
#for idx,data_train in enumerate(train_loader):
    #print(idx)