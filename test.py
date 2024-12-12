from src.glnet import net
import torch
import os
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from os.path import join, isfile
from PIL import Image
import numpy as np
from os import listdir

import cv2
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from src.utils import get_training_set, is_image_file,network_parameters

checkpoint=checkpoint=torch.load(os.path.join('models/lolv1wkanb/','best.pth'))
model=net().cuda()
model.load_state_dict(checkpoint,strict=True)
device = torch.device('cuda:0')
trans = transforms.Compose([
    ToTensor()
])
channel_swap = (1, 2, 0)
model.eval()
testset='/home/huangweiyan/workspace/model/cv/data/LOLv1/Test'
# Pay attention to the data structure
test_LL_folder = os.path.join(testset, "input")
test_NL_folder = os.path.join(testset, "target")
test_est_folder = os.path.join(testset, "temp")
try:
    os.stat(test_est_folder)
except:
    os.makedirs(test_est_folder)
test_LL_list = [join(test_LL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
est_list = [join(test_est_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]

for i in range(test_LL_list.__len__()):
    with torch.no_grad():
        LL_images = Image.open(test_LL_list[i]).convert('RGB')
        width, height = LL_images.size

        # 计算新的宽度和高度，使其都是8的倍数
        new_width = width - (width % 8)
        new_height = height - (height % 8)

        # 调整图像大小
        resized_LL_image = LL_images.resize((new_width, new_height))
        img_in = trans(resized_LL_image)

        LL_tensor = img_in.unsqueeze(0).to(device)

        # 使用模型进行推理
        prediction = model(LL_tensor)
        # 如果需要，进行预测结果的后处理
        prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
        prediction = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)

        # 将NumPy数组转换回Pillow图像
        est_image = Image.fromarray(prediction)

        # 将调整后的预测结果图像调整回原始尺寸
        resized_est_image = est_image.resize((width, height), resample=Image.BILINEAR)

        # 保存调整后的预测结果图像
        resized_est_image.save(est_list[i])
psnr_score = 0.0
ssim_score = 0.0
for i in range(test_NL_list.__len__()):
    gt = cv2.imread(test_NL_list[i])
    est = cv2.imread(est_list[i])
    psnr_val = compare_psnr(gt, est, data_range=255)
    ssim_val = compare_ssim(gt, est, channel_axis=-1)
    psnr_score = psnr_score + psnr_val
    ssim_score = ssim_score + ssim_val
psnr_score = psnr_score / (test_NL_list.__len__())
ssim_score = ssim_score / (test_NL_list.__len__())
print(psnr_score)
