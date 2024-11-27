from torch.utils.data import DataLoader
import torch
from PIL import Image
import torch.optim as optim  
import torch.nn as nn  
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter   
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from src.utils import create_dataloaders,delete_files,checkpoint,copy_files_to_destination,checkpoint_last,torchPSNR,torchSSIM,network_parameters,GradualWarmupScheduler,get_training_set,calculate_psnr,calculate_ssim,create_dataloaders2,is_image_file
from src.loss import CombinedLoss,multi_VGGPerceptualLoss,temploss
import time
import torchvision.transforms as transforms
from timm.utils import NativeScaler
import random
import numpy as np
from src.mamba import net 
import os
import cv2
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from torchvision.transforms import ToTensor
from fvcore.nn import FlopCountAnalysis
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from src.GLMix.models.glnet import glnet_4g
def myeval(model, epoch, writer):
    print("==> Start testing")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tStart = time.time()
    trans = transforms.Compose([
        ToTensor()
    ])
    channel_swap = (1, 2, 0)
    model.eval()
    # Pay attention to the data structure
    test_LL_folder = os.path.join('/home/huangweiyan/workspace/model/cv/data/LOLv1/Test', "input")
    test_NL_folder = os.path.join('/home/huangweiyan/workspace/model/cv/data/LOLv1/Test', "target")
    test_est_folder = os.path.join('output','mymodel', 'last')
    test_est_folder_best = os.path.join('output','mymodel', 'best')

    try:
        os.stat(test_est_folder)
    except:
        os.makedirs(test_est_folder)

    try:
        os.stat(test_est_folder_best)
    except:
        os.makedirs(test_est_folder_best)

    test_LL_list = [os.path.join(test_LL_folder, x) for x in sorted(os.listdir(test_LL_folder)) if is_image_file(x)]
    test_NL_list = [os.path.join(test_NL_folder, x) for x in sorted(os.listdir(test_LL_folder)) if is_image_file(x)]
    est_list = [os.path.join(test_est_folder, x) for x in sorted(os.listdir(test_LL_folder)) if is_image_file(x)]
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
    save_folder = os.path.join('models', 'mymodel')
    file_checkpoint_last = checkpoint_last(model, save_folder)
    global best_psnr
    if best_psnr > psnr_score:
        delete_files(est_list)
    else:
        file_checkpoint = checkpoint(model, save_folder)
        best_psnr = psnr_score
        copy_files_to_destination(est_list, test_est_folder_best)
    print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
    print("best_psnr:", best_psnr)
    writer.add_scalar('psnr', psnr_score, epoch)
    writer.add_scalar('ssim', ssim_score, epoch)
    return psnr_score, ssim_score
BATCH=8
img_size=256


model_dir='/home/huangweiyan/workspace/model/cv/checkpointback'
model_restored=net()
model_restored.cuda()

#checkpoint=torch.load(os.path.join(model_dir,'model_latest.pth'))
#model_restored.load_state_dict(checkpoint['state_dict'],strict=False)


with torch.no_grad():
    model_restored.eval()
    input=(torch.rand(1,3,400,600).cuda(),)
    flops=FlopCountAnalysis(model_restored,input)

loss_scaler = NativeScaler()
log_dir = os.path.join('/home/huangweiyan/workspace/model/cv/log', time.strftime('%m%d_%H%M'))
writer = SummaryWriter(log_dir)
scaler = amp.GradScaler()


train_loader,test_loader=create_dataloaders2(train='/home/huangweiyan/workspace/model/cv/data/LOLv1/Train',
                                    test='/home/huangweiyan/workspace/model/cv/data/LOLv1/Test',
                                    crop_size=img_size,augimg=True,batch_size=BATCH)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Charloss = multi_VGGPerceptualLoss().to(device)
Charloss=CombinedLoss('cuda')
#Charloss = nn.SmoothL1Loss()
#Charloss=temploss().to(device)
EPOCHS=601
TEST_AFTER=4
p_number = network_parameters(model_restored)

best_psnr = 0
best_ssim = 0

best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

LR_INITIAL=5e-4
#optimizer=optim.SGD(model_restored.parameters(), lr=LR_INITIAL, momentum=0.9)
optimizer = optim.Adam(model_restored.parameters(), lr=LR_INITIAL, betas=(0.9, 0.999),eps=1e-8)
#scheduler_cosine =CosineAnnealingRestartLR(optimizer, periods=[300,700], restart_weights=(1,0.02))
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS-10, eta_min=5e-5)

scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_cosine)
scheduler.step()



print(f'''==> Training details:
------------------------------------------------------------------
    patches size: {str(img_size)}
    Model parameters:   {p_number}
    Start/End epochs:   {str(1) + '~' + str(EPOCHS)}
    Batch sizes:        {BATCH}
    Learning rate:      {LR_INITIAL}
    Flops:              {flops.total()/(1024*1024*1024)}
    
''')
print('------------------------------------------------------------------')


print("\nEvaluation after every {} Iterations !!!\n".format(TEST_AFTER))
torch.cuda.empty_cache()
for epoch in range(1,EPOCHS+1):
    epoch_start_time = time.time()
    epoch_loss = 0
    test_loss=0
    train_id = 1
    model_restored.train()
    #if epoch>300:
       # optimizer=optimizer_back
    for idx,data in enumerate(train_loader):
        optimizer.zero_grad()
        target = data[0].cuda()
        input = data[1].cuda()
        o1 = model_restored(input)
        loss = Charloss(target,o1)
        #loss = Charloss(o1 ,target)
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

    if epoch % TEST_AFTER == 0:
            psnr_score, ssim_score = myeval(model_restored, epoch, writer)
            torch.cuda.empty_cache()

    scheduler.step()
    torch.save({'epoch': epoch, 
            'state_dict': model_restored.state_dict(),
            'optimizer' : optimizer.state_dict()
            }, os.path.join(model_dir,"model_latest.pth")) 
    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()
total_finish_time = (time.time() - total_start_time)  # seconds
torch.save({'epoch': epoch, 
        'state_dict': model_restored.state_dict(),
        'optimizer' : optimizer.state_dict()
        }, os.path.join(model_dir,"model_res.pth")) 
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))



