from torch.utils.data import DataLoader
import torch
import torch.optim as optim  
import torch.nn as nn  
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter   
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from src.utils import create_dataloaders,torchPSNR,torchSSIM,network_parameters,GradualWarmupScheduler,CosineAnnealingRestartLR,calculate_psnr,calculate_ssim
from src.loss import CombinedLoss,multi_VGGPerceptualLoss,temploss
import time
from timm.utils import NativeScaler
import random
import numpy as np
from src.model import net 
import os
from fvcore.nn import FlopCountAnalysis
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

BATCH=32
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


train_loader,test_loader,radom_loader=create_dataloaders(train='/home/huangweiyan/workspace/model/cv/data/LOLv1/Train',
                                    test='/home/huangweiyan/workspace/model/cv/data/LOLv1/Test',
                                    crop_size=img_size,augimg=True,batch_size=BATCH)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Charloss = multi_VGGPerceptualLoss().to(device)
#Charloss=CombinedLoss('cuda')
#Charloss=temploss().to(device)
EPOCHS=1030
TEST_AFTER=4
p_number = network_parameters(model_restored)

best_psnr = 22.7
best_ssim = 0.8174

best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

LR_INITIAL=5e-5
#optimizer=optim.SGD(model_restored.parameters(), lr=LR_INITIAL, momentum=0.9)
optimizer = optim.Adam(model_restored.parameters(), lr=LR_INITIAL, betas=(0.9, 0.999),eps=1e-8, weight_decay=0.02)
#scheduler_cosine =CosineAnnealingRestartLR(optimizer, periods=[300,700], restart_weights=(1,0.02))
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-8)

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
        o1,o2,o3 = model_restored(input)
        #loss = Charloss(target,o1)
        loss = Charloss(o1,o2,o3, target)
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

    if epoch % TEST_AFTER == 0:
        for name, param in model_restored.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"{name}/grad", param.grad, epoch)  # 梯度直方图
                writer.add_scalar(f"{name}/grad_norm", param.grad.norm(), epoch)  # 梯度范数
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        with torch.no_grad():
            for ii, test in enumerate(test_loader):
                target = test[0].cuda()
                input = test[1].cuda()
                h, w = target.shape[2], target.shape[3]
                o1,o2,o3= model_restored(input)
                loss = Charloss(o1,o2,o3, target)
                #restored = restored[:, :, :h, :w]
                #loss = Charloss(target,o1)
                test_loss +=loss.item()
                for res, tar in zip(o1, target):
                    psnr_val_rgb.append(torchPSNR(res, tar))
                    ssim_val_rgb.append(torchSSIM(o1, target))
            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch_psnr = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restored.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestPSNR.pth"))
            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
                epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

            # Save the best SSIM model of validation
            if ssim_val_rgb > best_ssim:
                best_ssim = ssim_val_rgb
                best_epoch_ssim = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restored.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestSSIM.pth"))
            print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
                epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))    

            writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
            writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
            writer.add_scalar('val/loss', test_loss, epoch)
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