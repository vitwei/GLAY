import glob
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import cv2

from natsort import natsort
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips



class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        score, diff = ssim(imgA, imgB,full=True, multichannel=True,channel_axis=2)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)
        return psnr_val


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

def format_result(psnr, ssim, lpips):
    return f'{psnr:0.2f}, {ssim:0.3f} {lpips:0.3f}'

def measure_dirs(dirA, dirB, use_gpu, verbose=False,GT_mean=True):
    type_='png'
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type_}'))
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type_}'))

    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)

    measure = Measure(use_gpu=use_gpu)

    results = []
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()
        t = time.time()
        IMGA=imread(pathA)
        IMGB=imread(pathB)
        if  GT_mean==True:
            IMGA = np.array(IMGA) 
            IMGB = np.array(IMGB)
            mean_target = cv2.cvtColor(IMGA, cv2.COLOR_RGB2GRAY).mean()
            mean_restored = cv2.cvtColor(IMGB, cv2.COLOR_RGB2GRAY).mean()
            IMGB = np.clip(IMGB * (mean_target/mean_restored), 0, 255)
            IMGB=IMGB.astype(np.uint8)
            IMGA=IMGA.astype(np.uint8)

        result['psnr'], result['ssim'], result['lpips'] = measure.measure(IMGA, IMGB)
        d = time.time() - t
        vprint(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(**result)}, {d:0.1f}")

        results.append(result)

    psnr = np.mean([result['psnr'] for result in results])
    ssim = np.mean([result['ssim'] for result in results])
    lpips = np.mean([result['lpips'] for result in results])

    vprint(f"Final Result: {format_result(psnr, ssim,lpips)}, {time.time() - t_init:0.1f}s")
    return [psnr.item(),ssim.item(),lpips.item(),GT_mean]




dirB='/home/huangweiyan/workspace/model/cv/data/LOLv1/save2test'
dirA='/home/huangweiyan/workspace/model/cv/data/LOLv1/Test/target'
measure_dirs(dirA, dirB, use_gpu=False, verbose=True,GT_mean=True)