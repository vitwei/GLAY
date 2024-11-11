from test2save import split1
import torch
from src.model import mymodel
import os
from glob import glob
from evaluation import measure_dirs
from natsort import natsorted
model_dir='/home/huangweiyan/workspace/model/cv/checkpoint'
inp_dir='/home/huangweiyan/workspace/model/cv/data/LOL/Test/input'
out_dir='/home/huangweiyan/workspace/model/cv/data/LOL/save2test'
dirA='/home/huangweiyan/workspace/model/cv/data/LOL/Test/target'
model=mymodel(img_size=384)
model.eval()
fileslist=glob(os.path.join(model_dir, '*.pth'))
res=[]
for inp_dir in fileslist:
    checkpoint = torch.load(inp_dir,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    files = natsorted(glob(os.path.join(out_dir, '*.JPG'))
                  + glob(os.path.join(out_dir, '*.png'))
                  + glob(os.path.join(out_dir, '*.PNG')))
    split1(model,files)
    with open('res.txt','a') as file:
        file.writelines([os.path.basename(inp_dir),str(measure_dirs(dirA, out_dir, use_gpu=False, verbose=True,GT_mean=False))])
        file.writelines([os.path.basename(inp_dir),str(measure_dirs(dirA, out_dir, use_gpu=False, verbose=True,GT_mean=True))])
