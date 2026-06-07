import os
import random
import shutil

import cv2
import lpips
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torch.utils.data
import torchvision.transforms as transforms
from trainer import Trainer
from utils import get_config, unloader, get_model_list
import matplotlib.pyplot as plt
from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,default='results/vggface_lofgan')
parser.add_argument('--dataset', type=str,default='vggface')
#parser.add_argument('--real_dir', type=str,default='datasets/for_fid/flower')
#parser.add_argument('--fake_dir', type=str,default='animal')
parser.add_argument('--ckpt', type=str, default='gen_00100000.pt')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_sample_test', type=int, default=3)
parser.add_argument('--n_test', type=int, default=3)
args = parser.parse_args()

conf_file = os.path.join(args.name, 'configs.yaml')
config = get_config(conf_file)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)


if __name__ == '__main__':
    # SEED = 0
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    # real_dir = args.real_dir
    # fake_dir = os.path.join(args.name, args.fake_dir)
    # print('real dir: ', real_dir)
    # print('fake dir: ', fake_dir)
    #
    # if os.path.exists(fake_dir):
    #     shutil.rmtree(fake_dir)
    # os.makedirs(fake_dir, exist_ok=True)
    # if os.path.exists(real_dir):
    #     shutil.rmtree(real_dir)
    # os.makedirs(real_dir, exist_ok=True)

    data = np.load(config['data_root'])
    if args.dataset == 'flower':
        data = data[85:]
        num = 12
    elif args.dataset == 'animal':
        data = data[119:]
        num = 30
    elif args.dataset == 'vggface':
        data = data[1802:]
        num = 30
    elif args.dataset == 'dce':
        #data = data[:]
        num = 125


    #per = np.random.permutation(data.shape[1])
    #data = data[:, per, :, :, :]


    data_for_gen = data[:, :num, :, :, :]
    data_for_fid = data[:, :num, :, :, :]


    #if os.path.exists(fake_dir):
    trainer = Trainer(config)
    if args.ckpt:
        last_model_name = os.path.join(args.name, 'chs', args.ckpt)
        print(last_model_name)
    else:
        print("不存在")
    trainer.load_ckpt(last_model_name)
    trainer.cuda()
    trainer.eval()
    out=[]
    for cls in tqdm(range(data_for_gen.shape[0]), desc='generating fake images'):
        ncls=[]
        for i in range(2000):
            idx = np.random.choice(data_for_gen.shape[1], args.n_sample_test)
            imgs = data_for_gen[cls, idx, :, :, :]
            imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], dim=0).unsqueeze(0).cuda()
            fake_x = trainer.generate(imgs)
            output = unloader(fake_x[0].cpu())
            output=np.array(output)
            #print(output)
            if i==0:
                ncls=output.copy()
            elif ncls.shape==output.shape:
                ncls=np.dstack([[ncls, output]])
            else:
                ncls = np.insert(ncls,i, output, axis=0)
            #print(ncls.shape)
            #output.save(os.path.join(path, '{}_{}.png'.format(cls, str(i).zfill(3))), 'png')
        if cls == 0:
            out = ncls.copy()
        elif out.shape == ncls.shape:
            out = np.dstack([[out, ncls]])
        else:
            out = np.insert(out, cls, ncls, axis=0)
        print(out.shape)
    np.save('classification/input/'+args.dataset+'_pro2000_1.npy', out) #_1为lof _6为net6