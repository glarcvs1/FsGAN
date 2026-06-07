import dataset
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='configs/dce_lofgan.yaml')
    parser.add_argument('--output_dir', type=str, default='results/dce_lofgan')
    parser.add_argument('-r', "--resume", action="store_true")
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    config = get_config(args.conf)
    data = np.load(config['data_root'])
    data=data[16:]
    data_for_gen = data[:, :125, :, :, :]
    print(data_for_gen)