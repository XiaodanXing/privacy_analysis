import cv2
import numpy as np
import os
from tqdm import tqdm
import torch.nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageOps
import random
from dataset import LMDBDataset
import argparse


import torch

parser = argparse.ArgumentParser()

parser.add_argument("--syn_method", default='train_A1_all_diffusion')
parser.add_argument("--real_method", default='train_A2')
parser.add_argument("--filepath", default='/home/xiaodan/PycharmProjects/privacy_analysis/image_compressing/features')
parser.add_argument("--label", default='nope')
parser.add_argument("--class_names", default='pe,nope')
parser.add_argument("--out_dir", default='./avg_images')

args = parser.parse_args()




random.seed(0)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose(1,0)))
    return distH


def compute_privacy():
    # for all in real:
    #   for all in fake:
    #     mse(real,fake)
    # tolerace = 0.6
    for syn_method in args.syn_method.split(','):
        print(syn_method)
        real_path = os.path.join(args.filepath,args.real_method,)
        true_dataset = LMDBDataset(real_path)
        true_loader = DataLoader(
            true_dataset, batch_size=32, shuffle=True, num_workers=16, drop_last=True
        )


        syn_path = os.path.join(args.filepath,syn_method,)
        syn_dataset = LMDBDataset(syn_path)
        syn_loader = DataLoader(
            syn_dataset, batch_size=32, shuffle=True, num_workers=16, drop_last=True
        )

        dists2 = []
        for top_real, bottom_real, _ in tqdm(true_loader):
            # dist1 = 0
            dist2 = 0
            num = syn_loader.__len__()
            for j, (top, bottom, _) in enumerate(syn_loader):
                # dist1_part = np.mean(CalcHammingDist(top_real.flatten(1),top.flatten(1)),axis=1)
                dist2_part = np.mean(CalcHammingDist(bottom_real.flatten(1),bottom.flatten(1)),axis=1)
                # dist1 += dist1_part
                dist2 += dist2_part
            # dists1.append(dist1/num)
            dists2.append(dist2/num)

        # np.savetxt('./privacy/%s_top.txt' % (args.syn_method), dists1)
        np.savetxt('./privacy/%s_bottom.txt' % (syn_method), dists2)
    a = 1

compute_privacy()