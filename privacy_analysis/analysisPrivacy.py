import os
import matplotlib.pyplot as plt
import numpy as np

def normalize(value):
    return (value + 88000000)/(37000000)

fpath = '/home/xiaodan/PycharmProjects/evaluator/privacy'
ref = 'train_A1_part_bottom.txt'
ref = np.loadtxt(os.path.join(fpath,ref))
ref = normalize(ref)
print('real', 1 - np.sum(ref < np.mean(ref)) / (1271 * 32))
print('reference distance',np.mean(ref),np.std(ref))
for file in ['train_A2_stylegan0.00_bottom.txt',
             'train_A2_stylegan0.20_bottom.txt',
             'train_A2_stylegan0.40_bottom.txt',
             'train_A2_stylegan0.60_bottom.txt',
             'train_A2_stylegan0.80_bottom.txt',
             'train_A2_stylegan1.00_bottom.txt',
             'train_A2_diffusion_bottom.txt']:
    fname = os.path.join(fpath,file)
    privacy = np.loadtxt(fname)
    privacy = normalize(privacy)

    print(file, 1 - np.sum(privacy<np.mean(ref))/(1271*32))
    a = 1
