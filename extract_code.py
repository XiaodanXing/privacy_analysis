import argparse
import os
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm
from biDataset import biDatasetGeneral
from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, filename in pbar:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--ckpt', type=str,default='.test_512/checkpoint/vqvae_001.pt')
    parser.add_argument('--name', type=str,default='.test_512/features')
    parser.add_argument('--path', type=str,default='/media/NAS03/xiaodan/pathology/imgs/train/Fat')
    parser.add_argument('--channel', type=int,default=1)

    args = parser.parse_args()

    device = 'cuda'


    dataset = biDatasetGeneral(args.path, resolution=args.size, num_channels=args.channel)
    loader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=4)

    model = VQVAE(in_channel=args.channel)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)
