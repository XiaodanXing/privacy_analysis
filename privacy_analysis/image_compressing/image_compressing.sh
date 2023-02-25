#!/bin/bash


cd /home/xiaodan/PycharmProjects/privacy_analysis/image_compressing/

extract="train_A1_all_diffusion"
path="/media/NAS04/chestXpert/CheXpert-v1.0/synthesis"
mkdir ./features/$extract
python extract_code.py --size 512 --channel 1 --ckpt vqvae_011.pt --name ./features/$extract --path $path/$extract

extract="train_A1"
path="/media/NAS04/chestXpert/CheXpert-v1.0/synthesis"
mkdir ./features/$extract
python extract_code.py --size 512 --channel 1 --ckpt vqvae_011.pt --name ./features/$extract --path $path/$extract

extract="train_A2"
path="/media/NAS04/chestXpert/CheXpert-v1.0/synthesis"
mkdir ./features/$extract
python extract_code.py --size 512 --channel 1 --ckpt vqvae_011.pt --name ./features/$extract --path $path/$extract
