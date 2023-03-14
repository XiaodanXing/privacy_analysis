#!/bin/bash


cd ./privacy_analysis/feature_extraction/

extract="train_A1_all_diffusion"
path="./synthesis"
mkdir ./features/$extract
python extract_code.py --size 512 --channel 1 --ckpt vqvae_011.pt --name ./features/$extract --path $path/$extract

extract="train_A1"
path="./synthesis"
mkdir ./features/$extract
python extract_code.py --size 512 --channel 1 --ckpt vqvae_011.pt --name ./features/$extract --path $path/$extract

extract="train_A2"
path="./synthesis"
mkdir ./features/$extract
python extract_code.py --size 512 --channel 1 --ckpt vqvae_011.pt --name ./features/$extract --path $path/$extract
