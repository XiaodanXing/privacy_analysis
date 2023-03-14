
import os
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from PIL import Image,ImageOps

import random

random.seed(0)


class biDatasetGeneral(Dataset):

    def __init__(self,filepath = '/media/NAS02/xiaodan/osic/montages_osic2/',

                 num_channels=1,resolution=1024):  # crop_size,


        self.imlist = []

        self.filepath = filepath


        for path, subdirs, files in os.walk(self.filepath):
            for name in files:
                self.imlist.append(os.path.join(path, name))


        self.transforms = transforms

        self.num_channels = num_channels
        self.transforms =  transforms.Compose([

            transforms.Resize([resolution,resolution]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    def __getitem__(self, idx):
        if idx < len(self.imlist):
            img_name = self.imlist[idx]

            img = Image.open(img_name)
            if self.num_channels ==1:
                img = ImageOps.grayscale(img)
            img = self.transforms(img)



            return img,img_name


    def __len__(self):
        return len(self.imlist)