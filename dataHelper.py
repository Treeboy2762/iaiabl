from __future__ import division
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import skimage.io as sio

class MRIDataset(Dataset):

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.fnames = os.listdir(data_dir)
        self.map_dict = {'y': 1, 'no': 0}

    def __len__(self):

        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):

        fname = self.fnames[idx]
        img_path = os.path.join(self.data_dir, fname)
        img = sio.imread(img_path, as_gray=True)[np.newaxis, :]
        img = (img / 255).astype(np.float32)
        if fname.startswith('y'):
            label = 1
        else:
            label = 0
        
        return img, label

class MRIMaskedDataset(Dataset):

    def __init__(self, data_dir):

        self.data_dir = data_dir
        fnames = os.listdir(data_dir)
        self.fnames = [fname for fname in fnames if not fname.endswith('mask.png')]
        self.map_dict = {'y': 1, 'no': 0}

    def __len__(self):

        return len(self.fnames)

    def __getitem__(self, idx):

        fname = self.fnames[idx]
        img_path = os.path.join(self.data_dir, fname)
        mask_path = os.path.join(self.data_dir, fname[:-4] + '_mask.png')

        img = sio.imread(img_path, as_gray=True)
        img = (img / 255).astype(np.float32)
        mask = sio.imread(mask_path, as_gray=True)
        if fname.startswith('y'):
            label = 1
        else:
            label = 0
        
        img_with_mask = np.stack([img, mask])
        return img_with_mask, label