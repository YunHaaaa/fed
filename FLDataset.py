import numpy as np
import random
import os
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class FLDataset(data.Dataset):
    def __init__(self, labels, samples = None, sample_paths = None, transformer=None):
        self.samples = samples
        self.labels = labels

        self.orders = list(range(len(labels)))
        random.shuffle(self.orders)

        self.path_root = os.path.expanduser('~/Datasets')
        self.sample_paths = sample_paths

        self._transformer = transformer

    def get_image(self, index):
        framename = self.path_root + '/' + self.sample_paths[index]
        img = Image.open(framename).convert('RGB')
        return img
    
    def __getitem__(self, index):
        img = None
        if self.sample_paths is None:
            arr = self.samples[index]
            if self.samples[index].shape[0] == 3: # CHW => HWC
                arr = np.transpose(self.samples[index], (1,2,0))
            img = Image.fromarray(arr, 'RGB')
        else:
            img = self.get_image(index)
        return self._transformer(img), int(self.labels[index])

    def __get_batch__(self, batch_size, batch_index, reset_order = False):
        if reset_order:
            random.shuffle(self.orders)

        idx_start = batch_index * batch_size
        idx_end = idx_start + batch_size
        if idx_end > len(self.labels):
            raise Exception("Batch out of index")

        imgs, labels = [], [], []
        for order_idx in range(idx_start, idx_end):
            idx = self.orders[order_idx]
            img, label =  self.__getitem__(idx)
            imgs.append(img)
            labels.append(label)
        return torch.stack(imgs), torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)