from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import os
import h5py
sep = os.sep
class MTM_dataset(Dataset):
    '''pytorch dataset for dataloader'''
    def __init__(self, h5path, csvpath, task, datatype):
        super().__init__()
        self.h5data = h5path
        self.csvpath = csvpath
        self.task = task
        # read data info
        csvinfo = pd.read_csv(self.csvpath)
        train_data_cl = list(csvinfo[csvinfo['dataset'] == 1]['ID'])
        train_data_seg, validation_data_seg = train_test_split(train_data_cl, test_size=0.3, random_state=42)
        # Select the .h5 file to use
        slicelist = os.listdir(h5path)
        self.chooseslicepath= []
        for slice in slicelist:
            slicename = slice.split('_')[0]
            if datatype == 'train':
                if slicename in train_data_seg:
                    self.chooseslicepath.append(os.path.join(h5path, slice))
            elif datatype == 'valid':
                if slicename in validation_data_seg:
                    self.chooseslicepath.append(os.path.join(h5path, slice))
            else:
                raise ValueError('The datatype must be set train or test.')

    def __len__(self):
        return len(self.chooseslicepath)

    def __getitem__(self,idx):
        patientpath = self.chooseslicepath[idx]
        # get data
        image = h5py.File(patientpath,'r')['image'][:]
        label = h5py.File(patientpath,'r')['label'][:]
        # tensor
        image_tensor = torch.from_numpy(image.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.float32))
        # add one channel
        image_tensor = image_tensor.unsqueeze(0)
        label_tensor = label_tensor.unsqueeze(0)
        return image_tensor, label_tensor

class MTM_dataset_all(Dataset):
    '''pytorch all dataset for dataloader'''
    def __init__(self, h5path, csvpath):
        super().__init__()
        self.h5data = h5path
        self.csvpath = csvpath
        # read data info
        csvinfo = pd.read_csv(csvpath)

        all_data = list(csvinfo[csvinfo['dataset'] == 0]['ID'])
        # Select the .h5 file to use
        slicelist = os.listdir(h5path)
        self.chooseslicepath= []
        for slice in slicelist:
            slicename = slice.split('_')[0]
            if slicename in all_data:
                self.chooseslicepath.append(os.path.join(h5path,slice))

    def __len__(self):
        return len(self.chooseslicepath)

    def __getitem__(self,idx):
        patientpath = self.chooseslicepath[idx]
        # get data
        image = h5py.File(patientpath, 'r')['image'][:]
        label = h5py.File(patientpath, 'r')['label'][:]
        # tensor
        image_tensor = torch.from_numpy(image.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.float32))
        # add one channel
        image_tensor = image_tensor.unsqueeze(0)
        label_tensor = label_tensor.unsqueeze(0)
        return image_tensor, label_tensor, patientpath

def data_loaders(datapath, pklpath, task, batch_size, workers):
    '''dataloader load dataset'''
    dataset_train = MTM_dataset(datapath, pklpath, task, datatype='train')
    dataset_valid = MTM_dataset(datapath, pklpath, task, datatype='valid')

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init,
    )
    return loader_train, loader_valid

def data_loaders_all(datapath, pklpath, batch_size, workers):
    dataset_all = MTM_dataset_all(datapath, pklpath)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_all = DataLoader(
        dataset_all,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init,
    )
    return loader_all