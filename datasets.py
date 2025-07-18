import h5py
import torch
import numpy as np
import torch.utils.data as Data

class H5Datasets(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['vis_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        vis = np.array(h5f['vis_patchs'][key])
        ir = np.array(h5f['ir_patchs'][key])
        h5f.close()
        return torch.Tensor(vis).cuda(), torch.Tensor(ir).cuda()