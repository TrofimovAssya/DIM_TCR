from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pdb
from collections import OrderedDict
import pandas as pd


class TCRDataset(Dataset):
    """TCR CDR3 dataset"""

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy', nb_kmer = 1000):
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path, header=None)
        self.data = list(self.data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tcr = np.load(f'{self.root_dir}/{idx}_tcr.npy')
        
        return tcr

    def input_size(self):
        return self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info



def get_dataset(opt, exp_dir):

    if opt.dataset == 'tcr':
        dataset = TCRDataset(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file, nb_kmer = opt.nb_kmer)
    else:
        raise NotImplementedError()

    #TODO: check the num_worker, might be important later on, for when we will use a bunch of big files.
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,shuffle=False,num_workers=1)
    return dataloader

