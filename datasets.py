from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pdb
from collections import OrderedDict
import pandas as pd


class TCRDataset(Dataset):
    """TCR CDR3 dataset for the DIM-TCR"""

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy',
                 data_suffix = '_gd', data_type = '_tcr'):
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path, header=None)
        self.data = list(self.data[0])
        self.suffix = data_suffix
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = self.data[idx]
        #tcr = np.load(f'{self.root_dir}/{idx}{self.data_type}{self.suffix}.npy')
        tcr = np.load(f'{self.root_dir}/benchmark_dataset{idx}{self.suffix}.npy')
        return tcr

    def input_size(self):
        return self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info

class TCRclassify(Dataset):
    """TCR CDR3 dataset for the CNN classifier"""

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy', data_suffix = '_gd'):
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path, header=None)
        self.data = list(self.data[0])
        self.suffix = data_suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = self.data[idx]
        tcr = np.load(f'{self.root_dir}/benchmark_dataset{idx}{self.suffix}.npy')
        target = np.load(f'{self.root_dir}/benchmark_dataset{idx}_targets.npy')
        target = np.argmax(target,axis=1)
        return tcr, target

    def input_size(self):
        return self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info

class TCRautoencoder(Dataset):
    """TCR CDR3 only dataset"""

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy', data_suffix = '_gd'):
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path, header=None)
        self.data = list(self.data[0])
        self.suffix = data_suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #tcr = np.load(f'{self.root_dir}/{idx}_tcr{self.suffix}.npy')
        tcr = np.load(f'{self.root_dir}/benchmark_dataset{idx}{self.suffix}.npy')
        return tcr

    def input_size(self):
        return self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info


class TCRPepcontrastive(Dataset):
    """Dataset for the fullDIM (TCR -pep) contrastive loss"""

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy', data_suffix = '_gd'):
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path, header=None)
        self.data = list(self.data[0])
        self.suffix = data_suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = self.data[idx]
        tcr = np.load(f'{self.root_dir}/batch_{idx}_tcr_gd.npy')
        pep = np.load(f'{self.root_dir}/batch_{idx}_pep_gd.npy')
        target = np.load(f'{self.root_dir}/batch_{idx}_target.npy')
        return tcr,pep,target

    def input_size(self):
        return self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info


def get_dataset(opt, exp_dir):
    '''
    This function initializes the specific dataset.
    '''

    if opt.dataset == 'tcr':
        dataset = TCRDataset(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file,
                             data_suffix = opt.suffix, data_type = opt.datatype)
    elif opt.dataset == 'classify':
        dataset = TCRclassify(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file, data_suffix = opt.suffix)
    elif opt.dataset == 'ae':
        dataset = TCRautoencoder(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file, data_suffix = opt.suffix)
    elif opt.dataset == 'contrastive':
        dataset = TCRPepcontrastive(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file, data_suffix = opt.suffix)
    else:
        raise NotImplementedError()

    #TODO: add the caching argument and in the datasets
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,shuffle=False,num_workers=1)
    return dataloader

