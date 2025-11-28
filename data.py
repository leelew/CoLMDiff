import torch
from torch.utils.data import Dataset



class TrainDataset(Dataset):
    def __init__(self, opt, x_start, x_condition=None):
        super(TrainDataset, self).__init__()
        self.opt = opt
        self.x_start = x_start
        self.x_condition = x_condition
    
    def __len__(self):
        return self.x_start.size(0)
    
    def __getitem__(self, index):
        data_dict = {
            'x0': self.x_start[index]
        }
        
        if self.x_condition is not None:
            data_dict['cond'] = self.x_condition[index]
        
        return data_dict
