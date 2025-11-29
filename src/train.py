import numpy as np
import torch
from torch.utils.data import DataLoader

from data import TrainDataset
from model import DDPM
from trainer import Trainer
from config import parse_args



def train(train_data=None, val_data=None):
##############################################################################
# Loading configuration
##############################################################################
    opt = parse_args()
    
    if train_data is None or val_data is None:
        train_data = {
            'x_start': torch.from_numpy(np.load(opt["path_data"]+'train_x_start.npy', allow_pickle=True)),
        }
        val_data = {
            'x_start': torch.from_numpy(np.load(opt["path_data"]+'val_x_start.npy', allow_pickle=True)),
        }
        if opt.get('conditioned', False):
            train_data['x_condition'] = torch.from_numpy(np.load(opt["path_data"]+'train_x_condition.npy', allow_pickle=True))
            val_data['x_condition'] = torch.from_numpy(np.load(opt["path_data"]+'val_x_condition.npy', allow_pickle=True))

##############################################################################
# Loading data & create dataloader
##############################################################################
    if opt["phase"] == 'train':
        train_dataset = TrainDataset(opt, train_data['x_start'], train_data.get('x_condition', None))
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt['train']['batch_size'],
            shuffle=True,
            num_workers=opt['train'].get('num_data_workers', 6),
            pin_memory=True,
            drop_last=True
        )
        
        val_dataset = TrainDataset(opt, val_data['x_start'], val_data.get('x_condition', None))
        val_loader = DataLoader(
            val_dataset,
            batch_size=opt['train']['batch_size'],
            shuffle=False,
            num_workers=opt['train'].get('num_data_workers', 4),
            pin_memory=True
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

##############################################################################
# Training or testing
##############################################################################
    model = DDPM(opt)

    trainer = Trainer(model, opt)
    if opt["phase"] == 'train':
        trainer.train(train_loader, val_loader)
        


if __name__ == '__main__':
    # train_data = {
    #     'x_start': torch.randn(100, 3, 64, 64),
    #     'x_condition': torch.randn(100, 3, 64, 64)
    # }
    # val_data = {
    #     'x_start': torch.randn(20, 3, 64, 64),
    #     'x_condition': torch.randn(20, 3, 64, 64)
    # }
    train()