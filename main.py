import argparse
import yaml

import torch
from torch.utils.data import DataLoader

from data_loader import DiffDataset
from model import DDPM
from trainer import Trainer



def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPM Model')
    parser.add_argument('--path_config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu_ids', type=str, default='1,2,3', help='GPU IDs (e.g., 0,1,2,3)')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help='Phase: train or test')
    return parser.parse_args()


def main(train_data, val_data=None):
##############################################################################
# Loading configuration
##############################################################################
    args = parse_args()
    with open(args.path_config, 'r') as f:
        opt = yaml.safe_load(f)
    opt["gpu_ids"] = [int(x.strip()) for x in args.gpu_ids.split(',')]
    opt["phase"] = args.phase
    
##############################################################################
# Loading data & create dataloader
##############################################################################
    if opt["phase"] == 'train':
        train_dataset = DiffDataset(opt, train_data['x_start'], train_data.get('x_condition', None))
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt['train']['batch_size'],
            shuffle=True,
            num_workers=opt['train'].get('num_data_workers', 6),
            pin_memory=True,
            drop_last=True
        )
        
        val_dataset = DiffDataset(opt, val_data['x_start'], val_data.get('x_condition', None))
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
    train_data = {
        'x_start': torch.randn(100, 3, 64, 64),
        'x_condition': torch.randn(100, 3, 64, 64)
    }
    val_data = {
        'x_start': torch.randn(20, 3, 64, 64),
        'x_condition': torch.randn(20, 3, 64, 64)
    }
    main(train_data, val_data)