import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn



class Trainer:
    """Class for train and evaluation of the model"""
    
    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
        
        print("="*80)
        print("Trainer initialized successfully")
        print(f"  Device: {self.model.device}")
        print(f"  Multi-GPU: {isinstance(self.model, nn.DataParallel)}")
        if isinstance(self.model, nn.DataParallel):
            print(f"  GPU Count: {len(opt['gpu_ids'])}")
        print("="*80)
    
    def train(self, train_loader, val_loader=None):
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        self.model.diffusion.train()
        train_losses = []

        for epoch in range(self.opt['train']['epochs']):
            # train
            self.model.diffusion.train()

            epoch_loss = 0
            num_batches = len(train_loader)
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.opt["train"]["epochs"]}')

            for _, train_data in enumerate(pbar):
                # train one sample
                self.model.load_one_sample(train_data)
                self.model.train_one_sample()

                # update scheduler
                self.model.scheduler.step()

                # print loss
                loss = self.model.log_dict['loss']
                epoch_loss += loss
                pbar.set_postfix({'loss': f'{loss:.3f}'})

            # logging train info
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")

            if (epoch + 1) % self.opt['train']['save_epoch_freq'] == 0:
                self.model.save_network(epoch)
                


