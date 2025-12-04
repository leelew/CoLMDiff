import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm

from .data import TrainDataset
from torch.utils.data import DataLoader



class Trainer:
    """Class for train and sampling of the model"""
    
    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
    
    def train(self, x):
        train_dataset = TrainDataset(x['x0'], x.get('cond', None))
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.opt['train']['batch_size'],
            shuffle=True,
            num_workers=self.opt['train'].get('num_data_workers', 6),
            pin_memory=True,
            drop_last=True
        )
        
        self.model.set_diffusion_schedule(self.opt['train']['noise_schedule'])
        self.model.diffusion.train()

        for epoch in range(self.opt['train']['epochs']):
            epoch_loss = 0
            num_batches = len(train_loader)
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.opt["train"]["epochs"]}')

            for _, train_data in enumerate(pbar):
                self.model.load_one_sample(train_data)
                self.model.train_one_sample()
                self.model.scheduler.step()

                loss = self.model.log_dict['loss']
                epoch_loss += loss
                pbar.set_postfix({'loss': f'{loss:.3f}'})

            avg_train_loss = epoch_loss / num_batches
            
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}") 

            if (epoch + 1) % self.opt['train']['save_epoch_freq'] == 0:
                self.model.save_network(epoch)

    def sampling(self, x_total, n_total, operator=None):
        self.model.set_diffusion_schedule(self.opt['sample']['noise_schedule'])
        self.model.diffusion.eval()

        gpus = [f"cuda:{i}" for i in self.opt['gpu_ids']]
        world_size = len(gpus)

        base = n_total // world_size
        n_per_gpu = [base] * world_size
        for i in range(n_total % world_size):
            n_per_gpu[i] += 1

        y = x_total.get('y', None)
        cond = x_total.get('cond', None)

        manager = mp.Manager()
        out_dict = manager.dict()

        procs = []
        for i in range(world_size):
            n_i = n_per_gpu[i]

            x_per_gpu = {
                'y': y[sum(n_per_gpu[:i]):sum(n_per_gpu[:i])+n_i] if y is not None else None,
                'cond': cond[sum(n_per_gpu[:i]):sum(n_per_gpu[:i])+n_i] if cond is not None else None
            }

            if n_i == 0:
                continue
            p = mp.Process(target=self._sample_worker, args=(i, world_size, n_i, self.model.diffusion.module, x_per_gpu, operator, out_dict))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        if 'samples' in out_dict:
            samples = out_dict['samples'].numpy()
            return samples
    
    def _sample_worker(self, rank, world_size, n_per_gpu, model, x_per_gpu, operator=None, out_dict=None):
        device = self.opt['gpu_ids'][rank]
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(device)

        model_i = model.to(device)
        for name, buffer in model_i.named_buffers():
            if buffer.device != device:
                buffer.data = buffer.data.to(device)

        if isinstance(x_per_gpu, dict):
            for key, item in x_per_gpu.items():
                if item is not None:
                    x_per_gpu[key] = item.to(device)
        x_gen = model_i.sample(x_per_gpu, n_per_gpu, operator=operator, device=device) 
        final = x_gen.detach().cpu()

        dist.all_gather(gathered := [torch.zeros_like(final, device=device) for _ in range(world_size)], final.to(device))
        if rank == 0:
            merged = torch.cat([g.cpu() for g in gathered], dim=0)
            out_dict['samples'] = merged 

        dist.destroy_process_group()

    def finetune(self, finetune_data, finetune_epochs=100):
        finetune_dataset = TrainDataset(finetune_data['x0'], finetune_data.get('cond', None))
        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=min(self.opt['train']['batch_size'], len(finetune_dataset)),
            shuffle=True,
            num_workers=self.opt['train'].get('num_data_workers', 6),
            pin_memory=True,
            drop_last=True
        )

        self.model.set_diffusion_schedule(self.opt['train']['noise_schedule'])
        self.model.diffusion.train()

        for epoch in range(finetune_epochs):            
            epoch_loss = 0
            num_batches = len(finetune_loader)
            pbar = tqdm(finetune_loader, desc=f'Epoch {epoch}/{finetune_epochs}')
            
            for _, finetune_batch in enumerate(pbar):
                self.model.load_one_sample(finetune_batch)
                self.model.train_one_sample()
                self.model.scheduler.step()
                
                loss = self.model.log_dict['loss']
                epoch_loss += loss
                pbar.set_postfix({'loss': f'{loss:.3f}'})
            
            avg_finetune_loss = epoch_loss / num_batches
            
            print(f" Epoch {epoch} Summary:")
            print(f"   Loss: {avg_finetune_loss:.4f}")