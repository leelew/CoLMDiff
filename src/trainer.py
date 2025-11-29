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



class Trainer:
    """Class for train and evaluation of the model"""
    
    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
        
        print("="*80)
        print("Trainer initialized successfully")
        print(f"  Device: {self.model.device}")
        print(f"  Multi-GPU: {isinstance(self.model.diffusion, nn.DataParallel)}")
        if isinstance(self.model.diffusion, nn.DataParallel):
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

    def _sample_worker(self, rank, world_size, n_per_gpu, model, x_per_gpu, path_sampling):
        device = self.opt['gpu_ids'][rank]
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(device)

        model_i = model.to(device)
        if isinstance(x_per_gpu, dict):
            for key, item in x_per_gpu.items():
                if item is not None:
                    x_per_gpu[key] = item.to(device)

        x_gen = model_i.sample(x_per_gpu, n_per_gpu, device=device)
        final = x_gen.detach().cpu()

        dist.all_gather(gathered := [torch.zeros_like(final, device=device) for _ in range(world_size)], final.to(device))
        if rank == 0:
            merged = torch.cat([g.cpu() for g in gathered], dim=0)
            torch.save(merged, path_sampling)

        dist.destroy_process_group()

    def sampling(self, x_total, n_total, path_sampling):
        gpus = [f"cuda:{i}" for i in self.opt['gpu_ids']]
        world_size = len(gpus)

        base = n_total // world_size
        n_per_gpu = [base] * world_size
        for i in range(n_total % world_size):
            n_per_gpu[i] += 1

        y = x_total.get('y', None)
        cond = x_total.get('cond', None)

        procs = []
        for i in range(world_size):
            n_i = n_per_gpu[i]

            x_per_gpu = {
                'y': y[sum(n_per_gpu[:i]):sum(n_per_gpu[:i])+n_i] if y is not None else None,
                'cond': cond[sum(n_per_gpu[:i]):sum(n_per_gpu[:i])+n_i] if cond is not None else None
            }

            if n_i == 0:
                continue
            p = mp.Process(target=self._sample_worker, args=(i, world_size, n_i, self.model.diffusion.module, x_per_gpu, path_sampling))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        if os.path.exists(path_sampling):
            merged = torch.load(path_sampling, map_location="cpu", weights_only=False)
            print(merged.dtype)
            print(merged.shape)
            return merged
        else:
            raise FileNotFoundError("Merged sampling file not found.")
    

            





                


