import os
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from .base import BaseModel
from .denoiser import UNet, LinearNet
from .diffusion import GaussianDiffusion



class DDPM(BaseModel):
    """
    A high-level wrapper for training and evaluating Denoising Diffusion
    Probabilistic Models (DDPM). This class integrates the denoising network,
    diffusion process, optimization utilities, noise scheduling, checkpoint
    management, and training/testing basic routines into a unified interface.

    This module constructs a complete DDPM system consisting of:
      • A U-Net denoiser predicting ε or x₀ depending on the chosen objective.
      • A GaussianDiffusion module implementing forward diffusion
        q(x_t | x_{t-1}) / q(x_t | x_0) and learned backward process
        p_θ(x_{t-1} | x_t), including sampling logic.
      • Training utilities (optimizers, schedulers, gradient clipping).
      • Noise schedule configuration for both training and inference phases.
      • Checkpoint saving/loading and network summarization.

    The class supports both single-GPU and multi-GPU DataParallel execution.
    If multiple GPU IDs are provided, the diffusion module is wrapped in
    torch.nn.DataParallel. All components are moved to the active device using
    set_device().
    """
    def __init__(self, opt):
        super().__init__(opt)

##############################################################################
# Define model architecture by options
##############################################################################
        # define denoiser network
        if opt['model']['2d']:
            self.denoiser = UNet(
                img_size=opt['model']['height_2d'],
                in_channel=opt['model']['in_channel_2d'],
                out_channel=opt['model']['out_channel_2d'],
                embed_dim=opt['model']['embed_dim'], 
                norm_groups=opt['model']['groups'],
                dropout=opt['model']['dropout'],
                channel_mults=tuple(opt['model']['channel_mults']),
                num_blocks=opt['model']['num_blocks']
            )
        elif opt['model']['1d']:
            self.denoiser = LinearNet(
                in_channel=opt['model']['in_channel_1d'],
                embed_dim=opt['model']['embed_dim'] 
            )
    
        # define diffusion model
        self.diffusion = GaussianDiffusion(
            denoiser=self.denoiser,
            opt=opt
        )

##############################################################################
# Initialized model weights during training phase
##############################################################################
        # initialize weighted of diffusion model
        if opt['phase'] == 'train':
            self.init_weights(self.diffusion, init_type=opt['train']['init_method'])

##############################################################################
# Move model to device and handle multi-GPU if needed
##############################################################################
        assert torch.cuda.is_available()
        if len(opt['gpu_ids']) > 1: 
            self.diffusion = nn.DataParallel(self.diffusion,
                                             device_ids=opt['gpu_ids'],           
                                             output_device=opt['gpu_ids'][0])   
        self.diffusion = self.set_device(self.diffusion)

##############################################################################
# Setup noise schedule for both training or validation phase (?)
##############################################################################
        if opt['phase'] == 'train':
            self.set_diffusion_schedule(opt['train']['noise_schedule']) 

##############################################################################
# Define optimizers and learning schedulers for training phase
##############################################################################
        if opt['phase'] == 'train':
            self.set_optimizer()
            self.set_learning_schedule()

##############################################################################
# Setup paths for checkpoints and results
##############################################################################
        self.path_ckp = opt['path']['path_ckp']

##############################################################################
# Define logging dictionary
##############################################################################
        if opt['phase'] == 'train':
            self.log_dict = OrderedDict()

##############################################################################
# print model information for diagnosis
##############################################################################
        self.print_network()

##############################################################################
# Initialized model weights functions
##############################################################################
    def _weights_init_normal(self, model: nn.Module, std: float = 0.02) -> None:
        """Initializes model weights from Gaussian distribution."""
        classname = model.__class__.__name__
        if classname in ['LinearNet', 'UNet', 'GaussianDiffusion', 'Sequential', 
                        'ModuleList', 'ModuleDict', 'DataParallel']:
            return
        if classname.find("Conv") != -1:
            nn.init.normal_(model.weight.data, 0.0, std)
            if model.bias is not None:
                model.bias.data.zero_()
        elif classname.find("Linear") != -1:
            nn.init.normal_(model.weight.data, 0.0, std)
            if model.bias is not None:
                model.bias.data.zero_()
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(model.weight.data, 1.0, std)
            nn.init.constant_(model.bias.data, 0.0)

    def _weights_init_kaiming(self, model: nn.Module, scale: float = 1) -> None:
        """He initialization of model weights."""
        classname = model.__class__.__name__
        if classname in ['LinearNet', 'UNet', 'GaussianDiffusion', 'Sequential', 
                        'ModuleList', 'ModuleDict', 'DataParallel']:
            return
        if classname.find("Conv2d") != -1:
            nn.init.kaiming_normal_(model.weight.data)
            model.weight.data *= scale
            if model.bias is not None:
                model.bias.data.zero_()
        elif classname.find("Linear") != -1:
            nn.init.kaiming_normal_(model.weight.data)
            model.weight.data *= scale
            if model.bias is not None:
                model.bias.data.zero_()
        elif classname.find("BatchNorm2d") != -1:
            nn.init.constant_(model.weight.data, 1.0)
            nn.init.constant_(model.bias.data, 0.0)

    def _weights_init_orthogonal(self, model: nn.Module) -> None:
        """Fills the model weights to be orthogonal matrices."""
        classname = model.__class__.__name__
        if classname in ['LinearNet', 'UNet', 'GaussianDiffusion', 'Sequential', 
                        'ModuleList', 'ModuleDict', 'DataParallel']:
            return
        if classname.find("Conv") != -1:
            nn.init.orthogonal_(model.weight.data)
            if model.bias is not None:
                model.bias.data.zero_()
        elif classname.find("Linear") != -1:
            nn.init.orthogonal_(model.weight.data)
            if model.bias is not None:
                model.bias.data.zero_()
        elif classname.find("BatchNorm2d") != -1:
            nn.init.constant_(model.weight.data, 1.0)
            nn.init.constant_(model.bias.data, 0.0)

    def init_weights(self, net: nn.Module, init_type: str = "kaiming", scale: float = 1, std: float = 0.02) -> None:
        """Initializes network weights."""
        if init_type == "normal":
            weights_init_normal_ = functools.partial(self._weights_init_normal, std=std)
            net.apply(weights_init_normal_)
        elif init_type == "kaiming":
            weights_init_kaiming_ = functools.partial(self._weights_init_kaiming, scale=scale)
            net.apply(weights_init_kaiming_)
        elif init_type == "orthogonal":
            net.apply(self._weights_init_orthogonal)
        else:
            raise NotImplementedError("Initialization method [{:s}] not implemented".format(init_type))

##############################################################################
# Setup noise and learning schedule functions
##############################################################################
    def set_diffusion_schedule(self, schedule_opt):
        """Setup noise schedule for diffusion process"""
        if isinstance(self.diffusion, nn.DataParallel):
            self.diffusion.module.set_noise_schedule(
                schedule_opt=schedule_opt,
                device=self.device
            )
        else:
            self.diffusion.set_noise_schedule(
                schedule_opt=schedule_opt,
                device=self.device
            )

    def set_optimizer(self):
        """Setup optimizer for training"""
        train_opt = self.opt['train']
        optim_params = []
        
        if isinstance(self.diffusion, nn.DataParallel):
            network = self.diffusion.module
        else:
            network = self.diffusion

        for k, v in network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        
        if train_opt['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                optim_params,
                lr=train_opt['lr'],
                betas=(train_opt.get('beta1', 0.9), train_opt.get('beta2', 0.999)),
                weight_decay=train_opt.get('weight_decay', 0)
            )
        elif train_opt['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                optim_params,
                lr=train_opt['lr'],
                betas=(train_opt.get('beta1', 0.9), train_opt.get('beta2', 0.999)),
                weight_decay=train_opt.get('weight_decay', 0.01)
            )
        else:
            raise NotImplementedError(f'Optimizer {train_opt["optimizer"]} not implemented')
    
    def set_learning_schedule(self):
        """Setup learning rate scheduler"""
        if self.opt['train']['lr_schedule'] == 'MultiStepLR':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[20000],
                gamma=0.5
            )
        elif self.opt['train']['lr_schedule'] == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_0=100000,
                eta_min=1e-6
            )
        else:
            self.scheduler = None

##############################################################################
# Basic functions for training and testing
##############################################################################
    def load_one_sample(self, data):
        """Load one sample training data into device"""
        self.data = self.set_device(data)

    def train_one_sample(self):
        """Train the model using one sample"""
        self.optimizer.zero_grad()

        # forward pass
        loss = self.diffusion(self.data)
        if loss.dim() > 0:
            loss = loss.mean()

        # backward pass
        loss.backward()
        
        # gradient clipping 
        if self.opt['train']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.diffusion.parameters(),
                self.opt['train']['grad_clip']
            )
        
        # optimizer step
        self.optimizer.step()
        
        # log loss
        self.log_dict['loss'] = loss.item()

    def print_network(self) -> None:
        """Prints the network architecture using torchinfo."""
        # Get the actual model (handle DataParallel)
        network = self.diffusion.module if isinstance(self.diffusion, nn.DataParallel) else self.diffusion
        
        # Create dummy input based on model type
        if self.opt['model']['2d']:
            x = {
                'x0': torch.randn(1, self.opt['model']['in_channel_2d'], 
                                self.opt['model']['height_2d'], 
                                self.opt['model']['width_2d']).to(self.device),
            }
            if self.opt['model']['conditional']:
                x['cond'] = torch.randn(1, self.opt['model']['in_channel_2d'], 
                                    self.opt['model']['height_2d'], 
                                    self.opt['model']['width_2d']).to(self.device)
        elif self.opt['model']['1d']:
            x = {
                'x0': torch.randn(1, self.opt['model']['in_channel_1d']).to(self.device),
            }
            if self.opt['model']['conditional']:
                x['cond'] = torch.randn(1, self.opt['model']['in_channel_1d']).to(self.device)
        
        print("="*50)
        if isinstance(self.diffusion, nn.DataParallel):
            print(f"Model: DataParallel({network.__class__.__name__})")
            print(f"GPUs: {len(self.opt['gpu_ids'])}")
        else:
            print(f"Model: {network.__class__.__name__}")
        print("="*50)
        
        # Print detailed summary
        summary(
            network,
            input_data=(x,),
            depth=4,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            row_settings=["var_names"],
            verbose=1
        )

    def save_network(self, epoch: int) -> None:
        """Saves the network checkpoint."""
        network = self.diffusion.module if isinstance(self.diffusion, nn.DataParallel) else self.diffusion

        state_dict = network.state_dict() 
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        checkpoint = {
            "epoch": epoch,
            "model": state_dict,
            "lr_schedule": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

        path_save = os.path.join(self.path_ckp, f"checkpoint_E{epoch}.pth")
        torch.save(checkpoint, path_save)
        print(f"Saved model checkpoint to {path_save}")

    def load_network(self) -> None:
        """Load network parameters from checkpoint."""
        if self.opt['phase'] == 'sample':
            network = self.diffusion.module if isinstance(self.diffusion, nn.DataParallel) else self.diffusion

            if os.path.isdir(self.path_ckp):
                checkpoint_files = [f for f in os.listdir(self.path_ckp) if f.endswith('.pth')]
                if not checkpoint_files:
                    raise FileNotFoundError(f"No checkpoint files found in {self.path_ckp}")
                
                checkpoint_files.sort(key=lambda x: int(x.split('_E')[-1].split('.')[0]))
                checkpoint_path = os.path.join(self.path_ckp, checkpoint_files[-1])
                print(f"Loading checkpoint: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            network.load_state_dict(checkpoint['model'], strict=True)
            print("Checkpoint loaded successfully")