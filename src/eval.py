import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from data import TestDataset
from model import DDPM
from trainer import Trainer
from config import parse_args



def eval(test_data: dict):
##############################################################################
# Loading configuration
##############################################################################
    opt = parse_args()

##############################################################################
# Loading trained diffusion model
##############################################################################
    model = DDPM(opt)
    trainer = Trainer(model, opt)

##############################################################################
# Generate samples 
##############################################################################
    outputs = trainer.sampling(test_data, n_total=test_data['num_samples'], path_sampling=opt['path']['path_sampling'])

    return outputs


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"


    mp.set_start_method("spawn", force=True)

    test_data = {
        'num_samples': 128
    }
    outputs = eval(test_data)
    np.save('generated_samples.npy', outputs)