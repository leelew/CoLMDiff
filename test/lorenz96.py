import os
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
np.random.seed(12362186)

import torch
torch.manual_seed(123456987)

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import matplotlib.pyplot as plt

from src.data import TrainDataset
from src.model import DDPM
from src.trainer import Trainer
from src.config import parse_args



# ============================================================================
# Lorenz-96 Model for Data Assimilation
# ============================================================================
def lorenz96(x, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i+1) % N] - x[(i-2)%N]) * x[(i-1)%N] - x[i] + F
    return dxdt


def rk4_step(x, dt, F=8.0, sigma_SDE=0.0):
    k1 = lorenz96(x, F)
    k2 = lorenz96(x + 0.5 * dt * k1, F)
    k3 = lorenz96(x + 0.5 * dt * k2, F)
    k4 = lorenz96(x + dt * k3, F)
    x_new = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    if sigma_SDE > 0:
        noise = sigma_SDE * np.sqrt(dt) * np.random.randn(len(x))
        x_new = x_new + noise

    return x_new


def operator(x_true):
    y = x_true**3
    return y


def init_x0(num_dim):
    x0_true = np.random.randn(num_dim) * 2 + 4
    return x0_true


def get_true_traj(num_dim, F=8, dt=0.01, total_steps=1000, spinup_steps=100, sigma_obs=0.5):
    x0_true = init_x0(num_dim)
    x_true = [x0_true]
    for i in range(total_steps):
        x0_true = rk4_step(x0_true, dt, F)

        if i == spinup_steps + 30:
            x0_true = x0_true + np.random.randn(num_dim)*3
        if i == spinup_steps + 60:
            x0_true = x0_true + np.random.randn(num_dim)*3

        x_true.append(x0_true)
    x_true = np.stack(x_true, axis=0)[spinup_steps:, :]
    y_noise = operator(x_true) + np.random.randn(*(x_true.shape)) * sigma_obs

    return x_true, y_noise


# ============================================================================
# EnKF
# ============================================================================
def EnKF(num_dim, 
         steps, 
         sample_size, 
         x0_noise, 
         y_noise, 
         dt=0.01, 
         F=8, 
         sigma_SDE=0.0, 
         sigma_obs=0.5):
    """
    Ensemble Kalman Filter for Lorenz96
    
    Args:
        num_dim: State dimension
        steps: Number of assimilation steps
        sample_size: Ensemble size
        x0_noise: Initial ensemble [sample_size, num_dim]
        y_noise: Observations [steps+1, num_dim] (includes initial observation)
        dt: Time step
        F: Forcing parameter
        sigma_SDE: Model error std
        sigma_obs: Observation error std
    
    Returns:
        xa: Analysis ensemble mean [steps, num_dim]
    """
    R = (sigma_obs**2) * np.eye(num_dim)

    xa = []
    xf_ens = x0_noise.copy()

    for n in range(steps):
        yf_ens = operator(xf_ens) 
        xf_mean = np.nanmean(xf_ens, axis=0)
        yf_mean = np.nanmean(yf_ens, axis=0)

        dxf = xf_ens - xf_mean[None, :]  
        dyf = yf_ens - yf_mean[None, :]  
        pf_xy = (np.transpose(dxf, (1, 0)) @ dyf) / (sample_size - 1)
        pf_yy = (np.transpose(dyf, (1, 0)) @ dyf) / (sample_size - 1) + R

        K = pf_xy @ np.linalg.inv(pf_yy)  

        y_noise_pert = y_noise[n, :][None, :] + sigma_obs * np.random.randn(sample_size, num_dim)  
        innov = y_noise_pert - yf_ens  

        xa_ens = xf_ens + (innov @ np.transpose(K, (1, 0)))  
        xa_mean = np.nanmean(xa_ens, axis=0)
        xa.append(xa_mean) 

        if n < steps - 1:
            xf_ens = np.zeros_like(xa_ens)
            for i in range(sample_size):
                xf_ens[i,:] = rk4_step(xa_ens[i,:], dt=dt, F=F, sigma_SDE=sigma_SDE)

    return np.stack(xa, axis=0) 


# ============================================================================
# Diffusion Filtering
# ============================================================================
def EnDF(opt,
         steps, 
         sample_size, 
         x0_noise, 
         y_noise, 
         dt=0.01, 
         F=8, 
         sigma_SDE=0.0):

    # initialize diffusion model and trainer    
    model = DDPM(opt)
    trainer = Trainer(model, opt)

    # pre-train diffusion model on p(x0)
    train_data = {'x0': torch.from_numpy(x0_noise).float()}
    trainer.train(train_data)

    # analysis step with diffusion model
    xa = []
    xf_ens = x0_noise.copy()

    for n in range(steps):
        # prepare observation
        x = {'y': torch.from_numpy(y_noise[n, :]).float().unsqueeze(0).repeat(sample_size, 1)}

        # sampling from p(x_t|[y_t])
        xa_samples = trainer.sampling(x, sample_size, operator)
        xa_mean = np.nanmean(xa_samples, axis=0)
        xa.append(xa_mean)

        # prediction step p(x_t+1|x_t,[y_t])
        xf_ens = np.zeros_like(xa_samples)
        for i in range(sample_size):
            xf_ens[i,:] = rk4_step(xa_samples[i,:], dt=dt, F=F, sigma_SDE=sigma_SDE) 

        # analysis step p(x_t+1|[y_t+1])
        finetune_data = {'x0': torch.from_numpy(xa_samples).float()}
        if n < steps - 1:
            trainer.finetune(finetune_data, finetune_epochs=opt['train']['epochs'])

    return np.stack(xa, axis=0)


# ============================================================================
# main program
# ============================================================================
def app(num_dim, sample_size, dt, steps, spinup_steps=100, F=8, sigma_SDE=0.0, sigma_x0=0.5, sigma_obs=0.5):
    # Generate true trajectory and noisy observations
    x_true, y_noise = get_true_traj(num_dim, F, dt, steps+spinup_steps, spinup_steps, sigma_obs)

    # Initialize ensemble with noisy initial conditions
    x0_noise = x_true[0:1, :] + np.random.randn(sample_size, num_dim) * sigma_x0

    # Run EnKF
    x_enkf = EnKF(num_dim, steps, sample_size, x0_noise, y_noise, dt, F, sigma_SDE, sigma_obs)

    # Run Diffusion Filtering
    x_diff = EnDF(parse_args(), steps, sample_size, x0_noise, y_noise, dt, F, sigma_SDE)

    return x_true, x_enkf, x_diff




if __name__ == "__main__":
    x_true, x_enkf, x_diff = app(num_dim=10, sample_size=1000, dt=0.01, steps=100)
    print(x_true.shape, x_enkf.shape, x_diff.shape)

    plt.figure()
    plt.plot(np.nanmean(x_true, axis=1), label='True')
    plt.plot(np.nanmean(x_enkf, axis=1), label='EnKF')
    plt.plot(np.nanmean(x_diff, axis=1), label='EnDF')
    plt.legend()
    plt.savefig('lorenz96.png')
