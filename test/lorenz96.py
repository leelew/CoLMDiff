import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(123456987)
np.random.seed(12362186)



# ============================================================================
# Lorenz-96 Model 
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


def observation_operator(x_true):
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
    y_noise = observation_operator(x_true) + np.random.randn(*(x_true.shape)) * sigma_obs

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
    R = (sigma_obs**2) * np.eye(num_dim)

    xa = []
    xf_ens = x0_noise.copy()
    for n in range(steps):
        # prediction step
        for i in range(sample_size):
            xf_ens[i,:] = rk4_step(xf_ens[i,:], dt=dt, F=F, sigma_SDE=sigma_SDE) 
        yf_ens = observation_operator(xf_ens)  
        xf_mean = np.nanmean(xf_ens, axis=0)
        yf_mean = np.nanmean(yf_ens, axis=0)

        # covariance matrix
        dxf = xf_ens - xf_mean[None, :]  
        dyf = yf_ens - yf_mean[None, :]  
        pf_xy = (np.transpose(dxf, (1, 0)) @ dyf) / (sample_size - 1)
        pf_yy = (np.transpose(dyf, (1, 0)) @ dyf) / (sample_size - 1) + R

        # Kalman gain
        K = pf_xy @ np.linalg.inv(pf_yy)  

        # innovation
        y_noise_pert = y_noise[n+1, :][None, :] + sigma_obs * np.random.randn(sample_size, num_dim)  
        innov = y_noise_pert - yf_ens  

        # analysis step
        xa_ens = xf_ens + (innov @ np.transpose(K, (1, 0)))  
        xa_mean = np.nanmean(xa_ens, axis=0)
        xa.append(xa_mean) 
        
    return np.stack(xa, axis=0) 


def app(num_dim, sample_size, dt, steps, spinup_steps=100, F=8, sigma_SDE=0.0, sigma_x0=0.5, sigma_obs=0.5):
    # Generate true trajectory and noisy observations
    x_true, y_noise = get_true_traj(num_dim, F, dt, steps+spinup_steps, spinup_steps, sigma_obs)

    # Initialize ensemble with noisy initial conditions
    x0_noise = x_true[0:1, :] + np.random.randn(sample_size, num_dim) * sigma_x0

    # Run EnKF
    x_enkf = EnKF(num_dim, steps, sample_size, x0_noise, y_noise, dt, F, sigma_SDE, sigma_obs)

    return x_true[1:], x_enkf



if __name__ == "__main__":
    x_true, x_enkf = app(num_dim=10, sample_size=1000, dt=0.01, steps=100)
    print(x_true.shape, x_enkf.shape)
    plt.figure()
    plt.plot(np.nanmean(x_true, axis=1), label='True')
    plt.plot(np.nanmean(x_enkf, axis=1), label='EnKF')
    plt.legend()
    plt.savefig('enkf_lorenz96.png')
