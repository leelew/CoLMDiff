import math
from functools import partial

import tqdm
import numpy as np
import torch
import torch.nn as nn



class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process supporting both unconditional and conditional
    denoising models, with customizable noise schedules and sampling solvers.

    Args:
        denoiser (nn.Module):
            The neural network used to predict either noise ε, the clean image x0,
            depending on the training objective. Typically a UNet.

        timesteps (int):
            Total number of diffusion steps T. Controls the granularity of the
            forward q(x_t | x_{t-1}) and reverse p(x_{t-1} | x_t) process.
            A common choice is 1000 steps.

        objective (str):
            The training target of the diffusion model:
                - "pred_noise":     model predicts the added noise ε_t
                - "pred_x0":        model predicts the clean signal x0
            Determines how x0 and noise are reconstructed during training and sampling.

        conditional (bool):
            Whether the diffusion model is conditioned on external information.
            If True, the denoiser receives concatenated inputs (e.g., concatenation
            of x_t with a low-resolution or masked input). If False, the model
            performs unconditional generation.

        schedule_opt (dict):
            Configuration dictionary describing the forward noise schedule,
            such as:
                {
                    "schedule": "cosine" | "linear" | "warmup",
                    "beta_start": ...,
                    "beta_end": ...,
                    "warmup_frac": ...
                }
            These hyperparameters define β_t, α_t = 1-β_t, and ᾱ_t,
            which govern the forward diffusion q(x_t | x_0).

        solver (str):
            The sampling algorithm used for reverse diffusion, such as:
                - "base":        ancestral DDPM sampling
                - "ddim":        deterministic DDIM sampling
                - "dpm_solver":  high-order fast ODE-based solvers
            This determines how p(x_{t-1}|x_t) is approximated at inference time.
    """
    def __init__(self,
                 denoiser: nn.Module,
                 timesteps: int,
                 objective: str,
                 conditional: bool,
                 solver: str):
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = timesteps
        self.objective = objective
        self.conditional = conditional  
        self.solver = solver

##############################################################################
# Set up noise schedule
##############################################################################
    def _warmup_beta(self, 
                     linear_start, 
                     linear_end, 
                     n_timestep, 
                     warmup_frac):
        """Create a warmup beta schedule"""
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        warmup_time = int(n_timestep * warmup_frac)
        betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        return betas

    def _make_beta_schedule(self,
                            schedule, 
                            n_timestep, 
                            linear_start=1e-4, 
                            linear_end=2e-2, 
                            cosine_s=8e-3):
        if schedule == 'quad':
            betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
        elif schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup10':
            betas = self._warmup_beta(linear_start, linear_end, n_timestep, 0.1)
        elif schedule == 'warmup50':
            betas = self._warmup_beta(linear_start, linear_end, n_timestep, 0.5)
        elif schedule == 'const':
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
        elif schedule == "cosine":
            timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_noise_schedule(self, schedule_opt, device):
        """Set noise schedule for diffusion process"""
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = self._make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=self.num_timesteps,
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_o)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # Note:
        # The posterior variance (β̃_t) is written in the numerically stable form:
        #
        #     β̃_t = 1 / (1 / (1 - ᾱ_{t-1}) + α_t / β_t)
        #
        # This expression is mathematically equivalent to the standard DDPM formula:
        #
        #     β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        #
        # but avoids division by very small values when t is close to 0, where
        # (1 - ᾱ_t) → 0. The stable form prevents NaNs and negative variances,
        # ensuring robust computation of q(x_{t-1} | x_t, x_0) across all timesteps.  
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)      
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # We take the log of the posterior variance, but β̃_t = 0 at t = 0,
        # which would produce log(0) = -inf and break numerical stability.
        # To avoid NaNs and -inf, we clip β̃_t to a minimum value (1e-20)
        # before taking the logarithm. This keeps the reverse diffusion
        # process stable while preserving the correct variance elsewhere.
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))

        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

##############################################################################
# diffusion forward process (q)
##############################################################################
    def _extract_into_tensor(self, a, t, x_shape):
        """Extract values from a 1-D tensor for a batch of indices."""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_mean_variance(self, x_start, t):
        """Get the distribution q(x_t | x_0) before T"""
        pass

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

##############################################################################
# reverse diffusion process (p) & sampling
##############################################################################
    def q_posterior(self, x_start, x_t, t):
        """Get the distribution q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def _predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, x_t, t, x_condition, clip_denoised=True):
        """Get the distribution p(x_{t-1} | x_t), equally q(x_{t-1} | x_t, x_0) with x_0 predicted by model"""
        if self.conditional:
            model_output = self.denoiser(torch.cat([x_condition, x_t], dim=1), t)
        else:
            model_output = self.denoiser(x_t, t)

        if self.objective == "pred_noise":
            x_start_pred = self._predict_start_from_noise(x_t, t=t, noise=model_output)
        elif self.objective == "pred_x0":
            x_start_pred = model_output
        if clip_denoised:
            x_start_pred.clamp_(-5., 5.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start_pred, x_t=x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def _make_noise(self, shape, device, repeat=False):
        def repeat_noise(): return torch.randn(
            (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

        def noise(): return torch.randn(shape, device=device)
        return repeat_noise() if repeat else noise()

    @torch.no_grad()
    def p_sample(self, x_t, t, x_condition, clip_denoised=True):
        """Sample one step from p(x_{t-1} | x_t)"""
        b, *_, device = *x_t.shape, x_t.device
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t, x_condition, clip_denoised)
        noise = self._make_noise(x_t.shape, device, repeat_noise=False)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise    
    
    @torch.no_grad()
    def p_sample_loop(self, x, continuous=False):
        """Generate samples from p(x_0) by iteratively applying p(x_{t-1} | x_t) from Gaussian noise"""
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))

        shape = (x['x_start'].size(0), 1, x['x_start'].size(2), x['x_start'].size(3))
        noise = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling timestep', total=self.num_timesteps):
            img = self.p_sample(noise, 
                                torch.full((shape[0],), i, device=device, dtype=torch.long), # time step t
                                x['x_condition'] if self.conditional else None, 
                                clip_denoised=True)
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
        
        if continuous:
            return ret_img
        else:
            return img
        
    @torch.no_grad()
    def sample(self, x, continuous=False):
        """Generate samples from p(x_0) using different sampling solvers"""
        if self.solver == 'ddim':
            pass
        elif self.solver == 'dpm_solver':
            pass
        elif self.solver == 'base':
            return self.p_sample_loop(x, continuous)

##############################################################################
# Calculate loss function for training
##############################################################################
    def p_losses(self, x, t, noise=None):
        """Compute the loss function for diffusion model"""
        # diffusion process q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x['x_start'])
        x_t = self.q_sample(x_start=x['x_start'], t=t, noise=noise)

        # model prediction
        if self.conditional:
            model_output = self.denoiser(torch.cat([x['x_condition'], x_t], dim=1), t)
        else:
            model_output = self.denoiser(x_t, t)
        
        # calculate loss
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x['x_start']
        loss_func = nn.MSELoss(reduction='mean').to(x['x_start'].device)
        loss = loss_func(target, model_output)
        return loss

    def forward(self, x): 
        """forward process for training"""
        t = torch.randint(0, self.num_timesteps, (x['x_start'].size(0),), device=x['x_start'].device).long()
        return self.p_losses(x, t)

        
    


