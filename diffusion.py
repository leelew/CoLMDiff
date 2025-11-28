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
                - "ddpm":        ancestral DDPM sampling
                - "ddim":        deterministic DDIM sampling
                - "dpm_solver":  high-order fast ODE-based solvers
            This determines how p(x_{t-1}|x_t) is approximated at inference time.
    """
    def __init__(self,
                 denoiser: nn.Module,
                 opt: dict):
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = opt['model']['timesteps']
        self.objective = opt['model']['objective']
        self.conditional = opt['model']['conditional']  

        # for fast sampling
        self.solver = opt['sample']['solver']
        if self.solver == 'ddim':
            self.ddim_sample_steps = opt['sample']['ddim']['ddim_sample_steps']
            self.ddim_eta = opt['sample']['ddim']['ddim_eta']

        # for guidance sampling
        self.sample_method = opt['sample']['method']
        if self.sample_method == 'p(x|y):sdedit':
            self.start_t = opt['sample']['sdedit']['sdedit_start_t']
        else:
            self.start_t = self.num_timesteps - 1

        self.channels = opt['model']['in_channel']
        self.height = opt['model']['height']
        self.width = opt['model']['height'] 

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

        # DDIM parameters
        if self.solver == 'ddim':
            c = self.start_t // self.ddim_sample_steps
            self.ddim_timesteps = np.asarray(list(range(0, self.start_t, c)))
    
            ddim_alphas = self.alphas_cumprod.cpu()[self.ddim_timesteps]
            ddim_alphas_prev = np.asarray([alphas_cumprod[0]] + alphas_cumprod[self.ddim_timesteps[:-1]].tolist())
            ddim_sigmas = self.ddim_eta * np.sqrt((1 - ddim_alphas_prev) / (1 - ddim_alphas) * (1 - ddim_alphas / ddim_alphas_prev))

            self.register_buffer('ddim_sigmas', ddim_sigmas)
            self.register_buffer('ddim_alphas', ddim_alphas)
            self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
            self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
            sigmas_for_original_sampling_steps = self.ddim_eta * torch.sqrt(
                (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                            1 - self.alphas_cumprod / self.alphas_cumprod_prev))
            self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

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
    
    def _prepare_data_for_sampling(self, x, num_samples, device):
        """Prepare input data dict for sampling"""
        img_shape = (num_samples, self.channels, self.height, self.width)

        data_for_sampling = {}
        if self.sample_method == 'p(x)':
            x_t = torch.randn(*img_shape, device=device)
            cond = None
        elif self.sample_method == 'p(x|y):cond':
            x_t = torch.randn(*img_shape, device=device)
            cond = x['cond'].to(device)
        elif self.sample_method == 'p(x|y):sdedit':
            x_t = self.q_sample(x_start=x['y'].to(device), t=torch.full((1,), self.start_t, device=device, dtype=torch.long))
            cond = None
        elif self.sample_method == 'p(x|y):dps':   
            x_t = torch.randn(*img_shape, device=device)
            cond = None

        data_for_sampling['x_t'] = x_t
        data_for_sampling['cond'] = cond
        return data_for_sampling

    @torch.no_grad()
    def p_sample_ddpm(self, x_t, t, x_condition, clip_denoised=True):
        """Sample one step from p(x_{t-1} | x_t)"""
        b, *_, device = *x_t.shape, x_t.device
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t, x_condition, clip_denoised)
        noise = self._make_noise(x_t.shape, device, repeat=False)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_ddim(self, x_t, t, t_prev, x_condition, clip_denoised=True, eta=0.0):
        """
        DDIM sampling: one step from x_t to x_{t_prev}
        
        DDIM generalizes DDPM by introducing a parameter η (eta) that controls
        the stochasticity of the sampling process:
        - η = 0: deterministic sampling (pure DDIM)
        - η = 1: equivalent to DDPM sampling
        
        The DDIM update rule is:
        x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}-σ_t²) · ε_θ(x_t,t) + σ_t · ε
        
        where:
        - x̂_0 is the predicted clean sample
        - ε_θ(x_t,t) is the predicted noise
        - σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-1})
        - ε ~ N(0, I) is random noise
        
        Args:
            x_t: current sample at timestep t
            t: current timestep
            t_prev: previous timestep (t-1 or smaller for skipping steps)
            x_condition: conditional input (if conditional=True)
            clip_denoised: whether to clip predicted x_0 to [-5, 5]
            eta: controls stochasticity (0=deterministic, 1=stochastic like DDPM)
        
        Returns:
            x_{t_prev}: sample at timestep t_prev
        """
        # Get model prediction
        if self.conditional:
            model_output = self.denoiser(torch.cat([x_condition, x_t], dim=1), t)
        else:
            model_output = self.denoiser(x_t, t)
        
        # Predict x_0 from model output
        if self.objective == "pred_noise":
            pred_noise = model_output
            x_0_pred = self._predict_start_from_noise(x_t, t, pred_noise)
        elif self.objective == "pred_x0":
            x_0_pred = model_output
            # Predict noise from x_0: ε = (x_t - √ᾱ_t · x_0) / √(1-ᾱ_t)
            pred_noise = (
                self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * 
                (x_t - self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_0_pred)
            )
        
        # Clip predicted x_0 if needed
        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-5., 5.)
        
        # Extract alpha values
        alpha_t = self._extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        alpha_t_prev = self._extract_into_tensor(self.alphas_cumprod, t_prev, x_t.shape)
        
        # Compute sigma for the stochastic term
        # σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-1})
        sigma_t = (
            eta * 
            torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * 
            torch.sqrt(1 - alpha_t / alpha_t_prev)
        )
        
        # Compute the direction pointing to x_t
        # √(1-ᾱ_{t-1}-σ_t²) · ε_θ(x_t,t)
        dir_xt_coef = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2)
        dir_xt = dir_xt_coef * pred_noise
        
        # DDIM update: x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}-σ_t²) · ε_θ + σ_t · ε
        x_t_prev_mean = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt
        
        # Add stochastic term if eta > 0
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev_mean + sigma_t * noise
        else:
            x_t_prev = x_t_prev_mean
        
        return x_t_prev    
    
    @torch.no_grad()
    def p_sample_ddpm_loop(self, x, continuous=False, device=None):
        """Generate samples from p(x_0) by iteratively applying p(x_{t-1} | x_t) from Gaussian noise"""
        sample_inter = (1 | (self.num_timesteps // 10))

        shape = x['x_t'].shape
        img = x['x_t'] #torch.randn(shape, device=device)
        ret_img = img if continuous else None

        for i in tqdm.tqdm(reversed(range(0, self.start_t+1)), desc='DDPM sampling', total=self.start_t):
            img = self.p_sample_ddpm(img, 
                                     torch.full((shape[0],), i, device=device, dtype=torch.long), # time step t
                                     x['cond'] if self.conditional else None, 
                                     clip_denoised=True)
            if continuous and i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
        
        if continuous:
            return ret_img
        else:
            return img
    
    @torch.no_grad()
    def p_sample_ddim_loop(self, x, continuous=False, device=None):
        """
        DDIM sampling loop with accelerated sampling using fewer timesteps.
        
        DDIM allows skipping timesteps, enabling faster sampling than DDPM.
        For example, with 1000 training steps, we can sample with only 50 steps.
        
        Args:
            x: dict containing 'x_start' for shape and optionally 'x_condition'
            ddim_timesteps: number of sampling steps (can be < num_timesteps for acceleration)
            ddim_eta: controls stochasticity (0=deterministic, 1=stochastic)
            continuous: whether to return intermediate samples
        
        Returns:
            Generated sample(s)
        """
        batch_size = x['x_t'].size(0)
        
        # Start from random noise
        #shape = (batch_size, 1, x['x_t'].size(2), x['x_t'].size(3))
        img = x['x_t'] #torch.randn(shape, device=device)
        
        ddim_timesteps = self.ddim_timesteps.tolist() if isinstance(self.ddim_timesteps, np.ndarray) else self.ddim_timesteps

        # For continuous output
        sample_inter = max(1, len(ddim_timesteps) // 10)
        ret_img = img if continuous else None
        
        # DDIM sampling loop
        for i in tqdm.tqdm(reversed(range(len(ddim_timesteps))), desc='DDIM sampling', total=len(ddim_timesteps)):
            # Determine previous timestep
            t_curr = ddim_timesteps[i]
            t_prev = ddim_timesteps[i-1] if i > 0 else 0
            
            # Create timestep tensors
            t_curr_tensor = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)
            t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
            
            # DDIM sampling step
            img = self.p_sample_ddim(
                img,
                t_curr_tensor,
                t_prev_tensor,
                x['cond'] if self.conditional else None,
                clip_denoised=True,
                eta=self.ddim_eta
            )
            
            # Store intermediate results if continuous
            if continuous and i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
        
        if continuous:
            return ret_img
        else:
            return img
        
    @torch.no_grad()
    def sample(self, x, num_samples, continuous=False, device=None):
        """
        Generate samples from p(x_0) using different sampling solvers.
        
        Args:
            x: input dict with 'x_start' and optionally 'x_condition'
            continuous: whether to return intermediate samples
            ddim_timesteps: number of sampling steps for DDIM (only used when solver='ddim')
            ddim_eta: stochasticity parameter for DDIM (0=deterministic, 1=stochastic)
        
        Returns:
            Generated sample(s)
        """
        x = self._prepare_data_for_sampling(x, num_samples, device=device)
        if self.solver == 'ddim':
            return self.p_sample_ddim_loop(x, continuous, device=device)
        elif self.solver == 'ddpm': 
            return self.p_sample_ddpm_loop(x, continuous, device=device)

##############################################################################
# Calculate loss function for training
##############################################################################
    def p_losses(self, x, t, noise=None):
        """Compute the loss function for diffusion model"""
        # diffusion process q(x_t | x_0)
        if noise is None:
            x_start = x['x0'].to(torch.float32)
            noise = torch.randn(x_start.shape, device=x_start.device, dtype=x_start.dtype)
        x_t = self.q_sample(x_start=x['x0'], t=t, noise=noise)

        # model prediction
        if self.conditional:
            model_output = self.denoiser(torch.cat([x['cond'], x_t], dim=1), t)
        else:
            model_output = self.denoiser(x_t, t)
        
        # calculate loss
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x['x0']
        loss = nn.functional.mse_loss(target, model_output, reduction='mean')
        return loss

    def forward(self, x): 
        """forward process for training"""
        t = torch.randint(0, self.num_timesteps, (x['x0'].size(0),), device=x['x0'].device).long()
        return self.p_losses(x, t)

        
    


