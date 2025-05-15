# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - Paper: Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from probts.model.nn.prob.diffusion_layers import DiffusionEmbedding
from functools import partial
from inspect import isfunction


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation, target_dim):
        super().__init__()
        self.target_dim = target_dim
        
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)

        if self.target_dim > 1:
            self.dilated_conv = nn.Conv1d(
                residual_channels,
                2 * residual_channels,
                3,
                padding=dilation,
                dilation=dilation,
                padding_mode="circular",
            )
            self.conditioner_projection = nn.Conv1d(
                1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
            )
        else:
            self.dilated_conv = nn.Conv1d(residual_channels,2 * residual_channels,1)
            self.conditioner_projection = nn.Conv1d(1, 2 * residual_channels, 1)

        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.target_dim = target_dim

        if self.target_dim > 1:
            self.linear1 = nn.Linear(cond_length, target_dim // 2)
            self.linear2 = nn.Linear(target_dim // 2, target_dim)
        else:
            self.linear = nn.Linear(cond_length, target_dim)

    def forward(self, x):
        if self.target_dim > 1:
            x = self.linear1(x)
            x = F.leaky_relu(x, 0.4)
            x = self.linear2(x)
            x = F.leaky_relu(x, 0.4)
        else:
            x = self.linear(x)
            x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_length,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
        padding=2
    ):
        super().__init__()
        if target_dim > 1:
            self.input_projection = nn.Conv1d(
                1, residual_channels, 1, padding=padding, padding_mode="circular"
            )
            self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
            self.output_projection = nn.Conv1d(residual_channels, 1, 3)
        else:
            # self.input_projection = nn.Identity()
            self.input_projection = nn.Conv1d(1, residual_channels, 1)
            self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 1)
            self.output_projection = nn.Conv1d(residual_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, cond_length=cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                    target_dim=target_dim,
                )
                for i in range(residual_layers)
            ]
        )

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        target_dim,
        f_hidden_size,
        conditional_length,
        beta_end=0.1,
        diff_steps=100,
        loss_type="l2",
        betas=None,
        beta_schedule="linear",
        padding=2,
        residual_channels=8,
        solver_num_steps=10,
        solver='euler',
        solver_schedule='linear'
    ):
        super().__init__()
        self.dist_args = nn.Linear(
            in_features=f_hidden_size, out_features=conditional_length
        )
        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditional_length,
            residual_channels=residual_channels,
            padding=padding,
        )
        self.target_dim = target_dim
        self.__scale = None

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4 ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.solver_num_steps = solver_num_steps
        self.solver_schedule = solver_schedule
        self.solver = solver

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, cond=cond)
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def ddpm_sample(self, shape, cond):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(
                img, cond, torch.full((b,), i, device=device, dtype=torch.long)
            )
        return img

    @torch.no_grad()
    def euler_sample(self, shape, cond, timestamps):
        device = self.betas.device
        b = shape[0]
        seq = torch.randn(shape, device=device)

        alphas = torch.flip(self.alphas_cumprod[timestamps], dims=(0,))
        betas = torch.flip(self.betas[timestamps], dims=(0,))

        for i, t in enumerate(timestamps[::-1][:-1]):
            alpha_cur = alphas[i]
            beta_cur = betas[i]
            t = torch.full((b,), t, device=device, dtype=torch.long)

            eps = self.denoise_fn(seq, t, cond=cond)
            noise = torch.randn_like(seq)

            sigma_t = torch.sqrt(1 - alpha_cur)
            velocity = - eps / sigma_t
            delta = timestamps[i + 1] - timestamps[i]

            seq += beta_cur * (seq / 2 + velocity) * delta
            seq += torch.sqrt(beta_cur * delta) * noise

        return seq

    @torch.no_grad()
    def heun_sample(self, shape, cond):
        timestamps = np.linspace(0, self.num_timesteps - 1, self.solver_num_steps)
        delta = timestamps[1] - timestamps[0]
        device = self.betas.device
        b = shape[0]
        seq = torch.randn(shape, device=device)

        alphas = torch.flip(self.alphas_cumprod[timestamps], dims=(0,))
        betas = torch.flip(self.betas[timestamps], dims=(0,))

        for i, t in enumerate(timestamps[::-1][:-1]):
            alpha_cur = alphas[i]
            beta_cur = betas[i]
            t = torch.full((b,), t, device=device, dtype=torch.long)
            eps = self.denoise_fn(seq, t, cond=cond)
            noise = torch.randn_like(seq)

            sigma_t = torch.sqrt(1 - alpha_cur)
            velocity = - eps / sigma_t
            drift = beta_cur * (seq / 2 + velocity) * delta
            diff = torch.sqrt(beta_cur * delta) * noise
            seq_hat = seq + drift + diff

            if i < self.solver_num_steps - 1:
                i += 1
                alpha_next = alphas[i]
                beta_next = betas[i]
                t = torch.full((b,), timestamps[::-1][i], device=device, dtype=torch.long)
                eps_hat = self.denoise_fn(seq_hat, t, cond=cond)

                velocity = - eps_hat / torch.sqrt(1 - alpha_next)
                corr = beta_next * (seq_hat / 2 + velocity) * delta
                diff_next = torch.sqrt(beta_next * delta) * noise
                seq += (drift + corr) / 2 + (diff + diff_next) / 2
            else:
                seq = seq_hat

        return seq

    @torch.no_grad()
    def ddim_sample(self, shape, cond, timestamps):
        device = self.betas.device
        b = shape[0]
        seq = torch.randn(shape, device=device)

        alphas = torch.flip(self.alphas_cumprod[timestamps], dims=(0,))

        for i, t in enumerate(timestamps[::-1][:-1]):
            alpha_cur = alphas[i]
            alpha_prev = alphas[i + 1]
            t = torch.full((b,), t, device=device, dtype=torch.long)

            eps = self.denoise_fn(seq, t, cond=cond)
            predicted_seq0 = (seq - torch.sqrt(1 - alpha_cur) * eps) / torch.sqrt(alpha_cur)
            direction_seqt = torch.sqrt(1 - alpha_prev) * eps
            seq = torch.sqrt(alpha_prev) * predicted_seq0 + direction_seqt

        return seq

    @torch.no_grad()
    def dpm1_sample(self, shape, cond, timestamps):
        device = self.betas.device
        b = shape[0]
        seq = torch.randn(shape, device=device)

        alphas = torch.flip(self.alphas_cumprod[timestamps], dims=(0,))
        means = torch.sqrt(alphas)
        sigmas = torch.sqrt(1 - alphas)
        lambdas = torch.log(means / sigmas)

        for i, t in enumerate(timestamps[::-1][:-1]):
            mean_cur = means[i]
            mean_next = means[i + 1]
            sigma_next = sigmas[i + 1]
            t = torch.full((b,), t, device=device, dtype=torch.long)

            eps = self.denoise_fn(seq, t, cond=cond)
            delta = lambdas[i + 1] - lambdas[i]
            seq = mean_next / mean_cur * seq - sigma_next * torch.expm1(delta) * eps

        return seq

    @torch.no_grad()
    def dpm2_sample(self, shape, cond):
        device = self.betas.device
        b = shape[0]
        seq = torch.randn(shape, device=device)

        alphas = self.alphas_cumprod
        means = torch.sqrt(alphas)
        sigmas = torch.sqrt(1 - alphas)
        lambdas = torch.log(means / sigmas)

        def inverse_lambda(lamb, norm_factor=1):
            beta_start = 0.0001
            beta_end = 0.1
            tmp = 2. * (beta_end - beta_start) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = beta_start**2 + tmp
            res = tmp / (torch.sqrt(Delta) + beta_start) / (beta_end - beta_start)
            return res * norm_factor

        norm_factor = float(self.num_timesteps / inverse_lambda(lambdas[-1]))
        lambdas = torch.linspace(lambdas[-1], lambdas[0], self.solver_num_steps)

        for i in range(len(lambdas) - 1):
            t = int(torch.round(inverse_lambda(lambdas[i], norm_factor=norm_factor))) - 1
            t_next = int(inverse_lambda(lambdas[i + 1], norm_factor=norm_factor))
            r = int(inverse_lambda((lambdas[i] + lambdas[i + 1]) / 2, norm_factor=norm_factor))
            mean_cur = means[t]
            mean_next = means[t_next]
            mean_r = means[r]
            sigma_next = sigmas[t_next]
            sigma_r = sigmas[r]
            delta = lambdas[i + 1] - lambdas[i]

            t = torch.full((b,), t, device=device, dtype=torch.long)
            r = torch.full((b,), r, device=device, dtype=torch.long)
            eps = self.denoise_fn(seq, t, cond=cond)

            seq_r = mean_r / mean_cur * seq - sigma_r * torch.expm1(delta / 2) * eps
            eps_r = self.denoise_fn(seq_r, r, cond=cond)
            seq = mean_next / mean_cur * seq - sigma_next * torch.expm1(delta) * eps_r

        return seq

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.target_dim,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape

        if self.solver_schedule == "linear":
            timestamps = np.linspace(0, self.num_timesteps - 1, self.solver_num_steps)
        elif self.solver_schedule == "quad":
            timestamps = (np.linspace(0, self.num_timesteps ** 0.5 - 1, self.solver_num_steps) ** 2).astype(int)
        else:
            raise NotImplementedError(f"unknown steps schedule: {self.solver_schedule}")

        if self.solver == 'ddpm':
            x_hat = self.ddpm_sample(shape, cond)  # TODO reshape x_hat to (B,T,-1)
        elif self.solver == 'euler':
            x_hat = self.euler_sample(shape, cond, timestamps)
        elif self.solver == 'heun':
            x_hat = self.heun_sample(shape, cond)
        elif self.solver == 'ddim':
            x_hat = self.ddim_sample(shape, cond, timestamps)
        elif self.solver == 'dpm1':
            x_hat = self.dpm1_sample(shape, cond, timestamps)
        elif self.solver == 'dpm2':
            x_hat = self.dpm2_sample(shape, cond)
        else:
            raise NotImplementedError(f"unknown solver: {self.solver}")

        if self.scale is not None:
            x_hat *= self.scale
        return x_hat

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)

        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def loss(self, x, cond, *args, **kwargs):
        if self.scale is not None:
            x /= self.scale

        B, T, _ = x.shape

        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
        loss = self.p_losses(
            x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, *args, **kwargs
        )

        return loss
