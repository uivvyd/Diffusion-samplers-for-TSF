# ---------------------------------------------------------------------------------
# Portions of this file are derived from CSDI
# - Source: https://github.com/ermongroup/CSDI
# - Paper: CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation
# - License: MIT license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
from einops import repeat
from probts.model.forecaster import Forecaster
from probts.model.nn.prob.diffusion_layers import diff_CSDI


class CSDI(Forecaster):
    def __init__(
        self, 
        channels: int = 64,
        emb_time_dim: int = 128,
        emb_feature_dim: int = 16,
        num_steps: int = 50,
        schedule: str = "quad",
        beta_start: float = 0.0001,
        beta_end: float = 0.5,
        diffusion_embedding_dim: int = 128,
        num_heads: int = 8,
        n_layers: int = 4,
        sample_size: int = 64,
        linear_trans: bool = False,
        solver_num_steps=10,
        solver='euler',
        solver_schedule='linear',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.autoregressive = False
        self.dist_args = nn.Identity()

        self.solver_num_steps = solver_num_steps
        self.solver_schedule = solver_schedule
        self.solver = solver

        self.emb_time_dim = emb_time_dim
        self.emb_feature_dim = emb_feature_dim
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        side_dim = self.emb_total_dim
        self.sample_size = sample_size

        input_dim = 2
        self.diffmodel = diff_CSDI(channels, diffusion_embedding_dim, side_dim, num_steps, num_heads, n_layers, inputdim=input_dim,linear=linear_trans)

        # parameters for diffusion models
        self.num_steps = num_steps
        if schedule == "quad":
            self.beta = np.linspace(
                beta_start ** 0.5, beta_end ** 0.5, self.num_steps
            ) ** 2
        elif schedule == "linear":
            self.beta = np.linspace(
                beta_start, beta_end, self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1).to(self.device)

    def time_embedding(self, pos, device, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def get_masks(self, batch_data):
        hist_observed_mask = batch_data.past_observed_values[:, -self.context_length:, ...]
        target_observed_mask = batch_data.future_observed_values
        observed_mask = torch.cat((hist_observed_mask, target_observed_mask), dim=1)

        cond_mask = torch.cat((hist_observed_mask, torch.zeros_like(target_observed_mask)), dim=1)
        return observed_mask, cond_mask # [B L K]

    def get_side_info(self, observed_data, cond_mask, target_dimension_indicator, observed_tp=None):
        
        B, K, L = observed_data.shape
        if observed_tp is None:
            observed_tp = torch.arange(L) * 1.0
            observed_tp = repeat(observed_tp, 'l -> b l', b=B).to(observed_data.device)

        time_embed = self.time_embedding(observed_tp, observed_data.device, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B,L,K, emb)
        feature_embed = self.embed_layer(target_dimension_indicator)  # (B, K,emb)
        feature_embed = feature_embed.unsqueeze(1).expand(-1, L, -1, -1) # (B,L,K, emb)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)
        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)

        side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info # (B,D,K,L)

    def loss(self, batch_data, observed_tp=None):
        past_target_cdf = batch_data.past_target_cdf[:, -self.context_length:, ...]
        future_target_cdf = batch_data.future_target_cdf

        observed_data = torch.cat([past_target_cdf, future_target_cdf], dim=1)
        B, L, K = observed_data.shape
        t = torch.randint(0, self.num_steps, [B]).to(past_target_cdf.device)

        observed_mask, gt_mask = self.get_masks(batch_data)
        feature_id = batch_data.target_dimension_indicator

        if K > self.sample_size:
            # sample subset
            sampled_data = []
            sampled_mask = []
            sampled_feature_id = []
            sampled_gt_mask = []
            for i in range(len(observed_data)):
                ind = np.arange(K)
                np.random.shuffle(ind)
                sampled_data.append(observed_data[i,...,ind[:self.sample_size]])
                sampled_mask.append(observed_mask[i,...,ind[:self.sample_size]])
                sampled_feature_id.append(feature_id[i,ind[:self.sample_size]])
                sampled_gt_mask.append(gt_mask[i,...,ind[:self.sample_size]])
            observed_data = torch.stack(sampled_data,0)
            observed_mask = torch.stack(sampled_mask,0)
            feature_id = torch.stack(sampled_feature_id,0)
            gt_mask = torch.stack(sampled_gt_mask,0)

        observed_data = observed_data.permute(0,2,1) # [B K L]
        observed_mask = observed_mask.permute(0,2,1) # [B K L]
        cond_mask = gt_mask.permute(0,2,1) # [B K L]

        side_info = self.get_side_info(observed_data, cond_mask, feature_id, observed_tp)

        target_mask = observed_mask - cond_mask
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data).to(observed_data.device)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise


        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        residual = (noise - predicted) * target_mask

        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples):
        observed_data = torch.cat([batch_data.past_target_cdf[:, -self.context_length:, ...], torch.zeros_like(batch_data.future_target_cdf)], dim=1).permute(0,2,1) 
        _, cond_mask = self.get_masks(batch_data)
        cond_mask = cond_mask.permute(0,2,1)
        side_info = self.get_side_info(observed_data, cond_mask, batch_data.target_dimension_indicator)
        sample = self.sample(observed_data, cond_mask, side_info, num_samples)
        sample = sample.permute(0,1,3,2)
        return sample[:, : , -self.prediction_length:, :] # [B N L K]

    @torch.no_grad()
    def get_eps(self, sample, t, observed_data, cond_mask, side_info):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * sample).unsqueeze(1)  # [B 1 K L]
        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(observed_data.device))
        return predicted

    @torch.no_grad()
    def ddpm_sample(self, observed_data, cond_mask, side_info):
        current_sample = torch.randn_like(observed_data).to(observed_data.device)

        for t in range(self.num_steps - 1, -1, -1):
            eps = self.get_eps(current_sample, t, observed_data, cond_mask, side_info)

            coeff1 = 1 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
            current_sample = coeff1 * (current_sample - coeff2 * eps)

            if t > 0:
                noise = torch.randn_like(current_sample).to(observed_data.device)
                sigma = (
                                (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                        ) ** 0.5
                current_sample += sigma * noise

        return current_sample

    @torch.no_grad()
    def euler_sample(self, observed_data, cond_mask, side_info, timestamps):
        seq = torch.randn_like(observed_data).to(observed_data.device)

        alphas = torch.flip(torch.tensor(self.alpha)[timestamps], dims=(0,))
        betas = torch.flip(torch.tensor(self.beta)[timestamps], dims=(0,))

        for i, t in enumerate(timestamps[::-1][:-1]):
            alpha_cur = alphas[i]
            beta_cur = betas[i]
            noise = torch.randn_like(seq)
            eps = self.get_eps(seq, t, observed_data, cond_mask, side_info)

            sigma_t = torch.sqrt(1 - alpha_cur)
            velocity = - eps / sigma_t
            delta = timestamps[i + 1] - timestamps[i]

            seq += beta_cur * (seq / 2 + velocity) * delta
            seq += torch.sqrt(beta_cur * delta) * noise

        return seq

    @torch.no_grad()
    def heun_sample(self, observed_data, cond_mask, side_info):
        timestamps = np.linspace(0, self.num_steps - 1, self.solver_num_steps).astype(int)
        delta = timestamps[1] - timestamps[0]
        seq = torch.randn_like(observed_data).to(observed_data.device)

        alphas = torch.flip(torch.tensor(self.alpha)[timestamps], dims=(0,))
        betas = torch.flip(torch.tensor(self.beta)[timestamps], dims=(0,))

        for i, t in enumerate(timestamps[::-1][:-1]):
            alpha_cur = alphas[i]
            beta_cur = betas[i]
            noise = torch.randn_like(seq)
            eps = self.get_eps(seq, t, observed_data, cond_mask, side_info)

            sigma_t = torch.sqrt(1 - alpha_cur)
            velocity = - eps / sigma_t
            drift = beta_cur * (seq / 2 + velocity) * delta
            diff = torch.sqrt(beta_cur * delta) * noise
            seq_hat = seq + drift + diff

            if i < self.solver_num_steps - 1:
                i += 1
                alpha_next = alphas[i]
                beta_next = betas[i]
                eps_hat = self.get_eps(seq_hat, timestamps[::-1][i], observed_data, cond_mask, side_info)

                velocity = - eps_hat / torch.sqrt(1 - alpha_next)
                corr = beta_next * (seq_hat / 2 + velocity) * delta
                diff_next = torch.sqrt(beta_next * delta) * noise
                seq += (drift + corr) / 2 + (diff + diff_next) / 2
            else:
                seq = seq_hat

        return seq

    @torch.no_grad()
    def ddim_sample(self, observed_data, cond_mask, side_info, timestamps):
        seq = torch.randn_like(observed_data).to(observed_data.device)

        alphas = torch.flip(torch.tensor(self.alpha)[timestamps], dims=(0,))
        
        for i, t in enumerate(timestamps[::-1][:-1]):
            alpha_cur = alphas[i]
            alpha_prev = alphas[i + 1]
            eps = self.get_eps(seq, t, observed_data, cond_mask, side_info)

            predicted_seq0 = (seq - torch.sqrt(1 - alpha_cur) * eps) / torch.sqrt(alpha_cur)
            direction_seqt = torch.sqrt(1 - alpha_prev) * eps
            seq = torch.sqrt(alpha_prev) * predicted_seq0 + direction_seqt

        return seq

    @torch.no_grad()
    def dpm1_sample(self, observed_data, cond_mask, side_info, timestamps):
        seq = torch.randn_like(observed_data).to(observed_data.device)

        alphas = torch.flip(torch.tensor(self.alpha)[timestamps], dims=(0,))
        means = torch.sqrt(alphas)
        sigmas = torch.sqrt(1 - alphas)
        lambdas = torch.log(means / sigmas)

        for i, t in enumerate(timestamps[::-1][:-1]):
            mean_cur = means[i]
            mean_next = means[i + 1]
            sigma_next = sigmas[i + 1]
            delta = lambdas[i + 1] - lambdas[i]
            eps = self.get_eps(seq, t, observed_data, cond_mask, side_info)

            seq = mean_next / mean_cur * seq - sigma_next * torch.expm1(delta) * eps

        return seq

    @torch.no_grad()
    def dpm2_sample(self, observed_data, cond_mask, side_info):
        seq = torch.randn_like(observed_data).to(observed_data.device)

        alphas = torch.tensor(self.alpha)
        means = torch.sqrt(alphas)
        sigmas = torch.sqrt(1 - alphas)
        lambdas = torch.log(means / sigmas)

        def inverse_lambda(lamb, norm_factor=1):
            beta_start = self.beta[0]
            beta_end = self.beta[-1]
            tmp = 2. * (beta_end - beta_start) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = beta_start ** 2 + tmp
            res = tmp / (torch.sqrt(Delta) + beta_start) / (beta_end - beta_start)
            return res * norm_factor

        norm_factor = float(self.num_steps / inverse_lambda(lambdas[-1]))
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

            eps = self.get_eps(seq, t, observed_data, cond_mask, side_info)
            seq_r = mean_r / mean_cur * seq - sigma_r * torch.expm1(delta / 2) * eps
            eps_r = self.get_eps(seq_r, r, observed_data, cond_mask, side_info)
            seq = mean_next / mean_cur * seq - sigma_next * torch.expm1(delta) * eps_r

        return seq

    @torch.no_grad()
    def sample(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(observed_data.device)

        if self.solver_schedule == "linear":
            timestamps = np.linspace(0, self.num_steps - 1, self.solver_num_steps).astype(int)
        elif self.solver_schedule == "quad":
            timestamps = (np.linspace(0, self.num_steps ** 0.5 - 1, self.solver_num_steps) ** 2).astype(int)
        else:
            raise NotImplementedError(f"unknown steps schedule: {self.solver_schedule}")

        for i in range(n_samples):
            if self.solver == 'ddpm':
                current_sample = self.ddpm_sample(observed_data, cond_mask, side_info)
            elif self.solver == 'euler':
                current_sample = self.euler_sample(observed_data, cond_mask, side_info, timestamps)
            elif self.solver == 'heun':
                current_sample = self.heun_sample(observed_data, cond_mask, side_info)
            elif self.solver == 'ddim':
                current_sample =  self.ddim_sample(observed_data, cond_mask, side_info, timestamps)
            elif self.solver == 'dpm1':
                current_sample =  self.dpm1_sample(observed_data, cond_mask, side_info, timestamps)
            elif self.solver == 'dpm2':
                current_sample =  self.dpm2_sample(observed_data, cond_mask, side_info)
            else:
                raise NotImplementedError(f"unknown solver: {self.solver}")

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples
