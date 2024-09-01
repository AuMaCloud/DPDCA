# -*- coding: utf-8 -*-
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

def default(val, d):
    return val if val is not None else (d() if callable(d) else d)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


###########################################################################
# Some of the following functions are ported from code base:
# https://github.com/sczzz3/EHRDiff/blob/main/model/linear_model.py
###########################################################################
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim, time_embedding_dim=None):
        super(MlpBlock, self).__init__()

        if time_embedding_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embedding_dim, input_dim),
            )

        self.out_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, input, time_embedding=None):
        if time_embedding is not None:
            time_features = self.time_mlp(time_embedding)
            transformed_input = input + time_features
        else:
            transformed_input = input
        # print("transformed_input dtype:", transformed_input.dtype)
        output = self.out_projection(transformed_input)
        # print("output dtype:", output.dtype)
        return output


class MlpDiffusion(nn.Module):
    def __init__(self, z_dim, time_dim, unit_dim):
        super(MlpDiffusion, self).__init__()

        num_linears = len(unit_dim)

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(z_dim),
            nn.Linear(z_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.block_in = MlpBlock(input_dim=z_dim, output_dim=unit_dim[0], time_embedding_dim=time_dim)
        self.block_mid = nn.ModuleList()
        for i in range(num_linears - 1):
            self.block_mid.append(MlpBlock(input_dim=unit_dim[i], output_dim=unit_dim[i + 1]))
        self.block_out = MlpBlock(input_dim=unit_dim[-1], output_dim=z_dim)


    def forward(self, x, time_steps):

        x = x.to(torch.float)
        time_embedding = self.time_embedding(time_steps)
        x = self.block_in(x, time_embedding)

        num_mid_blocks = len(self.block_mid)
        if num_mid_blocks > 0:
            for block in self.block_mid:
                x = block(x)

        x = self.block_out(x)
        return x


class Diffusion(nn.Module):
    def __init__(self, network, *, num_epochs_sample, dim, sigma_min, sigma_max, sigma_data, rho, P_mean, P_std):
        super().__init__()

        self.network=network
        self.num_epochs_sample = num_epochs_sample
        self.dim = dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std

    @property
    def device(self):
        return next(self.network.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    def preconditioned_network_forward(self, noised_ehr, sigma):
        batch, device = noised_ehr.shape[0], noised_ehr.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        p_sigma = rearrange(sigma, 'b -> b 1')

        x = self.c_in(p_sigma) * noised_ehr
        y = self.c_noise(sigma)

        net_out = self.network(
            x,
            y,
        )


        out = self.c_skip(p_sigma) * noised_ehr + self.c_out(p_sigma) * net_out

        return out

    def sample_schedule(self, num_epochs_sample=None):
        num_epochs_sample = default(num_epochs_sample, self.num_epochs_sample)

        N = num_epochs_sample
        inv_rho = 1 / self.rho

        steps = torch.arange(num_epochs_sample, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        return sigmas

    @torch.no_grad()
    def sample(self, batch_size=32, num_epochs_sample=None):

        num_epochs_sample = default(num_epochs_sample, self.num_epochs_sample)

        shape = (batch_size, self.dim)

        sigmas = self.sample_schedule(num_epochs_sample)

        sigmas_and_sigmas_next = list(zip(sigmas[:-1], sigmas[1:]))

        init_sigma = sigmas[0]

        ehr = init_sigma * torch.randn(shape, device=self.device)

        for sigma, sigma_next in sigmas_and_sigmas_next:

            sigma, sigma_next = map(lambda t: t.item(), (sigma, sigma_next))

            model_output = self.preconditioned_network_forward(ehr, sigma)

            denoised_over_sigma = (ehr - model_output) / sigma

            #Xti+1
            ehr_next = ehr + (sigma_next - sigma) * denoised_over_sigma

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(ehr_next, sigma_next,)

                denoised_prime_over_sigma = (ehr_next - model_output_next) / sigma_next
                ehr_next = ehr + 0.5 * (sigma_next - sigma) * (denoised_over_sigma + denoised_prime_over_sigma)

            ehr = ehr_next

        return ehr

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def forward(self, ehr):
        batch_size = ehr.shape[0]

        sigmas = (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

        p_sigmas = rearrange(sigmas, 'b -> b 1')

        noise = torch.randn_like(ehr)

        noised_ehr = ehr + p_sigmas * noise

        denoised = self.preconditioned_network_forward(noised_ehr, sigmas)

        losses = F.mse_loss(denoised, ehr, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)

        return losses.mean()


# def plot_dim_dist(train_data, syn_data, model_setting, best_corr):
#     train_data_mean = np.mean(train_data, axis=0)
#     temp_data_mean = np.mean(syn_data, axis=0)
#     corr = pearsonr(temp_data_mean, train_data_mean)
#     nzc = sum(temp_data_mean[i] > 0 for i in range(temp_data_mean.shape[0]))
#
#     fig, ax = plt.subplots(figsize=(12, 10))
#
#     # Scatter plot - Original data
#     ax.scatter(train_data_mean, train_data_mean, alpha=0.3, label='Original Data', color='blue', s=30)
#
#     # Scatter plot - Synthetic data
#     ax.scatter(train_data_mean, temp_data_mean, alpha=0.3, label='Synthetic Data', color='green', s=30)
#
#     # Linear regression line
#     slope, intercept = np.polyfit(train_data_mean, temp_data_mean, 1)
#     fitted_values = [slope * i + intercept for i in train_data_mean]
#
#     # Identity line
#     identity_values = [i for i in train_data_mean]
#
#     # Adjust line color and transparency
#     ax.plot(train_data_mean, fitted_values, 'b-', alpha=0.8, label='Linear Regression Line')
#     ax.plot(train_data_mean, identity_values, 'r--', alpha=0.8, label='Identity Line')
#
#     ax.set_title('Correlation: %.4f, Non-zero columns: %d, Slope: %.4f' % (corr[0], nzc, slope))
#     ax.set_xlabel('Feature prevalence')
#     ax.set_ylabel('Feature prevalence')
#
#     # Add axis labels
#     ax.legend()
#     plt.grid(True)
#
#     # Create file path
#     save_dir = '../experiments/UCI/figs'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     # Save the figure
#     cur_res_fig_path = os.path.join(save_dir, 'Cur_res.png')
#     fig.savefig(cur_res_fig_path)
#
#     flag = False
#     if corr[0] > best_corr:
#         best_corr = corr[0]
#         flag = True
#
#         best_res_fig_path = os.path.join(save_dir, 'Best_res.png')
#         fig.savefig(best_res_fig_path)
#
#     plt.close(fig)
#     return corr[0], nzc, flag
#
#
#
# def plot_dim_dist(train_data, syn_data, model_setting, best_corr):
#     train_data_mean = np.mean(train_data, axis=0)
#     temp_data_mean = np.mean(syn_data, axis=0)
#     corr = pearsonr(temp_data_mean, train_data_mean)
#     nzc = sum(temp_data_mean[i] > 0 for i in range(temp_data_mean.shape[0]))
#
#     # Create a DataFrame for seaborn plotting
#     data = {'Original Data': train_data_mean, 'Synthetic Data': temp_data_mean}
#     df = pd.DataFrame(data)
#
#     # Create a joint plot
#     sns.set(style="whitegrid")
#     g = sns.jointplot(x='Original Data', y='Synthetic Data', data=df, kind='reg', height=10)
#
#     # Update the plot title
#     g.fig.suptitle('Correlation: %.4f, Non-zero columns: %d' % (corr[0], nzc))
#
#     # Save the figure
#     save_dir = '../experiments/UCI/figs'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     cur_res_fig_path = os.path.join(save_dir, 'Cur_res.png')
#     g.savefig(cur_res_fig_path)
#
#     flag = False
#     if corr[0] > best_corr:
#         best_corr = corr[0]
#         flag = True
#
#         best_res_fig_path = os.path.join(save_dir, 'Best_res.png')
#         g.savefig(best_res_fig_path)
#
#     # Set equal aspect ratio on the joint plot
#     g.ax_joint.set_aspect('equal', adjustable='box')
#
#     plt.show()
#
#     return corr[0], nzc, flag


def plot_joint_distribution(train_data_mean, temp_data_mean, corr, nzc, save_path):
    # Create a DataFrame for seaborn plotting
    data = {'Original Data': train_data_mean, 'Synthetic Data': temp_data_mean}
    df = pd.DataFrame(data)

    # Create a joint plot
    sns.set(style="whitegrid")
    g = sns.jointplot(x='Original Data', y='Synthetic Data', data=df, kind='reg', height=10, marginal_kws=dict(bins=50))

    # Update the plot title
    g.fig.suptitle('Correlation: %.4f, Non-zero columns: %d' % (corr, nzc))

    # Save the figure
    plt.savefig(save_path)
    plt.show()

    # Set equal aspect ratio on the joint plot
    g.ax_joint.set_aspect('equal', adjustable='box')

def plot_code_count_distribution(train_data_mean, temp_data_mean, save_path):
    # Plot histograms for code count distributions
    plt.figure(figsize=(12, 6))

    # Plot the histogram for the code count distribution of original data
    sns.histplot(train_data_mean, bins=20, color='blue', kde=True, label='Original Data')

    # Plot the histogram for the code count distribution of synthetic data
    sns.histplot(temp_data_mean, bins=20, color='orange', kde=True, label='Synthetic Data')

    plt.title('Comparison of Code Count Distributions between Original and Synthetic Data')
    plt.legend()

    # Save the figure
    plt.savefig(save_path)
    plt.show()

def plot_dim_dist(train_data, syn_data, model_setting, best_corr):
    train_data_mean = np.mean(train_data, axis=0)
    temp_data_mean = np.mean(syn_data, axis=0)
    corr = pearsonr(temp_data_mean, train_data_mean)
    nzc = sum(temp_data_mean[i] > 0 for i in range(temp_data_mean.shape[0]))

    # Save the figure directory
    save_dir = '../experiments/UCI/figs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot the joint distribution
    cur_res_fig_path = os.path.join(save_dir, 'Cur_res.png')
    plot_joint_distribution(train_data_mean, temp_data_mean, corr[0], nzc, cur_res_fig_path)

    flag = False
    if corr[0] > best_corr:
        best_corr = corr[0]
        flag = True

        # Plot the best result
        best_res_fig_path = os.path.join(save_dir, 'Best_res.png')
        plot_joint_distribution(train_data_mean, temp_data_mean, corr[0], nzc, best_res_fig_path)

    # Plot the code count distribution
    code_count_fig_path = os.path.join(save_dir, 'Code_Count_Distribution.png')
    plot_code_count_distribution(train_data_mean, temp_data_mean, code_count_fig_path)

    return corr[0], nzc, flag

