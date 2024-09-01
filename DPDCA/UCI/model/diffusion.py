# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from UCI.model.autoencoder_tool import set_seed
from UCI.model.diffusion_tool import MlpDiffusion, Diffusion, plot_dim_dist

config = {
    'experimentName': 'UCI',
    'exp_path': '../experiments/UCI',
    'model_path': '../experiments/UCI/diffusionModel',
    'data_path': '../data',
    'batch_size': 256,
    'if_shuffle': True,
    'if_drop_last': True,
    'data_dim': 64,
    'time_dim': 16,
    'mlp_dims': [32, 16, 16, 16, 32],
    'num_epochs_sample': 32,
    'sigma_min': 0.02,
    'sigma_max': 80,
    'sigma_data': 0.14,
    'rho': 7,
    'p_mean': -1.2,
    'p_std': 1.2,
    'lr': 3e-4,
    'weight_decay': 0.00001,
    'num_epochs': 200,
    'warmup_steps': 20000,
    'noise_multiplier': 0.0005,
    'max_per_sample_grad_norm': 1.0,
    'delta': 1e-5,
    'syn_samples': 11500
}

model_setting = 'sigma_data' + str(config['sigma_data']) + '|' + \
                'p_mean' + str(config['p_mean']) + '|' + \
                'p_std' + str(config['p_std']) + '|' + \
                'steps' + str(config['num_epochs_sample']) + '|' + \
                'sigma_min' + str(config['sigma_min']) + '|' + \
                'sigma_max' + str(config['sigma_max']) + '|' + \
                'rho' + str(config['rho'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

####################
### Architecture ###
####################
class Dataset:
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index:int):
        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.data[index]

        return torch.from_numpy(sample)

    def get_data(self):
        return self.data

def train_diffusion():
    ############################################
    ### 1. create experiments and models DIR ###
    ############################################
    if not os.path.exists(config['exp_path']):
        os.system('mkdir "{0}"'.format(config['exp_path']))
    if not os.path.exists(config['model_path']):
        os.system('mkdir "{0}"'.format(config['model_path']))

    ######################
    ### 2. random seed ###
    ######################
    set_seed(2000)

    #############################
    ### 3. dataset processing ###
    #############################
    npy_file_path = os.path.join(config['data_path'], 'en_data.npy')
    train_data = np.load(npy_file_path).astype(float)
    train_data = train_data.transpose(0, 2, 1)  # 交换第二维和第三维
    train_data = np.squeeze(train_data, axis=1)

    # print(train_data[0])

    dataset = Dataset(data=train_data)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=config['if_shuffle'],
        drop_last=config['if_drop_last'],
    )

    #######################################
    ### 4. model and parameters settings###
    #######################################
    MlpModel = MlpDiffusion(
        z_dim=config['data_dim'],
        time_dim=config['time_dim'],
        unit_dim=config['mlp_dims'],
    )
    MlpModel.to(device)

    optimizer = torch.optim.AdamW(
        params=MlpModel.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    if config['if_drop_last']:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=(train_data.shape[0] // config['batch_size']) * config['num_epochs'],
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=(train_data.shape[0] // config['batch_size'] + 1) * config['num_epochs'],
        )

    diffusionModel = Diffusion(
        MlpModel,
        num_epochs_sample=config['num_epochs_sample'],
        dim=config['data_dim'],
        sigma_min=config['sigma_min'],
        sigma_max=config['sigma_max'],
        sigma_data=config['sigma_data'],
        rho=config['rho'],
        P_mean=config['p_mean'],
        P_std=config['p_std'],
    )

    ###################
    ### 6. TRAINING ###
    ###################

    train_dm_loss = 0
    train_cnt = 0
    best_corr = 0

    for epoch in range(config['num_epochs']):
        for step, batch in enumerate(dataloader):
            batch_size = batch.shape[0]
            batch = batch.to(device)
            # print(batch.shape)

            optimizer.zero_grad()

            loss_dm = diffusionModel(batch)
            train_dm_loss += loss_dm.item()
            train_cnt += batch_size

            current_step = epoch * len(dataloader) + step + 1

            if current_step % 10 == 0:
                print(
                    f"[Epoch {epoch + 1}/{config['num_epochs']}] "
                    f"[step {step + 1}/{len(dataloader)}] [loss: {loss_dm.item():.5f}]",
                    flush=True
                )

                MlpModel.eval()

                # 生成数据
                num_iters = config['syn_samples'] // config['batch_size']
                num_left = config['syn_samples'] % config['batch_size']
                syn_data = []
                for _ in range(num_iters):
                    syn_data.append(diffusionModel.sample(batch_size=config['batch_size']).detach().cpu().numpy())
                if num_left > 0:
                    syn_data.append(diffusionModel.sample(batch_size=num_left).detach().cpu().numpy())
                syn_data = np.concatenate(syn_data)
                corr, nzc, flag = plot_dim_dist(train_data, syn_data, model_setting, best_corr)
                print(f"[corr{corr:.4f}, none-zero columns: {nzc:d}]")

                if flag or corr >= best_corr:
                    best_corr = corr
                    torch.save({
                        'diffusion_state_dict':MlpModel.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                    }, os.path.join(config['model_path'], 'df.pth'))
                    print(f"New Weight saved!")
                print("**************************************")

                MlpModel.train()

            loss_dm.backward()
            optimizer.step()
            scheduler.step()

train_diffusion()



