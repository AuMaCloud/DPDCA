# -*- coding: utf-8 -*-
import pandas as pd
import torch
import numpy as np

from scipy.stats import pearsonr
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from Cervical.model.autoencoder_tool_cer import Autoencoder
from Cervical.model.diffusion_tool_cer import Diffusion, MlpDiffusion

def dm_ld(model_path, output_path, npy_file_path, device='cuda:0', z_dim=64, time_dim=16, unit_dim=[32, 16, 16, 16, 32],
          dim=64, P_mean=-1.2, P_std=1.2, sigma_data=0.14, num_epochs_sample=32, sigma_min=0.02, sigma_max=80, rho=7,
          batch_size=835):
    device = torch.device(device)

    dm = MlpDiffusion(z_dim=z_dim, time_dim=time_dim, unit_dim=unit_dim)
    checkpoint = torch.load(model_path)
    dm.load_state_dict(checkpoint['diffusion_state_dict'])
    dm.to(device)

    diffusion = Diffusion(
        dm,
        dim=dim,
        P_mean=P_mean,
        P_std=P_std,
        sigma_data=sigma_data,
        num_epochs_sample=num_epochs_sample,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
    )

    out = []
    dm.eval()

    sampled_seq = diffusion.sample(batch_size=batch_size)
    out.append(sampled_seq)
    out_seq = torch.cat(out)
    out_seq = out_seq.detach().cpu().numpy()
    res = np.clip(out_seq, 0, 1)

    train_data = np.load(npy_file_path).astype(float)
    train_data = train_data.transpose(0, 2, 1)
    train_data = np.squeeze(train_data, axis=1)
    train_data_mean = np.mean(train_data, axis=0)
    temp_data_mean = np.mean(out_seq, axis=0)

    corr = pearsonr(temp_data_mean, train_data_mean)

    np.save(output_path, np.expand_dims(out_seq, axis=-1))
    loaded_data = np.load(output_path)
    print(loaded_data.shape)

model_path = "../experiments/Cervical/diffusionModel/df.pth"
output_path = "../data/dm_data.npy"
npy_file_path = "../data/en_data.npy"
dm_ld(model_path, output_path, npy_file_path, batch_size=835)

# Replace with the actual module name

device = torch.device('cuda:0')
Tensor = torch.cuda.FloatTensor  # Assuming you want to use GPU

def test_dm():
    model_path = "../experiments/Cervical/autoencoderModel/ae.pth"
    checkpoint = torch.load(model_path)

    autoencoderModel = Autoencoder()
    autoencoderModel = autoencoderModel.to(device)

    state_dict = checkpoint['autoencoder_state_dict']

    autoencoderModel.load_state_dict(state_dict)

    autoencoderModel.eval()

    data = np.load('../data/dm_data.npy')

    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

    dataloader = DataLoader(data_tensor, batch_size=64, shuffle=False, drop_last=False)

    decoded_data_list = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            decoded_batch = autoencoderModel.decode(batch)
            decoded_data_list.append(decoded_batch.cpu().numpy())

    decoded_data = np.concatenate(decoded_data_list, axis=0)

    decoded_df = pd.DataFrame(decoded_data, columns=None, index=None)

    def transform_label(value):
        return 1 if value > 0.5 else 0

    decoded_df[34] = decoded_df[34].apply(lambda x: transform_label(x))

    decoded_df.to_csv('../data/dm_de_data.csv', index=False)

test_dm()

