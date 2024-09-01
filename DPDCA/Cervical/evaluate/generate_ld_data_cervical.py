# -*- coding: utf-8 -*-
import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader

from Cervical.model.autoencoder_tool_cer import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_and_save_data(input_csv, model_path, output_dir, output_filename):
    autoencoderModel = Autoencoder()
    autoencoderModel = autoencoderModel.to(device)
    checkpoint = torch.load(model_path)
    autoencoderModel.load_state_dict(checkpoint['autoencoder_state_dict'])
    autoencoderModel.eval()

    data = pd.read_csv(input_csv)

    input_data = data.iloc[:, :].to_numpy()

    dataloader = DataLoader(input_data, batch_size=64, shuffle=False, drop_last=False)

    encoded_datas = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            real_datas = torch.tensor(batch, dtype=torch.float32).to(device)
            real_datas = real_datas.unsqueeze(1)
            encoder_data = autoencoderModel.encoder(real_datas)
            encoded_datas.append(encoder_data)

    encoded_datas = torch.cat(encoded_datas, dim=0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_filename)
    np.save(output_path, encoded_datas.cpu().numpy())

encode_and_save_data('../data/original_data.csv', '../experiments/Cervical/autoencoderModel/ae.pth', '../data',
                     'en_data.npy')

# 加载编码后的Numpy文件
data = np.load('../data/en_data.npy')

# 假设data是835*64*1的数据
data = np.reshape(data, (835, 1, 64))

# 查看数据形状
print("Encoded Data Shape:", data.shape)

# 打印第一条数据
print("First Data Point:")
print(data[0])
