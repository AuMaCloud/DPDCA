# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
from sklearn.utils import shuffle

from torch.utils.data import DataLoader
from torch.autograd import Variable

from UCI.model.autoencoder_tool import Autoencoder, set_seed, weights_init, autoencoder_loss
from UCI.utils import analysis, dp_optimizer

config = {
    'experimentName': 'UCI',
    'exp_path': '../experiments/UCI',
    'model_path': '../experiments/UCI/autoencoderModel',
    'data_path': '../data',
    'batch_size': 128,
    'if_shuffle': True,
    'if_drop_last': True,
    'lr': 0.001,
    'weight_decay': 0.00001,
    'b1': 0.9,
    'b2': 0.999,
    'noise_multiplier': 0.65,  # 22.4 2 0.65 0.316
    'max_per_sample_grad_norm': 1.0,
    'delta': 1e-5,
    'num_epochs': 20,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

####################
### Architecture ###
####################
class Dataset:
    def __init__(self, data, transform=None):
        # Transform
        self.transform = transform
        # load data here
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            pass

        return torch.from_numpy(sample)

    def get_data(self):
        return self.data


def train_autoencoder():
    ############################################
    ### 1. Create experiments and models DIR ###
    ############################################
    if not os.path.exists(config['exp_path']):
        os.system('mkdir "{0}"'.format(config['exp_path']))
    if not os.path.exists(config['model_path']):
        os.system('mkdir "{0}"'.format(config['model_path']))

    ######################
    ### 2. Random seed ###
    ######################
    set_seed(2000)

    #############################
    ### 3. Dataset Processing ###
    #############################
    train_data = pd.read_csv(os.path.join(config['data_path'], 'original_data.csv'))
    train_data = train_data.iloc[:, :].to_numpy()

    # Train data loader
    train_dataset = Dataset(data=train_data, transform=False)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=train_data, replacement=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=config['if_shuffle'],
        drop_last=config['if_drop_last'],
    )

    ##############################
    ## 4. Privacy Calculation ####
    ##############################
    totalsamples = len(train_dataset)
    num_batches = len(train_dataloader)
    print("批次数目: {}".format(num_batches))
    iterations = config['num_epochs'] * num_batches
    print('实现了({}, {})的差分隐私'.format(
        analysis.epsilon(
            totalsamples,
            config['batch_size'],
            config['noise_multiplier'],
            iterations,
            config['delta']
        ),
        config['delta'],
    ))

    #######################################
    ### 5. model and Parameters Settings###
    #######################################
    autoencoderModel = Autoencoder()
    autoencoderDecoder = autoencoderModel.decoder
    Tensor = torch.FloatTensor

    # put into cuda
    autoencoderModel.to(device)
    autoencoderDecoder.to(device)
    Tensor = torch.cuda.FloatTensor

    # Weight initialization
    autoencoderModel.apply(weights_init)

    optimizer = dp_optimizer.optimizerDP(
        max_per_sample_grad_norm=config['max_per_sample_grad_norm'],
        noise_multiplier=config['noise_multiplier'],
        batch_size=config['batch_size'],
        params=autoencoderModel.parameters(),
        lr=config['lr'],
        betas=(config['b1'], config['b2']),
        weight_decay=config['weight_decay'],
    )

    ################
    ### TRAINING ###
    ################
    for epoch in range(config['num_epochs']):
        for step, batch in enumerate(train_dataloader):

            real_datas = Variable(batch.type(Tensor))

            optimizer.zero_grad()

            for i in range(config['batch_size']):
                # 1. one sample
                one_sample = real_datas[i:i + 1, :]
                # 2. reset grads
                optimizer.zero_grad()
                # 3. generate sample
                samples = autoencoderModel(one_sample)
                # 4. loss measures
                loss_ae = autoencoder_loss(samples, one_sample)
                # 5. backward
                loss_ae.backward()
                # 6. bound sensitivity
                optimizer.clip_grads_()

            ################### Privacy ################
            optimizer.add_noise_()
            # 7. Update model parameters based on gradient information
            optimizer.step()

            # Print training progress information
            current_step = epoch * len(train_dataloader) + step + 1

            if current_step % 10 == 0:
                print(
                    f"[Epoch {epoch + 1}/{config['num_epochs']}] "
                    f"[step {step + 1}/{len(train_dataloader)}] [loss: {loss_ae.item():.5f}]",
                    flush=True
                )

    torch.save({
        'autoencoder_state_dict': autoencoderModel.state_dict(),
        # 'autoencoder_encoder_state_dict': autoencoderModel.encoder.state_dict(),
        # 'autoencoder_decoder_state_dict': autoencoderModel.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(config['model_path'], "ae_01.pth"))


def get_ae_data(data_path):
    Tensor = torch.cuda.FloatTensor

    model_path = "../experiments/UCI/autoencoderModel/ae_01.pth"  # 替换为你的模型权重文件路径
    checkpoint = torch.load(model_path)

    autoencoderModel = Autoencoder()
    autoencoderModel = autoencoderModel.to(device)

    state_dict = checkpoint['autoencoder_state_dict']

    autoencoderModel.load_state_dict(state_dict)

    autoencoderModel.eval()

    data = pd.read_csv(data_path)

    input_data = data.iloc[:, 1:-1].to_numpy()

    dataloader = DataLoader(input_data, batch_size=64, shuffle=False, drop_last=False)

    encoded_datas = []
    original_data = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            real_datas = torch.tensor(batch, dtype=torch.float32).to(device)
            real_datas = real_datas.unsqueeze(1)
            encoder_data = autoencoderModel.encoder(real_datas)
            encoded_datas.append(encoder_data)
            original_data.append(batch)

    encoded_datas = torch.cat(encoded_datas, dim=0)
    original_data = torch.cat(original_data, dim=0)

    with torch.no_grad():
        decoded_data = autoencoderModel.decode(encoded_datas)

    if 'positive' in data_path:
        labels = torch.ones((decoded_data.shape[0], 1), dtype=torch.float32, device=device)

    else:
        labels = torch.zeros((decoded_data.shape[0], 1), dtype=torch.float32, device=device)

    decoded_data_with_labels = torch.cat([decoded_data, labels], dim=1)

    decoded_df = pd.DataFrame(decoded_data_with_labels.cpu().numpy(), columns=None, index=None)

    if 'positive' in data_path:
        decoded_df.to_csv('../data/ae_data_p.csv', index=False)
    else:
        decoded_df.to_csv('../data/ae_data_n.csv', index=False)

def merge_dataset():
    dataset1 = pd.read_csv("../data/ae_data_n.csv")

    dataset2 = pd.read_csv("../data/ae_data_p.csv")

    merged_dataset = pd.concat([dataset1, dataset2], axis=0, ignore_index=True)

    shuffled_dataset = shuffle(merged_dataset)

    shuffled_dataset.to_csv("../data/ae_data_01.csv", index=False)

    import os
    os.remove("../data/ae_data_n.csv")
    os.remove("../data/ae_data_p.csv")

train_autoencoder()
get_ae_data('../data/positive.csv')
get_ae_data('../data/negative.csv')
merge_dataset()

def ae_data(data_path):
    Tensor = torch.cuda.FloatTensor

    model_path = "../experiments/UCI/autoencoderModel/ae_01.pth"  # 替换为你的模型权重文件路径
    checkpoint = torch.load(model_path)

    autoencoderModel = Autoencoder()
    autoencoderModel = autoencoderModel.to(device)

    state_dict = checkpoint['autoencoder_state_dict']

    autoencoderModel.load_state_dict(state_dict)

    autoencoderModel.eval()

    data = pd.read_csv(data_path)

    input_data = data.iloc[:, :].to_numpy()

    dataloader = DataLoader(input_data, batch_size=64, shuffle=False, drop_last=False)

    encoded_datas = []
    original_data = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            real_datas = torch.tensor(batch, dtype=torch.float32).to(device)
            real_datas = real_datas.unsqueeze(1)
            encoder_data = autoencoderModel.encoder(real_datas)
            encoded_datas.append(encoder_data)
            original_data.append(batch)

    encoded_datas = torch.cat(encoded_datas, dim=0)
    original_data = torch.cat(original_data, dim=0)

    with torch.no_grad():
        decoded_data = autoencoderModel.decode(encoded_datas)

    decoded_df = pd.DataFrame(decoded_data.cpu().numpy(), columns=None, index=None)

    def transform_label(value):
        return 1 if value > 0.5 else 0

    decoded_df[178] = decoded_df[178].apply(lambda x: transform_label(x))

    decoded_df.to_csv('../data/ae_data_01.csv', index=False)

ae_data('../data/original_data.csv')
