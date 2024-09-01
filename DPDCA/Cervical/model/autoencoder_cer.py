import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from Cervical.model.autoencoder_tool_cer import Autoencoder, set_seed, weights_init, autoencoder_loss
from UCI.utils import analysis, dp_optimizer

config = {
    'experimentName': 'Cervical',
    'exp_path': '../experiments/Cervical',
    'model_path': '../experiments/Cervical/autoencoderModel',
    'data_path': '../data',
    'batch_size': 64,
    'if_shuffle': True,
    'if_drop_last': True,
    'lr': 0.0001,
    'weight_decay': 1e-5,
    'b1': 0.9,
    'b2': 0.999,
    'noise_multiplier': 0.0000000001,
    'max_per_sample_grad_norm': 1.0,
    'delta': 1e-5,
    'num_epochs': 2000,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        return sample

def train_autoencoder():
    if not os.path.exists(config['exp_path']):
        os.makedirs(config['exp_path'])
    if not os.path.exists(config['model_path']):
        os.makedirs(config['model_path'])

    set_seed(2000)

    train_data = pd.read_csv(os.path.join(config['data_path'], 'original_data.csv')).to_numpy()

    train_dataset = CustomDataset(train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=config['if_shuffle'],
        drop_last=config['if_drop_last'],
    )

    totalsamples = len(train_dataset)
    num_batches = len(train_dataloader)
    iterations = config['num_epochs'] * num_batches
    print("Privacy achieved with (ε, δ)=({}, {})".format(
        analysis.epsilon(
            totalsamples,
            config['batch_size'],
            config['noise_multiplier'],
            iterations,
            config['delta']
        ),
        config['delta'],
    ))

    autoencoderModel = Autoencoder().to(device)
    autoencoderModel.apply(weights_init)

    # optimizer = dp_optimizer.optimizerDP(
    #     max_per_sample_grad_norm=config['max_per_sample_grad_norm'],
    #     noise_multiplier=config['noise_multiplier'],
    #     batch_size=config['batch_size'],
    #     params=autoencoderModel.parameters(),
    #     lr=config['lr'],
    #     betas=(config['b1'], config['b2']),
    #     weight_decay=config['weight_decay'],
    # )

    optimizer = torch.optim.Adam(
        params=autoencoderModel.parameters(),
        lr=config['lr'],
        betas=(config['b1'], config['b2']),
        weight_decay=config['weight_decay']
    )


    for epoch in range(config['num_epochs']):
        # for step, batch in enumerate(train_dataloader):
        #     real_datas = batch.to(device)
        #
        #     optimizer.zero_grad()
        #
        #     for i in range(config['batch_size']):
        #         one_sample = real_datas[i:i + 1]
        #         samples = autoencoderModel(one_sample)
        #         loss_ae = autoencoder_loss(samples, one_sample)
        #         loss_ae.backward()
        #         optimizer.clip_grads_()
        #
        #     optimizer.add_noise_()
        #     optimizer.step()
        #
        #     current_step = epoch * len(train_dataloader) + step + 1
        #
        #     if current_step % 10 == 0:
        #         print(
        #             f"[Epoch {epoch + 1}/{config['num_epochs']}] "
        #             f"[step {step + 1}/{len(train_dataloader)}] [loss: {loss_ae.item():.5f}]",
        #             flush=True
        #         )

        for step, batch in enumerate(train_dataloader):
            real_datas = batch.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            samples = autoencoderModel(real_datas)

            # Compute loss
            loss_ae = autoencoder_loss(samples, real_datas)

            # Backward pass and optimize
            loss_ae.backward()
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
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(config['model_path'], "ae.pth"))

train_autoencoder()

def ae_data(data_path):
    model_path = "../experiments/Cervical/autoencoderModel/ae.pth"
    checkpoint = torch.load(model_path)
    autoencoderModel = Autoencoder().to(device)
    autoencoderModel.load_state_dict(checkpoint['autoencoder_state_dict'])
    autoencoderModel.eval()

    data = pd.read_csv(data_path).to_numpy()
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

    encoded_datas = []
    original_data = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            real_datas = batch.to(device)
            encoder_data = autoencoderModel.encoder(real_datas.unsqueeze(1))
            encoded_datas.append(encoder_data)
            original_data.append(batch)

    encoded_datas = torch.cat(encoded_datas, dim=0)
    original_data = torch.cat(original_data, dim=0)

    with torch.no_grad():
        decoded_data = autoencoderModel.decoder(encoded_datas)

    # Flatten decoded_data to 2D array
    decoded_data_flat = decoded_data.cpu().numpy().reshape(-1, decoded_data.size(-1))

    decoded_df = pd.DataFrame(decoded_data_flat, columns=None, index=None)

    def transform_label(value):
        return 1 if value > 0.5 else 0

    decoded_df[34] = decoded_df[34].apply(lambda x: transform_label(x))

    decoded_df.to_csv('../data/ae_data.csv', index=False)

ae_data('../data/original_data.csv')
