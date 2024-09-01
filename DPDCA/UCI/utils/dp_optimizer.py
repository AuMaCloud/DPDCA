# -*- coding: utf-8 -*-
import os
import torch
from torch.nn.utils import clip_grad_norm_

from torch.optim import AdamW


#############################################
## 1. Generate a random Secure generator ####
#############################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_secure_seed(secure_seed=None):
    if secure_seed is not None:
        return secure_seed
    else:
        return int.from_bytes(os.urandom(8), byteorder="big", signed=True)

def set_random_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

secure_generator = set_random_seed(set_secure_seed(), device)

#########################
## 2. Generate noise ####
#########################
def generate_noise(noise_intensity, max_norm, parameters):

    if noise_intensity > 0:
        return torch.normal(
            0,
            noise_intensity * max_norm,
            parameters.grad.shape,
            device=device,
            generator=secure_generator,
        )

    return 0.0

##########################
## 3. optimizer of DP ####
##########################
class optimizerDP(AdamW):
    def __init__(self, max_per_sample_grad_norm, noise_multiplier, batch_size, *args, **kwargs):
        super(optimizerDP, self).__init__(*args, **kwargs)

        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size

        # Initialize aggregate gradients
        self.aggregate_grads = []
        for param in self.param_groups[0]['params']:
            if param.requires_grad:
                self.aggregate_grads.append(torch.zeros_like(param.data))
            else:
                self.aggregate_grads.append(None)

    def clip_grads_(self):
        # Clip gradients in-place
        params = self.param_groups[0]['params']
        clip_grad_norm_(params, max_norm=self.max_per_sample_grad_norm, norm_type=2)

        # Accumulate gradients
        for group in self.param_groups:
            for param, accum_grad in zip(group['params'], self.aggregate_grads):
                if param.requires_grad:
                    accum_grad.add_(param.grad.data)

    def zero_grad(self):
        # Zero gradients for aggregate gradients
        for accum_grad in self.aggregate_grads:
            if accum_grad is not None:
                accum_grad.zero_()

    def add_noise_(self):
        for group in self.param_groups:
            for param, accum_grad in zip(group['params'], self.aggregate_grads):
                if param.requires_grad:
                    # Accumulate gradients
                    param.grad.data = accum_grad.clone()
                    # Add noise and update grads
                    noise = generate_noise(self.noise_multiplier, self.max_per_sample_grad_norm, param)
                    param.grad += noise / self.batch_size

    def step(self, *args, **kwargs):
        super(optimizerDP, self).step(*args, **kwargs)


def create_optimizer(max_per_sample_grad_norm, noise_multiplier, batch_size, *args, **kwargs):
    optimizer = optimizerDP(max_per_sample_grad_norm, noise_multiplier, batch_size, *args, **kwargs)
    return optimizer

# AdamWDP = create_optimizer(AdamW)

