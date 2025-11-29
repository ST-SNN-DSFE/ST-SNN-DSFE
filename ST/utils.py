import warnings
import os
import torch.nn as nn
from torch.nn import functional as F
import torch
import datetime
import logging
import shutil
import numpy as np
import einops

class CrossInteraction(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossInteraction, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=3, batch_first=True)
        self.fc = nn.Linear(in_dim, out_dim)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x1, x2):
        attn_output, _ = self.attention(x1, x2, x2)
        out = self.lrelu(self.fc(attn_output))
        return out

################ noise ################
def add_gaussian_noise(data, mean=0.0, std=0.005):
    noise = torch.randn_like(data) * std + mean
    return data + noise

def add_poisson_noise(data, scale=0.05):
    noise = torch.poisson(torch.abs(data) * scale) / scale
    return data + noise

def add_white_noise(data, snr=40):
    signal_power = torch.mean(data ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise

def add_salt_pepper_noise(data, prob=0.005):
    noise = torch.rand_like(data)
    data[noise < (prob / 2)] = 0
    data[noise > (1 - prob / 2)] = 1

def add_noise(data, noise_type="gaussian", snr=40, scale=0.05, prob=0.005):
    if noise_type == "gaussian":
        return add_gaussian_noise(data)
    elif noise_type == "poisson":
        return add_poisson_noise(data, scale=scale)
    elif noise_type == "white":
        return add_white_noise(data, snr=snr)
    elif noise_type == "salt_pepper":
        return add_salt_pepper_noise(data, prob=prob)
    else:
        raise ValueError(f"No noise: {noise_type}")
################ noise ################

class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=4, epsilon=0.14, ):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



def set_logging_config(logdir):
    def beijing(sec, what):
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()


    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.Formatter.converter = beijing

    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, ("log.txt"))),
                                  logging.StreamHandler(os.sys.stdout)])


def save_checkpoint(state, is_best, dir, subject_name):
    torch.save(state, os.path.join(f'{dir}', f'{subject_name}_checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(f'{dir}', f'{subject_name}_checkpoint.pth.tar'),
                        os.path.join(f'{dir}', f'{subject_name}_model_best.pth.tar'))


def normalize_adj(adj):

    D = torch.diag(torch.sum(adj, dim=1))
    D_ = torch.diag(torch.diag(1 / torch.sqrt(D))) # D^(-1/2)
    lap_matrix = torch.matmul(D_, torch.matmul(adj, D_))

    return lap_matrix


class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, subject_name, val_acc, model, epoch):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(subject_name, val_acc, model, epoch)
        elif score <= self.best_score + self.delta:
            pass
        else:
            self.best_score = score
            self.save_checkpoint(subject_name, val_acc, model, epoch)
            self.counter = 0

    def save_checkpoint(self, subject_name, val_acc, model, epoch):
        '''Saves model when validation acc increase.'''
        if self.verbose:
            self.trace_func(f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}) in epoch ({epoch}).  Saving model ...')
        model_save_path = self.path + subject_name + ".pt"
        torch.save(model.state_dict(), model_save_path)
        self.val_acc_max = val_acc