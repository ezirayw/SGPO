### Written based on https://github.com/microsoft/evodiff/blob/main/evodiff/pretrained.py ###
import json
import os
import torch
from sequence_models.constants import MSA_ALPHABET
from evodiff.utils import Tokenizer
from evodiff.collaters import D3PMCollater
from models.pretraining.model.d3pm_evodiff import ByteNetDiffusion

def load_sequence_checkpoint(model_name):
    
    best_checkpoint = os.path.join("checkpoints", model_name, "best_model.ckpt")
    lightning_model = ByteNetDiffusion.load_from_checkpoint(best_checkpoint)
    model = lightning_model.network
    tokenizer = lightning_model.tokenizer

    # checkpoint = torch.load("path/to/your_model.ckpt")
    # state_dict = checkpoint['state_dict']
    # model = ByteNetDiffusion.load_from_checkpoint(best_checkpoint)

    return model, tokenizer

def D3PM_FINETUNED_38M(model_name, return_all=False):
    model, tokenizer = load_sequence_checkpoint(model_name)
    dt = model.embedder.timesteps
    Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=dt)
    collater = D3PMCollater(tokenizer=tokenizer, num_timesteps=dt, Q=Q_t, Q_bar=Q_prod)
    scheme = 'd3pm'
    if return_all:
        return model, collater, tokenizer, scheme, dt, Q_prod, Q_t
    else:
        return model, collater, tokenizer, scheme