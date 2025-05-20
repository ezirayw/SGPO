from .base import BaseOperator
import numpy as np
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.models import SingleTaskGP

class ProteinPredictor(nn.Module):
    '''
    Protein predictor class used for classifier guidance 
    '''

    def __init__(self, data_config, model_config, checkpoint=None, device='cuda'):
        super().__init__()
        '''
        Load predictor model
        '''
        self.device=device
        self.model = MLPModel(data_config, model_config)
        self.loss_fn = nn.CrossEntropyLoss()
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))
        self.model.to(device)

    # def update_model(self, classifier):
    #     self.model = classifier
    #     self.model.to(self.device)

    def __call__(self, inputs, t):
        """
        inputs: (B, D)
        t: (B)
        """
        return self.model(inputs, t)
    
    def log_likelihood(self, inputs, y=None):
        return self(inputs, torch.zeros(inputs.shape[0],device=inputs.device)).squeeze(-1)


### Adapted from https://github.com/HannesStark/dirichlet-flow-matching/tree/main/model and https://github.com/hnisonoff/discrete_guidance/blob/main/applications/enhancer/models.py ###

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class MLPModel(nn.Module):
    """
    MLP model for classifying protein fitness.
    TODO: implement as  a regressor instead and perform classification after?
    """
    def __init__(self, data_config, model_config, time_conditioned=True):
        super().__init__()
        self.alphabet_size = data_config.alphabet_size
        #self.alphabet_size = 31
        # self.residues = np.ndarray(data_config.residues)

        if data_config.residues is not None:
            self.residues = np.array(data_config.residues)
            self.n_residues = len(self.residues)
        else:
            self.residues = None
            self.n_residues = len(data_config.full_seq)
        
        self.hidden_dim = model_config.hidden_dim
        self.time_conditioned = time_conditioned
        
        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embed_dim=self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim))
        self.embedder = nn.Linear(self.alphabet_size, self.hidden_dim) #* self.n_residues
        
        input_dim = self.hidden_dim*(self.n_residues+1) if time_conditioned else self.hidden_dim*self.n_residues
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), #why is there +1 here
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.cls_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim, 1)) #regreession

    def forward(self, 
                seq: torch.tensor = None,
                t: torch.tensor = None):
        """
        Args:
            seq: Input sequence with tokens as integers, shape (B, D)
            t: Input time, shape (B)
        """

        # (B,D), (C) -> (B,C)
        #select only the mutated indices of the sequence
        if self.residues is not None:
            seq = seq[:, self.residues]

        seq_encoded = F.one_hot(seq.long(), num_classes=self.alphabet_size).float()
        #print(seq_encoded.shape)
        #seq_encoded = seq_encoded.reshape(seq_encoded.shape[0], -1) #flatten the onehot encoding
        feat = self.embedder(seq_encoded)
        if self.time_conditioned:
            time_embed = self.time_embedder(t)
            # feat = feat + time_embed[:,None,:] #sum the feature and time embeddings
            feat = torch.cat([feat.view(feat.size(0), -1), time_embed], dim=-1)
        feat = self.mlp(feat)
        # return self.cls_head(feat.mean(dim=1)) #mean across the sequence length
        return self.cls_head(feat)

#TODO: probably easier to just write an embedding module separate from the GP
class Embedder(nn.Module):
    def __init__(self, data_config, model_config, time_conditioned=True, device="cuda"): 
        super().__init__()
        self.device = device
        self.time_conditioned = time_conditioned

        #could make this requires grad=False if not training
        self.alphabet_size = data_config.alphabet_size
        self.hidden_dim = self.alphabet_size + 1 #default to this for now, but can play around with this as necessary #also not sure why this is one more

        if data_config.residues is not None:
            self.residues = np.array(data_config.residues)
            self.n_residues = len(self.residues)
        else:
            self.residues = None
            self.n_residues = len(data_config.full_seq)
        
        if time_conditioned:
            self.time_embedder = GaussianFourierProjection(embed_dim=self.hidden_dim)
            self.embedder = nn.Linear(self.alphabet_size, self.hidden_dim) #this doesn't really do anything too interesting 

    def forward(self, seqs, t):
        """
        Args:
            seq: Input sequence with tokens as integers, shape (B, D)
            t: Input time, shape (B)
        """

        # (B,D), (C) -> (B,C)
        #select only the mutated indices of the sequence
        if self.residues is not None:
            seqs = seqs[:, self.residues]

        x = F.one_hot(seqs.long(), num_classes=self.alphabet_size).float()
        #print(seq_encoded.shape)
        #seq_encoded = seq_encoded.reshape(seq_encoded.shape[0], -1) #flatten the onehot encoding
        if self.time_conditioned:
            feat = self.embedder(x)
            time_embed = self.time_embedder(t)
            # feat = feat + time_embed[:,None,:] #sum the feature and time embeddings
            x = feat + time_embed[:,None,:] 
        #flatten
        x = x.flatten(start_dim=-2)
        return x

class SampledDeterministicModel(nn.Module): #alternatively could inherit from ProteinPredictor
    """
    Wrapper for deterministic models sampled from GP, to behave like nn.Module.
    """
    def __init__(self, sampled_model, data_config, model_config, time_conditioned=True):
        super().__init__()
        self.sampled_model = sampled_model  # this is already deterministic from the GP
        self.embedder = Embedder(data_config, model_config, time_conditioned)

    def forward(self, seq, t):
        x = self.embedder(seq, t)
        x = x.unsqueeze(1)
        #posterior = self.sampled_model.posterior(x)
        #return posterior.mean  # or posterior.rsample() if you want noisy evaluations
        return self.sampled_model(x)

    def __call__(self, inputs, t):
        """
        inputs: (B, D)
        t: (B)
        """
        return self.forward(inputs, t)
    
    def log_likelihood(self, inputs, y=None):
        return self(inputs, torch.zeros(inputs.shape[0],device=inputs.device)).squeeze(-1)
        