import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


##############################
# Time Embedding Module
##############################

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Use half as many parameters for sin and cos
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x: tensor of shape (B,) containing timesteps
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi  # shape: (B, embed_dim/2)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # shape: (B, embed_dim)


##############################
# Embedded Token-Based Regression Model
##############################

class EmbeddedMLPRegressor(nn.Module):
    """
    An MLP regressor for predicting protein fitness directly from
    embedded tokens.

    Args:
      embed_dim: Dimension of the embedded tokens.
      hidden_dim: Hidden dimension used inside the MLP.
      seq_length: The (fixed) length of the protein sequence.
      residues: (Optional) list/array of residue indices (0-indexed) to select.
                If provided, only these positions are used and the effective
                sequence length is set to len(residues).  # CHANGED: Added residues.
    """
    def __init__(self, embed_dim, hidden_dim, seq_length, mean_pool=False, residues=None):  # CHANGED: Added residues parameter.
        super().__init__()
        self.embed_dim = embed_dim
        self.mean_pool = mean_pool
        # CHANGED: If residues are provided, update the effective sequence length.
        if residues is not None:
            self.residues = residues  # store the indices (assumed to be 0-indexed)
            self.seq_length = len(residues)
        else:
            self.residues = None
            self.seq_length = seq_length
        # Time embedder: maps a scalar timestep to a vector (hidden_dim)
        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embed_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        #TODO: add some sort of mean pooling for ESM embeddings?
        
        # Process the flattened embedded sequence plus the time embedding.
        # Flattened sequence has dimension: seq_length * embed_dim.
        input_dim = embed_dim + hidden_dim if self.mean_pool else self.seq_length * embed_dim + hidden_dim 
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Final head: output a single value per example (fitness)
        self.regressor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embedded_seq, t):
        """
        Args:
          embedded_seq: Tensor of shape (B, full_seq_length, embed_dim) – the embedded tokens.
                        If residues are provided, only those positions will be selected.
          t: Tensor of shape (B,) containing the current diffusion timestep.
        Returns:
          A tensor of shape (B, 1) containing the predicted fitness.
        """
        # CHANGED: If residues are provided, select only the specified positions.
        if self.residues is not None:
            embedded_seq = embedded_seq[:, self.residues, :]
        B, L, D = embedded_seq.shape
        assert L == self.seq_length and D == self.embed_dim, "Input shape does not match model configuration."

        # Option 1: Flatten the embedded sequence.
        if self.mean_pool:
            flat_seq = embedded_seq.mean(dim=1)
        else:
            flat_seq = embedded_seq.view(B, -1)  # shape: (B, seq_length * embed_dim)
        #instead mean pool the embedding
        

        # Get time embedding (ensure t is float and on the same device)
        t = t.float()
        time_emb = self.time_embedder(t)  # shape: (B, hidden_dim)

        # Concatenate flattened sequence and time embedding.
        combined = torch.cat([flat_seq, time_emb], dim=-1)  # shape: (B, seq_length * embed_dim + hidden_dim)

        # Process through MLP and produce the regression output.
        feat = self.mlp(combined)
        out = self.regressor_head(feat)  # shape: (B, 1)
        return out


##############################
# Wrapper for Classifier Guidance (Regression)
##############################

class ProteinPredictorContinuous(nn.Module):
    """
    Protein regressor for classifier guidance that operates on embedded tokens.
    Designed to predict fitness from the embedded protein sequence and timestep.

    Args:
      data_config: configuration with keys such as full_seq and residues.
      model_config: configuration with attributes embed_dim, hidden_dim, and seq_length.
      checkpoint: (Optional) Path to a checkpoint to load model weights.
      device: Device to run the model on.
      **kwargs: Additional keyword arguments (e.g. data_config) that can be ignored.
    """
    def __init__(self, data_config, model_config, checkpoint=None, device='cuda', **kwargs):
        super().__init__()
        self.device = device
        self.data_config = data_config

        # CHANGED: Extract residues from data_config (assumed to be provided as a list of 1-indexed positions)
        if data_config.get("residues") is not None:
            residues = [i - 1 for i in data_config["residues"]]  # convert to 0-indexing
            effective_seq_length = len(residues)  # effective sequence length is the number of mutated residues
        else:
            residues = None
            effective_seq_length = data_config.seq_len

        # Instantiate the embedded MLP regressor with residue selection.
        self.model = EmbeddedMLPRegressor(model_config.embed_dim, model_config.hidden_dim, effective_seq_length, model_config.mean_pool, residues=residues)
        self.loss_fn = nn.MSELoss()

        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))
        self.model.to(device)

    def __call__(self, inputs, t):
        """
        Args:
          inputs: Tensor of shape (B, full_seq_length, embed_dim) containing embedded tokens.
                  The model will select mutated positions if configured.
          t: Tensor of shape (B,) containing timesteps.
        Returns:
          Predicted fitness values (B, 1).
        """
        return self.model(inputs, t)

    def loss(self, inputs, t, y):
        """
        Computes the regression loss given the inputs, timesteps, and target fitness y.
        Args:
          inputs: Tensor of shape (B, full_seq_length, embed_dim).
          t: Tensor of shape (B,).
          y: Ground-truth fitness values of shape (B, 1) (or (B,) if squeezed later).
        Returns:
          Scalar loss.
        """
        preds = self.model(inputs, t)
        return self.loss_fn(preds, y)

    def log_likelihood(self, inputs, y=None):
        return self(inputs, torch.zeros(inputs.shape[0],device=inputs.device)).squeeze(-1)

##############################
# Example Usage
##############################

if __name__ == "__main__":
    BATCH_SIZE = 4
    FULL_SEQ_LENGTH = 15  # For example, the full sequence has 15 positions.
    EMBED_DIM = 512
    HIDDEN_DIM = 256

    # Create dummy embedded tokens (simulate outputs from your diffusion model's embedding layer)
    # Input shape: (BATCH_SIZE, FULL_SEQ_LENGTH, EMBED_DIM)
    dummy_embeds = torch.randn(BATCH_SIZE, FULL_SEQ_LENGTH, EMBED_DIM).to("cuda")
    # Dummy timesteps (e.g., current diffusion timesteps)
    dummy_t = torch.randint(0, 1000, (BATCH_SIZE,)).to("cuda")
    # Dummy target fitness values (e.g., from your CSV's "fitness" column)
    dummy_fitness = torch.randn(BATCH_SIZE, 1).to("cuda")  # Replace with your actual targets

    # Create a data configuration similar to the discrete classifier.
    # Here "residues" are provided as a list of 1-indexed positions.
    data_config = {
        "full_seq": "AAAAAAAAAAAAAAA",  # Dummy full sequence (15 characters)
        "residues": [3, 5, 8]            # Only positions 3, 5, and 8 (1-indexed) will be used.
    }
    # Simulate a model configuration object.
    class ModelConfig:
        pass
    model_config = ModelConfig()
    model_config.embed_dim = EMBED_DIM
    model_config.hidden_dim = HIDDEN_DIM
    model_config.seq_length = FULL_SEQ_LENGTH  # This is the full sequence length; the effective length will be adjusted.

    # Instantiate the regressor.
    regressor = ProteinPredictorContinuous(data_config, model_config, device="cuda")
    # Forward pass.
    preds = regressor(dummy_embeds, dummy_t)
    loss = regressor.loss(dummy_embeds, dummy_t, dummy_fitness)
    print("Predicted fitness:", preds)
    print("Loss:", loss)
