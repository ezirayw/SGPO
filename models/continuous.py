import os
import torch
import torch.nn.functional as F
from .base import GenerativeModel
from models.pretraining.model.continuous_diffusion import GaussianDiffusion


class ContinuousModel(GenerativeModel):
    def __init__(self, model_name, seq_len, device='cuda'):
        """
        A wrapper for a continuous diffusion model.
        Loads the GaussianDiffusion model from a checkpoint.
        """
        super().__init__()
        self.model_name = model_name
        self.seq_len = seq_len
        self.device = device

        best_checkpoint = os.path.join("checkpoints", model_name, "best_model.ckpt")
        self.model = GaussianDiffusion.load_from_checkpoint(best_checkpoint)
        if self.model is None:
            raise ValueError("Must implement logic to load or instantiate GaussianDiffusion model.")

        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer

    @property
    def noise_schedule(self):
        """Expose the underlying model's noise_schedule for compatibility."""
        return self.model.noise_schedule
    @property
    def timestep(self):
        return self.model.noise_schedule.timesteps

    @torch.no_grad()
    def get_embeds(self, x):
        """
        Given a batch of token IDs, return the continuous embeddings
        using the underlying model's get_embeds method.
        """
        x = x.to(self.device)
        embeds = self.model.network.get_embeds(x)
        return embeds
    @torch.no_grad()
    def sample(self,
               infill_seed=None,
               infill_mask=None,
               corrupt_mask=None,
               num_samples=1,
               guidance_kwargs=None,
               bad_word_ids=None):
        """
        Wraps the model's own sampling function.
        """
        if self.tokenizer.__class__.__name__ == "ESMTokenizer":
            if bad_word_ids is None and hasattr(self.tokenizer, "alphabet"):
                bad_word_ids = [
                    i for i, tok in enumerate(self.tokenizer.alphabet.all_toks)
                    if tok not in list("ACDEFGHIKLMNPQRSTVWY")  # standard 20 AAs
                ]
        tokenized = self.model.sample(
            infill_seed=infill_seed,
            infill_mask=infill_mask,
            corrupt_mask=corrupt_mask,
            num_samples=num_samples,
            guidance_kwargs=guidance_kwargs,
            bad_word_ids=bad_word_ids,
        )
        return tokenized

    @torch.no_grad()
    def guided_sample(self,
                      infill_seed,
                      infill_mask,
                      corrupt_mask,
                      num_samples,
                      classifier,  # The classifier function/nn.Module
                      guidance_scale=1.0,  # Guidance strength λ
                      bad_word_ids=None):
        """
        Wraps the model's guided sampling function.
        """

        if self.tokenizer.__class__.__name__ == "ESMTokenizer":
            if bad_word_ids is None and hasattr(self.tokenizer, "alphabet"):
                bad_word_ids = [
                    i for i, tok in enumerate(self.tokenizer.alphabet.all_toks)
                    if tok not in list("ACDEFGHIKLMNPQRSTVWY")  # standard 20 AAs
                ]

        tokenized = self.model.guided_sample(
            infill_seed=infill_seed,
            infill_mask=infill_mask,
            corrupt_mask=corrupt_mask,
            num_samples=num_samples,
            classifier=classifier,
            guidance_scale=guidance_scale,
            bad_word_ids=bad_word_ids,
        )
        return tokenized

    def NOS_C_sample(self,
               infill_seed=None,
               infill_mask=None,
               corrupt_mask=None,
               num_samples=1,
               classifier = None,
               guidance_kwargs=None,
               bad_word_ids=None):
        """
        Wraps the model's own sampling function.
        """
        if self.tokenizer.__class__.__name__ == "ESMTokenizer":
            if bad_word_ids is None and hasattr(self.tokenizer, "alphabet"):
                bad_word_ids = [
                    i for i, tok in enumerate(self.tokenizer.alphabet.all_toks)
                    if tok not in list("ACDEFGHIKLMNPQRSTVWY")  # standard 20 AAs
                ]

        tokenized = self.model.NOS_C_sample(
            infill_seed=infill_seed,
            infill_mask=infill_mask,
            corrupt_mask=corrupt_mask,
            num_samples=num_samples,
            classifier=classifier,
            guidance_kwargs=guidance_kwargs,
            bad_word_ids=bad_word_ids,
        )
        return tokenized

    def tweedie(self, x, sigma):
        infill_mask = (torch.ones(self.seq_len) != self.tokenizer.pad_id-100).to(self.device)  # switch 30 for self.net.tokenizer.pad_id
        # infill_mask = infill_mask * self.mask
        attn_mask = torch.ones((x.shape[0], self.seq_len),dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # t = self.noise_schedule.sigma_inv(sigma)
            idx = (torch.abs(self.noise_schedule.sigmas.to(self.device) - sigma)).argmin()
            t = torch.stack([idx] * x.shape[0])
            f_out = self.model.network.forward(x/(sigma**2 + 1).sqrt(), t, attn_mask=attn_mask)
            out = self.model.network.pred_xstart(
                x,
                t,
                attn_mask=attn_mask,
                sequence_output=f_out['sequence_output'],
                infill_mask=infill_mask
            )
            x0 = out['xstart']
            probs = out['probs']
        return x0, probs
    
    def score(self, x, t):
        """
        For consistency with the discrete diffusion wrapper.
        Here you might compute the score function (i.e. the gradient of log p(x))
        but for classifier training this may not be used. For now, we return a dummy tensor.
        """
        d, probs = self.tweedie(x, t)
        return (d - x) / t ** 2

    def pred_mean(self, x, t):
        """
        For consistency: predict a “mean” from the noised input.
        Here we use the underlying network’s prediction of x₀.
        Note: the underlying network (e.g. GaussianDiffusionTransformer) has a
        method `pred_xstart` that returns a dict with key "xstart".
        """
        B, seq_len = x.shape[0], x.shape[1]
        # Create a default attention mask (all ones)
        attn_mask = torch.ones(B, seq_len, device=self.device, dtype=torch.bool)
        # Call the underlying network’s pred_xstart; you can pass additional args if needed.
        out = self.model.network.pred_xstart(x, t, attn_mask=attn_mask)
        return out["xstart"]

    def get_start(self, batch_size):
        """
        Returns a starting point for sampling.
        For a continuous diffusion model this can be Gaussian noise.
        The noise dimension is assumed to match the input channel dimension of the network.
        """
        # Assume the underlying network (e.g. GaussianDiffusionTransformer)
        # has an attribute "in_channels" that is the embedding dimension.
        in_channels = self.model.network.in_channels
        noise = torch.randn(batch_size, self.seq_len, in_channels, device=self.device)
        return noise

    def q_sample(self, x, t):
        """
        Given a batch of discrete token sequences x (shape: [B, seq_len]) and time steps t,
        perform the forward (noising) process:
          1. Convert tokens to continuous embeddings via the underlying network’s get_embeds.
          2. Use the noise schedule’s q_sample method to add noise.
        Returns the noised samples (typically a continuous tensor).
        """
        # x: (B, seq_len) token IDs
        embeds = self.model.network.get_embeds(x)  # (B, seq_len, embed_dim)
        # Use the underlying noise schedule’s q_sample to add noise at time t.
        x_t = self.model.noise_schedule.q_sample(embeds, t)
        return x_t