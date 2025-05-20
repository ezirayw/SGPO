from .base import Algo
import torch
import tqdm

from models.d3pm import D3PM
from models.continuous import ContinuousModel
from models.causalLM import CausalLM
from models.mdlm import MDLM

class Uncond(Algo):
    def __init__(self, net, forward_op, device='cuda', temp=1., top_p=1., data_config=None):
        super().__init__(net, forward_op, data_config=data_config, device=device)
        self.device = device

        #language model-specific parameters
        self.temp = temp
        self.top_p = top_p

    def inference(self, seq_len, num_samples=1, verbose=True, detokenize=False, project=False):
        """
        Inference for a single batch.
        """
        #inference with language model
        if type(self.net) == CausalLM:
            detokenized = self.net.sample(num_return_sequences=num_samples, temperature=self.temp, top_p=self.top_p)
            return None, detokenized #currently does not support tokenized samples
        else:
            #inference with continuous model
            if type(self.net) == ContinuousModel:
                ### TODO: complete based on code modified from https://github.com/ngruver/NOS/blob/569f1c85adf2ca2ea5e32efdf46276995fcf322c/seq_models/sample.py#L75 ###
                seq_len = seq_len # was previously hard-coded
                infill_seed = torch.randint(0, self.net.model.network.vocab_size, (seq_len,)).to(self.device) # random seed of token ids for now
                # 1 if != pad, else 0
                infill_mask = (torch.ones(seq_len) != self.net.tokenizer.pad_id-100).to(self.device) # switch 30 for self.net.tokenizer.pad_id
                # corrupt_mask: 1 for real tokens, 0 for pad (Equivalent to "fully corrupt all real tokens")
                corrupt_mask = infill_mask.clone().to(self.device)  # (B, L), 1=corrupt, 0=pad

                # Define the guidance_kwargs variable
                # guidance_kwargs = {
                #     "step_size": 0.1,
                #     "stability_coef": 1e-2,
                #     "num_steps": 2
                # }

                x = self.net.sample(
                    infill_seed=infill_seed,
                    infill_mask=infill_mask,
                    corrupt_mask=corrupt_mask,
                    num_samples=num_samples,
                )
                #convert to torch array of float
                x = torch.tensor(x, dtype=torch.float)
                #print(x.shape)

            #inference with D3PM model
            elif type(self.net) == D3PM:
                batch_size = num_samples
                timesteps = torch.linspace(self.net.timestep-1,1,int((self.net.timestep-1)/1), dtype=int) # iterate over reverse timesteps
                timesteps = tqdm.tqdm(timesteps) if verbose else timesteps
                x = self.net.get_start(batch_size)
                for t in timesteps:
                    x = self.net.p_sample(x, t)
            elif type(self.net) == MDLM:
                x = self.net.sample(num_samples)
            else:
                raise ValueError(f"Model {type(self.net)} not supported for unconditional sampling")
            
            if detokenize:
                detokenized = [self.net.tokenizer.untokenize(s) for s in x]
                return x, detokenized
            else:
                return x
