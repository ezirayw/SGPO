### Modified from https://github.com/Profluent-Internships/ProCALM/blob/main/runner.py ###
import os
from .base import GenerativeModel
import torch.nn.functional as F
import torch
import tqdm
from typing import Optional, Callable, Literal
import numpy as np
import random
from tokenizers import Tokenizer
from models.pretraining.model.progen2.model import ProGenForCausalLM
from models.pretraining.model.progen2.tokenizer import get_tokenizer, PAD_TOKEN_ID

class CausalLM(GenerativeModel): #should change inheritense eventually
    
    def __init__(self, model_name, seq_len, load_ref_model=False, checkpoint_name="best", device="cuda") -> None:
        #load the training config to determine how to load the trained model
        self.model_name = model_name
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.max_seq_len = int(seq_len*1.25) #only generate slightly longer ones
        
        #load a model with conditional adapters
        self.model_dir = os.path.join("checkpoints", model_name, checkpoint_name)

        if os.path.exists(os.path.join(self.model_dir, "model.safetensors")):
            #if there is a local safetensors file to load
            self.model = ProGenForCausalLM.from_pretrained(self.model_dir).to(device)
            self.tokenizer = get_tokenizer()
        else:
            print("No model found at", self.model_dir)
        
        #only load this for DPO to save memory
        if load_ref_model:
            self.ref_model = ProGenForCausalLM.from_pretrained(self.model_dir).to(device)

        self.model_config = self.model.config
        self.pad_token_id = PAD_TOKEN_ID
        
        # np.random.seed(42)
        # random.seed(42)
        # torch.manual_seed(42)
        # torch.cuda.manual_seed(42)
        # torch.cuda.manual_seed_all(42)
        # torch.backends.cudnn.deterministic = True
    
    def clean(self, sample):
        """
        Remove 1 and 2 from the sample.
        """
        return sample.replace('1', '').replace('2', '')

    def truncate(self, sample, terminals):
        """
        Truncates a sequence between the correct start and end tokens, else do nothing.
        """
        pos = []
        for terminal in terminals:
            find_pos = sample.find(terminal, 1)
            if find_pos != -1:
                pos.append(find_pos)
        if len(pos) > 0:
            return sample[:(min(pos)+1)]
        else:
            return sample
        
    
    def sample(
        self,
        context= '1', #start token
        num_return_sequences=40, #effectively the batch size
        temperature=1,
        top_p=0.95, #0.9 or 0.95
    ):
        """
        Runs one batch of generation with the specified conditions.
        """

        # def compute_total_weight_difference(model1, model2):
        #     total_diff = 0.0
        #     for param1, param2 in zip(model1.parameters(), model2.parameters()):
        #         total_diff += torch.sum(torch.abs(param1 - param2)).item()
        #     print(f"Total weight difference: {total_diff:.6f}")

        # # Example usage to check if model weights are being updated
        # compute_total_weight_difference(self.model, self.ref_model)

        self.model.eval()
        # for param in self.model.parameters():
        #     print(param)  # Print the full weight tensor
        #     break  # Stop after the first parameter

        self.temp = 'temp' + str(temperature)

        #running things packaged into the huggingface class (alternatively could use beam search instead of probabilistic decoding)
        with torch.no_grad():
            input_ids = torch.tensor(self.tokenizer.encode(context).ids).view([1, -1]).to(self.device)

            tokens_batch = self.model.generate(input_ids=input_ids, do_sample=True, temperature=temperature, max_length=self.max_seq_len, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=self.pad_token_id, eos_token_id=4) #self.seq_len

            as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
            self.sequences = self.tokenizer.decode_batch(as_lists(tokens_batch))
        
        self.sequences = [self.clean(self.truncate(seq, ['1', '2'])) for seq in self.sequences]

        return self.sequences


