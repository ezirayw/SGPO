import hydra
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import transformers

from models.pretraining.trainer import BaseModel
from models.pretraining.model.progen2.model import ProGenForCausalLM
from models.pretraining.model.progen2.tokenizer import get_tokenizer

class CausalLMPretraining(BaseModel):
    """
    Causal language model using progen. Inherits from pl.LightningModule.
    """
    #TODO: uses src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
    #outputs loss and any other relevant outputs
    def __init__(self, model_config, network, tokenizer, optimizer, lr_scheduler):
        """
        Initializes the Progen2 model for finetuning with pytorch lightning.
        model_config: Hydra config object
        network: ProGenForCausalLM model
        tokenizer: Tokenizer object
        optimizer: Optimizer object
        lr_scheduler: Learning rate scheduler object
        device: torch.device
        """
        super().__init__()
        #seems like you don't need to set device here
        #self.device = torch.device(model_config.device)

        if model_config.pretrained_ckpt is not None:
           self.network = ProGenForCausalLM.from_pretrained("jsunn-y/ProCALM", subfolder=model_config.pretrained_ckpt, cache_dir=model_config.cache_dir) 
           #config=config #cache_dir=config.pretrained_model_dir
           self.network.train()

        self.tokenizer = get_tokenizer()

        self.opt = hydra.utils.instantiate(optimizer, params=self.network.parameters())

        self.lr_scheduler = None
        if lr_scheduler:
            self.lr_scheduler = hydra.utils.instantiate(lr_scheduler, self.opt)
    
    def forward(self, batch):
        """
        Forward pass of the model.
        batch: dictionary of tensors
        """
        #loop through everything in the batch dictionary and move to device
        input_ids, labels, attention_mask  = batch["input_ids"], batch["labels"], batch["attention_mask"]
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.network(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        out = {"loss": outputs.loss}
        return out

        