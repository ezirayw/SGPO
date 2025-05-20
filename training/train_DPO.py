import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
import os
import wandb
import copy

### Modified from https://github.com/AI4PDLab/DPO_pLM/blob/main/DPO_pLM.py#L105 ###

# ---------------------------
# Loss Functions
# ---------------------------
def log_likelihood(batch, model, device):
    
    input_ids, labels, attention_mask  = batch["input_ids"], batch["labels"], batch["attention_mask"]
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)  # output needs to be the log-likelihood averaged over the number of tokens. 
    #all_log_likelihood = -1 * outputs.loss #this is averaged cross entropy loss over the whole batch
    #TODO: convert to log-likelihood and average over the number of tokens, instead #
    log_likelihoods = outputs.log_likelihoods
    return log_likelihoods

def dpo_paired_loss(batch, model, ref_model, tokenizer, device, beta=0.1):
    """
    Calculates the paired DPO loss.
    """
    # Extract positive and negative sequences
    positive_sequence = batch["positive_sequence"]
    negative_sequence = batch["negative_sequence"]

    # Log probabilities for positive sequences
    pos_ref_log_probs = log_likelihood(positive_sequence, device, ref_model, tokenizer)
    pos_policy_log_probs = log_likelihood(positive_sequence, device, model, tokenizer)
    pos_ratios = beta * (pos_policy_log_probs - pos_ref_log_probs)

    # Log probabilities for negative sequences
    neg_ref_log_probs = log_likelihood(negative_sequence, device, ref_model, tokenizer)
    neg_policy_log_probs = log_likelihood(negative_sequence, device, model, tokenizer)
    neg_ratios = beta * (neg_policy_log_probs - neg_ref_log_probs)

    # Compute the DPO paired loss
    loss = -F.logsigmoid(pos_ratios - neg_ratios)

    return  torch.mean(loss)
    
def dpo_weighted_loss(pi_log_likelihood, ref_log_likelihood, weights, beta=0.1):
    """
    Calculates DPO weighted loss. 
    Function kindly provided by Widatalla et.al 2024 "Aligning protein 
    generative models with experimental fitness via Direct Preference Optimization"
    """
    if ref_log_likelihood is None:
        pi_ratio = beta * pi_log_likelihood
    else:
        pi_ratio = beta * (pi_log_likelihood - ref_log_likelihood)
    
    weights = torch.softmax(weights, dim=0)

    loss = F.cross_entropy(pi_ratio, weights)
    #loss = F.nll_loss(pi_ratio, weights)
    return loss

def dpo_ranked_loss(pi_log_likelihood, pi_ref_loglikelihood, weights, beta=0.1):
    """
    Calculates the Dynamic Policy Optimization (DPO) ranked loss.
    In this case the ranking is on the batch dimension.
    """
    # Ensure weights have at least one dimension
    weights = torch.softmax(weights, dim=0)
    weights = weights.view(-1)  
    
    sorted_indices = torch.argsort(weights, descending=True)
    pi_log_likelihood = pi_log_likelihood[sorted_indices]
    pi_ref_loglikelihood = pi_ref_loglikelihood[sorted_indices] if pi_ref_loglikelihood is not None else None
    weights = weights[sorted_indices]
    #print(f"Sorted weights: {weights}")

    if pi_ref_loglikelihood is not None:
        pi_ratio = beta * (pi_log_likelihood - pi_ref_loglikelihood)
    else:
        pi_ratio = beta * pi_log_likelihood

    uniform_weights = torch.ones_like(pi_ratio)
    #print(f"pi ratios: {pi_ratio}")

    
    loss = F.mse_loss(pi_ratio, uniform_weights)
    return loss


# ---------------------------
# Training and Evaluation
# ---------------------------
def save_model(model, output_dir):
    """
    Saves the model to a specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def train_epoch(model, ref_model, train_loader, optimizer, device, mode, beta, wandb_logging=True):
    """
    Performs training for one epoch.
    """
    model.train()
    total_loss = []
    for batch in train_loader:

        if mode != 'paired':
            optimizer.zero_grad()
            ref_log_probs = log_likelihood(batch, ref_model, device)
            policy_log_probs = log_likelihood(batch, model, device)

            weights = batch["weights"].to(device) #fitness values for preference alignment
            
            if mode == "weighted":
                loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, beta)
            
            if mode == "ranked":
                loss = dpo_ranked_loss(policy_log_probs, ref_log_probs, weights, beta)

        #currently not supported    
        # if mode == "paired":
        #     loss = dpo_paired_loss(batch, model, ref_model, tokenizer, device, CONFIG["beta"])
        
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss}) if wandb_logging else None
        total_loss.append(loss.item())
    
    torch.cuda.empty_cache()

    return sum(total_loss) / len(total_loss)

def train_DPO(net, train_config, train_loader, eval_loader=None):
    """
    Finetune the model with DPO.
    """
    model = net.model #should be linked to the original net as the gradients are updated
    ref_model = net.ref_model

    n_epochs = train_config.n_epochs
    device = train_config.device
    mode = train_config.mode

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=train_config.adam_betas,
        eps=train_config.epsilon,
        weight_decay=train_config.adam_decay,
    )

    wandb.init(project="discrete_diffusion", name="DPO_finetuning") if train_config.wandb else None

    for epoch in range(n_epochs):
        # epoch_loss = 0
        
        train_loss = train_epoch(model, ref_model, train_loader, optimizer, device, mode, train_config.beta, train_config.wandb)
        wandb.log({"train_epoch_loss": train_loss}) if train_config.wandb else None

        if eval_loader is not None:
            eval_loss = evaluate(model, ref_model, eval_loader, optimizer, device, mode, train_config.beta, train_config.wandb)

        # print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

        #TODO: implement model saving
        #save_model(model, output_dir=f"output_iteration{iteration_num}")
            
        # print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader)}")
        # if wandb:
        #     wandb.log({"epoch_loss": epoch_loss/len(dataloader)})
        
        # t = torch.zeros(x.shape[0], dtype=torch.long).to(classifier.device)
        # loss = train_step(classifier, model, optimizer, criterion, x, t, y, project_fn)
        # print(loss)

    #net.model = model
    if train_config.wandb:
        wandb.finish()
        
    return net

def evaluate(model, ref_model, tokenizer, eval_loader, optimizer, device, mode):
    """
    TODO: Update this function and create an eval loader for it.
    Evaluates the model on the evaluation set.
    """
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch in eval_loader:
            if mode != 'paired':
                optimizer.zero_grad()
                sequences = batch["sequence"] 
                ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
                policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
                weights = batch["weight"].to(device)
                
                if mode == "weighted":
                    loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
                
                if mode == "ranked":
                    loss = dpo_ranked_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
                
        if mode == "paired":
            loss = dpo_paired_loss(batch, model, ref_model, tokenizer, device, CONFIG["beta"])
        
        total_loss.append(loss.item())
    
    torch.cuda.empty_cache()

    return sum(total_loss) / len(total_loss)