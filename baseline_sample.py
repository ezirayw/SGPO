import os
from omegaconf import OmegaConf
import pickle
import hydra
from hydra.utils import instantiate
import numpy as np
import pandas as pd
import tqdm
import random
import torch

def set_seed(seed):
    random.seed(seed)                  # Python random module
    np.random.seed(seed)               # NumPy random seed

def sample(config, n_samples, algorithm, dataset, batch_size, round):
    #calculate number of batches
    max_iterations = n_samples // batch_size
    sample_steps = range(0, max_iterations)
    
    # for i in batch_steps:
    for step in tqdm.tqdm(sample_steps):
        _, detokenized = algorithm.inference(num_samples=batch_size, detokenize=True)
        _ = dataset.update_data(detokenized, n_samples, round=round, BO=True, unique_only=True)
    return dataset

@hydra.main(version_base="1.3", config_path="configs", config_name="baseline_config")
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_name = config.exp_name
    exp_dir = os.path.join(config.problem.exp_dir, config.data.name, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    # save config 
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))

    data_config = config.data

    save_path = os.path.join(exp_dir, 'summary.csv')
    summary_df = pd.DataFrame()

    set_seed(config.seed)
    num_fasta_samples = 1000
    sample_batch_size = config.num_samples if config.problem.sample_batch_size > config.num_samples else config.problem.sample_batch_size

    for task, n_max_mutations in zip(["unconstrained"], [None]):
        save_dir = os.path.join(exp_dir, task)
        os.makedirs(save_dir, exist_ok=True)
        fasta_save_path = os.path.join(save_dir, f'generated.fasta')
        dataset = instantiate(config.problem.data, data_config=data_config, tokenizer=None, n_random_init=num_fasta_samples, n_max_mutations=n_max_mutations)
            #save dataset.seqs to fasta
        with open(fasta_save_path, 'w') as f:
            for j, seq in enumerate(dataset.seqs):
                f.write(f">{j}\n")
                f.write(f"{seq}\n")
        print(f"Saved random sequences to {fasta_save_path}")

    for i in range(config.n_repeats):
        for round in range(config.n_rounds + 1):
            set_seed(config.seed + i + round)
            for task, n_max_mutations in zip(["unconstrained"], [None]):
                print(f'Sampling random values')
                dataset = instantiate(config.problem.data, data_config=data_config, tokenizer=None, n_random_init=config.num_samples, n_max_mutations=n_max_mutations)

                dataset.summary_df["task"] = task
                dataset.summary_df["repeat"] = i
                dataset.summary_df["round"] = round
                dataset.summary_df["method"] = "random"
                summary_df = pd.concat([summary_df, dataset.summary_df], axis=0)
                summary_df.to_csv(save_path, index=False)

    # for i in range(config.n_repeats):
    #     for round in range(config.n_rounds + 1):
    #         set_seed(config.seed + i + round)
    #         for task, n_max_mutations in zip(["unconstrained"], [None]):
    #             print(f'Sampling prior values')
    #             data_config = config.data
    #             seq_len = data_config.seq_len

    #             #currently supports d3pm_finetune as the baseline model
    #             net = instantiate(config.model.model, model_name=config.pretrained_ckpt, seq_len=seq_len, device=device) 
    #             algorithm = instantiate(config.algorithm.method, temperature=0., n_max_mutations=n_max_mutations, net=net, data_config=data_config)
    #             dataset = instantiate(config.problem.data, data_config=data_config, tokenizer=net.tokenizer)
    #             sample(config, config.num_samples, algorithm, dataset, sample_batch_size, round)

    #             dataset.summary_df["task"] = task
    #             dataset.summary_df["repeat"] = i
    #             dataset.summary_df["round"] = round
    #             dataset.summary_df["method"] = "prior_baseline"
    #             summary_df = pd.concat([summary_df, dataset.summary_df], axis=0)
    #             summary_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
