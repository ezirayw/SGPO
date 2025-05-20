import os
from omegaconf import OmegaConf
import pickle
import hydra
from hydra.utils import instantiate
import torch
from torch.utils.data import DataLoader
import wandb
import tqdm
from util.seed import set_seed

@hydra.main(version_base="1.3", config_path="configs", config_name="sample_config")
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(config.problem.exp_dir, config.data.name, config.pretrained_ckpt.split('/')[0], config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    # save config 
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))

    net = instantiate(config.model.model, model_name=config.pretrained_ckpt, seq_len=config.data.seq_len, device=device, _recursive_=False)
    forward_model = instantiate(config.problem.model, device=device)
    data_config = config.data
    # if config.measurement:
    #     dataset = instantiate(config.problem.dataset)
    #     dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)
    # algorithm = instantiate(config.algorithm.method, net=net, forward_op=forward_model, data_config=data_config)
    algorithm = instantiate(OmegaConf.load("configs/algorithm/uncond.yaml").method, net=net, forward_op=forward_model, data_config=data_config)

    set_seed(config.seed)

    # if config.measurement:
    #     # Inverse problem setting
    #     for i, data in enumerate(dataLoader):
    #         data = data.to(device)
    #         y = forward_model(data)
    #         samples = algorithm.inference(observation=y, num_samples=config.num_samples, detokenize=True)
    #         save_path = os.path.join(exp_dir, f'result_{i}.pt')
    #         torch.save(samples, save_path)
    # else:

    #batched unconditional sampling
    #batch_size = config.batch_size
    batch_size = config.num_samples if config.batch_size > config.num_samples else config.batch_size
    all_sequences = []
    save_path = os.path.join(exp_dir, 'generated.fasta')
    batch_steps = tqdm.tqdm(range(0, config.num_samples, batch_size))
    batch_steps.set_description(f"Sampling {config.num_samples} samples")
    for i in batch_steps:
        _, detokenzied_samples = algorithm.inference(num_samples=batch_size, seq_len=config.data.seq_len, detokenize=True)
        all_sequences.extend(detokenzied_samples)
        #save to fasta
        with open(save_path, 'w') as f:
            for i, seq in enumerate(all_sequences):
                f.write(f">{i}\n")
                f.write(f"{seq}\n")
    print(f"Saved generated sequences to {save_path}")
            
if __name__ == "__main__":
    main()