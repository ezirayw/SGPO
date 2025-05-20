from .base import Algo
import torch
import tqdm
import numpy as np
from .sampling_utils import get_pc_sampler
from models.d3pm import D3PM
from models.mdlm import MDLM

class DAPS(Algo):
    """
        Implementation of decoupled annealing posterior sampling for discrete data.
        https://arxiv.org/abs/2407.01521 (continuous version)
    """

    def __init__(self, net, forward_op=None, num_steps=200, ode_steps=64, mh_steps=1000, alpha=1, beta=1, max_dist=1, data_config=None, n_max_mutations=None, device='cuda'):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super().__init__(net=net, forward_op=forward_op, data_config=data_config, n_max_mutations=n_max_mutations, device=device)
        
        self.type = "D3PM" if type(net) == D3PM else "MDLM" #check net type

        self.num_steps = num_steps
        self.time_steps = torch.linspace(self.net.timestep-1, 1, self.num_steps, dtype=int)

        ## Support larger stepsizes in models/d3pm.py p_sample
        self.ode_steps = ode_steps
        self.mh_steps = mh_steps
        
        if forward_op is not None:
            self.alpha = alpha
        else:
            self.alpha = 0.
        self.max_dist = max_dist
        self.beta = beta

        #TODO a lot of this can be removed since it is inherited
        # self.data_config = data_config
        # self.n_max_mutations = n_max_mutations
        # self.seq_len = data_config.seq_len
        # self.full_seq = [self.net.tokenizer.tokenize(s) for s in data_config.full_seq]
        # self.full_seq_string = data_config.full_seq
        # self.n_max_mutations = n_max_mutations
        # if data_config.residues is None:
        #     self.residues = list(range(self.seq_len))
        # else:
        #     self.residues = [i-1 for i in data_config.residues] #convert to 0-indexing

        # self.alphabet = list(data_config.alphabet[:20]) #only take the 20 standard amino acids
        # self.special_tokens = list(data_config.alphabet[20:])
        # self.n_residues = len(self.residues)
        # self.mask = torch.zeros(self.seq_len)
        # self.mask[self.residues] = 1
        # self.mask = self.mask.to(device).int()
        # self.full_seq = torch.from_numpy(np.array(self.full_seq)).to(device).int().squeeze(-1)

    def update_model(self, classifier):
        self.forward_op = classifier.eval()

    def project(self, x):
        return x * self.mask + self.full_seq * (1 - self.mask)
    
    def log_ratio(self, sigma, hm_dist):
        alpha = (1 - torch.exp(-sigma)) * (1 - 1/self.net.S)
        log_alpha = torch.log(alpha+1e-5)
        log_1alpha = torch.log(1 - alpha)
        log_ratio = hm_dist * log_alpha + (self.net.length - hm_dist) * log_1alpha
        return log_ratio
    
    @torch.no_grad()
    def metropolis_hasting(self, x0hat, op, y, sigma, steps, alpha, inpaint=False):
        residues = torch.tensor(self.residues, device=x0hat.device)
        # if inpaint:
        #     residues = torch.tensor(self.residues, device=x0hat.device)
        # else:
        #     residues = torch.tensor(self.all_residues, device=x0hat.device)
        n_residues = residues.shape[0]

        x = x0hat.clone()
        dim = self.net.S
        N, L = x0hat.shape[0], x0hat.shape[1]
        current_log_likelihood = op.log_likelihood(x, y) * alpha
        current_hm_dist = (x != x0hat).sum(dim=-1)
        for _ in range(steps):

            # Get proposal
            
            for _ in range(self.max_dist):
                proposal = x.clone() # proposal, shape = [N, L]
                # for _ in range(self.max_dist):
                # idx = torch.randint(L, (N,), device=x.device)
                idx = residues[torch.randint(n_residues, (N,), device=x.device)] # This is to restrict the mutation to the residues
                v = torch.randint(dim, (N,), device=x.device)
                proposal.scatter_(1, idx[:, None], v.unsqueeze(1))

            # Compute log prob difference
            log_likelihood = op.log_likelihood(proposal,y) * alpha
            hm_dist = (proposal != x0hat).sum(dim=-1)
            log_ratio = log_likelihood - current_log_likelihood
            log_ratio += self.log_ratio(sigma/self.beta, hm_dist) - self.log_ratio(sigma/self.beta, current_hm_dist)

            # Metropolis-Hasting step
            rho = torch.clip(torch.exp(log_ratio), max=1.0)
            seed = torch.rand_like(rho)
            x = x * (seed > rho).unsqueeze(-1) + proposal * (seed < rho).unsqueeze(-1)
            current_log_likelihood = log_likelihood * (seed < rho) + current_log_likelihood * (seed > rho)
            current_hm_dist = hm_dist * (seed < rho) + current_hm_dist * (seed > rho)
            
        return x

    def uncond_sample(self, net, xt, t_start, inpaint=False):
        # Now this only supports stepsize 1 generation
        # TODO: support larger stepsizes in models/d3pm.py p_sample
        timesteps = torch.linspace(t_start, 0, self.ode_steps, dtype=int)
        for t, t_next in zip(timesteps[:-1], timesteps[1:]):
            xt = net.p_sample(xt, t, t_next)
            if inpaint:
                xt = self.project(xt) #skip for now, don't project onto only the residues being mutated
        return xt


    def inference(self, observation=None, num_samples=1, verbose=True, detokenize=False, inpaint=False):

        # pbar = tqdm.trange(self.net.timestep-1) if verbose else range(self.net.timestep-1)
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        xt = self.net.get_start(num_samples)

        for i in pbar:

            # 1. reverse diffusion
            x0hat = self.uncond_sample(self.net, xt, self.time_steps[i], inpaint)
            
            # 2. Metropolis Hasting
            if self.type == "D3PM":
                sigma = self.net.sigma[self.time_steps[i]]
            else:  
                sigma = self.net.sigma(self.time_steps[i])
                
            if self.alpha == 0:
                x0y = x0hat
            else:
                x0y = self.metropolis_hasting(x0hat, self.forward_op, observation, sigma, steps=self.mh_steps, alpha=self.alpha, inpaint=inpaint)
            
            # 3. forward diffusion
            if i == self.num_steps-1:
                xt = x0y
                break
            xt = self.net.q_sample(x0y, self.time_steps[i+1].repeat(num_samples))
        xt = self.project(xt)

        if detokenize:
            detokenized = [self.net.tokenizer.untokenize(s) for s in xt]
            #replace special AA characters with a random ones
            #limit the number of mutations
            detokenized = self.project_sequences(detokenized)

            return xt, detokenized
        else:
            return xt
        

class Scheduler:
    def __init__(self, num_steps, sigma_max, sigma_min):
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        steps = np.linspace(0, 1, num_steps)
        time_step_fn = lambda r: (sigma_max ** (1/7) + r * (sigma_min ** (1/7) - sigma_max ** (1 / 7))) ** 7
        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, 1e-5) 
        factor_steps = np.array(
            [2 * time_steps[i] * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        self.sigma_steps = time_steps
        factor_steps = [max(f, 0) for f in factor_steps]
        self.time_steps, self.factor_steps =  time_steps, factor_steps
        
        
class DiffusionSampler:
    """
        Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def sample(self, model, x_start):
        pbar = range(self.scheduler.num_steps)
        x = x_start
        for step in pbar:
            sigma, factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step]
            score = model.score(x, torch.as_tensor(sigma).to(x.device))
            x = x + factor * score * 0.5
        return x  
        
class DAPS_continuous(Algo):
    """
        Implementation of decoupled annealing posterior sampling for continuous data.
        https://arxiv.org/abs/2407.01521 (original version)
    """

    def __init__(self, net, forward_op=None, sigma_max=100, sigma_min=0.01,
                 num_steps=100,
                 ode_steps=10,
                 alpha=1,
                 lr=1e-4,
                 lgvd_steps=100,
                 data_config=None,
                 n_max_mutations=None, 
                 device='cuda'):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super().__init__(net=net, forward_op=forward_op, data_config=data_config, n_max_mutations=n_max_mutations, device=device)

        self.num_steps = num_steps
        self.ode_steps = ode_steps
        self.lr = lr
        self.lgvd_steps = lgvd_steps
        self.annealing_scheduler = Scheduler(num_steps, sigma_max, sigma_min)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        if forward_op is not None:
            self.alpha = alpha
        else:
            self.alpha = 0.
            
        
    def update_model(self, classifier):
        self.forward_op = classifier.eval()
        
    def project(self, x):
        return x * self.mask + self.full_seq * (1 - self.mask)
    
    def uncond_sample(self, net, xt, sigma):
        diffusion_scheduler = Scheduler(self.ode_steps, sigma, self.sigma_min)
        sampler = DiffusionSampler(diffusion_scheduler)
        return sampler.sample(net, xt)
        
    def langevin_dynamics(self, x0hat, op, y, sigma, steps, alpha):
        x0_tmp = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x0_tmp], self.lr)
        for _ in range(steps):
            optimizer.zero_grad()
            log_likelihood = - alpha * op.log_likelihood(x0_tmp, y).sum()
            log_likelihood += ((x0_tmp - x0hat.detach())**2).sum() / (2 * sigma**2)
            log_likelihood.backward()
            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x0_tmp)
                x0_tmp.data = x0_tmp.data + np.sqrt(2 * self.lr) * epsilon
        return x0_tmp.detach()
        
    def inference(self, observation=None, num_samples=1, verbose=True, detokenize=False):
        pbar = tqdm.trange(self.num_steps-1) if verbose else range(self.num_steps-1)
        xt = self.net.get_start(num_samples) * self.sigma_max

        for i in pbar:
            # 1. reverse diffusion
            sigma = self.annealing_scheduler.sigma_steps[i]
            x0hat = self.uncond_sample(self.net, xt, sigma)
            
            # 2. Metropolis Hasting
            if self.alpha == 0:
                x0y = x0hat
            else:
                x0y = self.langevin_dynamics(x0hat, self.forward_op, observation, sigma, steps=self.lgvd_steps, alpha=self.alpha)
            
            # 3. forward diffusion
            xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[i+1]
        
        _, x0_seq_prob = self.net.tweedie(xt, torch.as_tensor(self.annealing_scheduler.sigma_steps[self.num_steps-1]).to(xt.device))
        x0 = x0_seq_prob.argmax(-1)
        x0 = self.project(x0)

        if detokenize:
            detokenized = [self.net.tokenizer.untokenize(s) for s in x0]
            #replace special AA characters with a random ones
            #limit the number of mutations
            detokenized = self.project_sequences(detokenized)

            return x0, detokenized
        else:
            return x0