import torch
from torch.utils.data import DataLoader
# from problem.protein import GPModel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
#from botorch.models.deterministic import get_deterministic_model
import gpytorch
import hydra
from hydra.utils import instantiate
import os
from problem.protein import Embedder
from scipy.stats import spearmanr

from problem.protein import SampledDeterministicModel
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples

def train_classifier(model, dataloader, data_config, model_config, train_config, save_dir=None, project_fn=None, time_conditioned=True):
    device = model.device
    embedder = Embedder(data_config, model_config, time_conditioned=time_conditioned).to(device).eval()

    all_emb = []
    all_y = []

    timesteps = range(model.timestep) if time_conditioned else [0] #whether to train on clean or noisy data

    for timestep in timesteps:
        for x, y in dataloader:
            #it is typically too long make shorter if necessary
            if time_conditioned:
                t = torch.ones((x.shape[0],)).to(torch.long)*timestep
            else:
                t = torch.zeros(x.shape[0], dtype=torch.long)

            x = x.to(device)
            t = t.to(device)

            xt = model.q_sample(x, t) if time_conditioned else x
            xt = project_fn(xt) if project_fn is not None else xt
            emb = embedder(xt, t)

            all_emb.append(emb)
            all_y.append(y)
    
    #stack the list into a single tensor
    train_x = torch.cat(all_emb, dim=0)
    train_y = torch.cat(all_y, dim=0)

    #downsample randomly (otherwise GP model training is too slow) 20000 is too slow, 10000 is a bit slow, took like almost 5 minutes
    #k = int(len(train_x)*0.2)
    torch.manual_seed(train_config.seed)
    indices = torch.randperm(train_x.size(0))[:train_config.max_samples]
    print("Training on {} samples".format(len(indices)))
    train_x = train_x[indices].double().detach().to(device)
    train_y = train_y[indices].double().reshape(-1, 1).detach().to(device)

    #likelihood = hydra.utils.instantiate(model_config.likelihood).to(device) #GPModel
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    classifier_partial = hydra.utils.instantiate(model_config, _partial_=True)
    classifier = classifier_partial(train_X=train_x, train_Y=train_y).to(device)
    #classifier = SingleTaskGP(train_x, train_y, lik).to(device) #GPModel
    
    # 4. Train mode
    # classifier.train()

    # 5. Optimizer & Loss
    #optimizer = torch.optim.Adam(classifier.parameters(), lr=0.1) #can play around with this
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(classifier.likelihood, classifier)
    fit_gpytorch_mll(mll) 

    #print the spearman correlation between predicted and true values
    with torch.no_grad():
       pred = classifier(train_x).mean
       pred = pred.cpu().numpy()
       true = train_y.cpu().numpy()
       spearman = spearmanr(pred, true)
       print("GP Train Spearman correlation: ", spearman.correlation)
    
    #sample a model and print the correlation
    sampled_model = get_thompson_sample(classifier, data_config, model_config, time_conditioned=time_conditioned, device=device)
    with torch.no_grad():
       pred = sampled_model.sampled_model(train_x.unsqueeze(1))
       pred = pred.cpu().numpy()
       true = train_y.cpu().numpy()
       spearman = spearmanr(pred, true)
       print("Sampled model train Spearman correlation: ", spearman.correlation)

    if save_dir is None:
        return classifier
    #save model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'train_X': train_x,
        'train_Y': train_y,
    }, os.path.join(save_dir, f"classifier_GP.pt"))

    #alternatively could write out your own optimization loop
    
    return classifier

def get_thompson_sample(classifier_GP, data_config, model_config, time_conditioned=True, device="cuda"):
    gp_sample = get_gp_samples(
            model=classifier_GP,
            num_outputs=1,
            n_samples=1,
            num_rff_features=1000,
    )
    sampled_model = PosteriorMean(model=gp_sample)
    return SampledDeterministicModel(sampled_model, data_config, model_config, time_conditioned=time_conditioned).to(device)

#this does not actually work right now
# def get_thompson_sample(gp_model, dataset):
#     x = dataset.x
#     t =  t = torch.zeros(x.shape[0], dtype=torch.long)

#     posterior = gp_model.posterior(x, t)
#     sample_shape = torch.Size([1])
#     sampled_func = posterior.rsample(sample_shape).squeeze(0)

#     # Wrap the sampled function as a deterministic model
#     def sampled_function(X):
#         # X shape: batch_size x d
#         # You may need to normalize X if your GP uses input transforms
#         # This function assumes X is already normalized the same way as the GP
#         posterior = gp_model.posterior(X)
#         with torch.no_grad():
#             return posterior.rsample(sample_shape).squeeze(0)

#     # Wrap it into a deterministic BoTorch model
#     thompson_model = get_deterministic_model(sampled_function)
#     return thompson_model