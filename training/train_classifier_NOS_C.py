import torch
from torch.utils.data import DataLoader
import hydra
from hydra.utils import instantiate
import os


def train_step(classifier, model, optimizer, criterion, x, t, y, project_fn=None):
    optimizer.zero_grad()
    # x is token IDs from the dataloader.
    # Get continuous embeddings using the model’s embedding table:

    # project to mutated only
    x = project_fn(x) if project_fn is not None else x

    x_embeds = model.get_embeds(x)

    # (Optionally add noise with the noise schedule:)
    #xt = model.noise_schedule.q_sample(x_embeds, t)
    # Feed the noisy embeddings and timestep to the regressor
    # add attention mask here
    attn_mask = None
    #y_pred = classifier(xt, t, attn_mask).squeeze()
    # y_pred = classifier(xt, attn_mask).squeeze()
    y_pred = model.model.network.get_labels(x_embeds, t, attn_mask).squeeze()
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_classifier(classifier, model, dataloader, train_config, save_dir, project_fn=None, ensemble_idx=0, time_conditioned=True):
    classifier = classifier.to(model.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=train_config.learning_rate)
    criterion = torch.nn.MSELoss()
    model.model.network.regression_head = classifier

    if train_config.wandb:
        import wandb
        wandb.init(project="discrete_diffusion", name="classifier_training")

    n_epochs = train_config.n_epochs
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x, y in dataloader:
            t = torch.randint(0, model.timestep, (x.shape[0],)).to(torch.long)
            # t = torch.randint(0, 1, (x.shape[0],)).to(torch.long)
            # t = torch.zeros(x.shape[0], dtype=torch.long).to(classifier.device)
            x = x.to(classifier.device)
            y = y.to(classifier.device)
            t = t.to(classifier.device)
            loss = train_step(classifier, model, optimizer, criterion, x, t, y, project_fn)
            epoch_loss += loss
            # print(f">Epoch {epoch+1} loss: {loss}\r", end="")
            if train_config.wandb:
                wandb.log({"loss": loss})
        # print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader)}")
        if train_config.wandb:
            wandb.log({"epoch_loss": epoch_loss / len(dataloader)})

        # t = torch.zeros(x.shape[0], dtype=torch.long).to(classifier.device)
        # loss = train_step(classifier, model, optimizer, criterion, x, t, y, project_fn)
        # print(loss)
    if train_config.wandb:
        wandb.finish()

    # save model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(classifier, os.path.join(save_dir, f"classifier_{ensemble_idx}.pt"))

    return classifier