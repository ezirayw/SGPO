### TODO: alternatively implement a manual callback to save model weights in a format compatible with evodiff and guided-diffusion codebase ###
#https://pytorch-lightning.readthedocs.io/en/1.2.10/common/weights_loading.html

from pytorch_lightning.callbacks import Callback
from transformers import PreTrainedTokenizer, PreTrainedModel
import os

class HuggingFaceCheckpointer(Callback):
    def __init__(self, save_dir: str, save_every_n_epochs: int = 1):
        """
        Save Hugging Face-compatible checkpoints during training.

        Args:
            save_dir (str): Directory where the Hugging Face model and tokenizer will be saved.
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer to save (optional).
            save_every_n_epochs (int): Save checkpoint every n epochs.
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(save_dir, exist_ok=True)
        self.best_val_loss = float("inf")
    
    def on_validation_end(self, trainer, pl_module):
        """Save the model if the validation loss is the best so far."""

        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"New best validation loss: {self.best_val_loss}")

            # Ensure the LightningModule has a Hugging Face model
            if hasattr(pl_module, "network") and isinstance(pl_module.network, PreTrainedModel):
                dir_name = os.path.join(self.save_dir, "best")
                pl_module.network.save_pretrained(dir_name)
                print(f"Hugging Face model saved to {dir_name}")
                # # Current tokenizer is not compatible with this saving
                # if pl_module.tokenizer is not None:
                #     pl_module.tokenizer.save_pretrained(epoch_save_dir)
                #     print(f"Tokenizer saved to {epoch_save_dir}")
            else:
                raise ValueError("The LightningModule must have a 'model' attribute of type `PreTrainedModel`.")

    # def on_train_epoch_end(self, trainer, pl_module):
    #     """Save the Hugging Face model and tokenizer at the end of the epoch."""
    #     if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
    #         epoch_save_dir = os.path.join(self.save_dir, f"epoch_{trainer.current_epoch + 1}")
    #         os.makedirs(epoch_save_dir, exist_ok=True)

    #         # Ensure the LightningModule has a Hugging Face model
    #         if hasattr(pl_module, "network") and isinstance(pl_module.network, PreTrainedModel):
    #             pl_module.network.save_pretrained(epoch_save_dir)
    #             print(f"Hugging Face model saved to {epoch_save_dir}")

    #             # # Current tokenizer is not compatible with this saving
    #             # if pl_module.tokenizer is not None:
    #             #     pl_module.tokenizer.save_pretrained(epoch_save_dir)
    #             #     print(f"Tokenizer saved to {epoch_save_dir}")
    #         else:
    #             raise ValueError("The LightningModule must have a 'model' attribute of type `PreTrainedModel`.")

    # def on_fit_end(self, trainer, pl_module):
    #     """Save the final model and tokenizer at the end of training."""
    #     final_save_dir = os.path.join(self.save_dir, "final")
    #     os.makedirs(final_save_dir, exist_ok=True)

    #     if hasattr(pl_module, "model") and isinstance(pl_module.model, PreTrainedModel):
    #         pl_module.model.save_pretrained(final_save_dir)
    #         print(f"Final Hugging Face model saved to {final_save_dir}")

    #         if self.tokenizer is not None:
    #             self.tokenizer.save_pretrained(final_save_dir)
    #             print(f"Final tokenizer saved to {final_save_dir}")