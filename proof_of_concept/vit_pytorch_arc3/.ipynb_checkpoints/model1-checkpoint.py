import lightning.pytorch as pl
from linformer import Linformer
from vit_pytorch.efficient import ViT
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Model1(pl.LightningModule):
    @classmethod
    def create_model(cls):
        efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )
        model = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=10,
            transformer=efficient_transformer,
            channels=13,
        )
        return model

    def __init__(self):
        super().__init__()
        model = Model1.create_model()
        self.model = model
        self.my_optimizer = optim.Adam(model.parameters(), lr=1e-5)

    def training_step(self, batch, batch_idx):
        # `x` is the input image
        # `y` is the target label
        x, y = batch
        y_hat = self.model(x)
        # `y_hat` is the predicted label

        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.my_optimizer
