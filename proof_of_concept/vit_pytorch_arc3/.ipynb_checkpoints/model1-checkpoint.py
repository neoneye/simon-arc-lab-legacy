import lightning.pytorch as pl
from linformer import Linformer
from vit_pytorch.efficient import ViT
import torch.nn as nn
import torch.optim as optim

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
        self.criterion = nn.CrossEntropyLoss()
        self.model = Model1.create_model()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        image, label_expected = batch

        #data = data.to(device)
        #label = label.to(device)

        label_predicted = self.model(image)
        loss = self.criterion(label_predicted, label_expected)
        
        #x = x.view(x.size(0), -1)
        #z = self.encoder(x)
        #x_hat = self.decoder(z)
        #loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
