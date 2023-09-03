import glob
from itertools import chain
import os
import random
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import lightning.pytorch as pl
from my_lightning_module import MyLightningModule
from dataset1 import Dataset1

class Train:
    @classmethod
    def makedir(cls):
        os.makedirs('data', exist_ok=True)
    
    @classmethod
    def unzip(cls):
        with zipfile.ZipFile('tasks.zip') as tasks_zip:
            tasks_zip.extractall('data')

    @classmethod
    def obtain_filenames(cls):
        dir_path = 'data/tasks'
        
        if not os.path.isdir(dir_path):
            raise ValueError(f"Cannot scan dir at path: '{dir_path}' check that it exist and contains data")
            
        train_list = glob.glob(dir_path + '/**/train/*.png', recursive=True)
        test_list = glob.glob(dir_path + '/**/test/*.png', recursive=True)
        if len(train_list) == 0 or len(test_list) == 0:
            raise ValueError("Both lists must be non-empty")
            
        #print(f"Train Data: {len(train_list)}")
        #print(f"Test Data: {len(test_list)}")

        labels = [path.split('/')[-1].split('.')[0] for path in train_list]
        seed = 42
        train_list, valid_list = train_test_split(
            train_list, 
            test_size=0.2,
            stratify=labels,
            random_state=seed)
        print(f"Train Data: {len(train_list)}")
        print(f"Validation Data: {len(valid_list)}")
        print(f"Test Data: {len(test_list)}")
        return (train_list, valid_list, test_list)

    @classmethod
    def create_dataloaders(cls):
        print("obtaining paths to dataset files")
        train_list, valid_list, test_list = Train.obtain_filenames()

        train_data = Dataset1(train_list)
        valid_data = Dataset1(valid_list)
        test_data = Dataset1(test_list)

        train_loader = DataLoader(dataset = train_data)
        valid_loader = DataLoader(dataset = valid_data)
        test_loader = DataLoader(dataset = test_data)

        #print(f"Train Data: {len(train_data)}, Train Loader: {len(train_loader)}")
        #print(f"Valid Data: {len(valid_data)}, Valid Loader: {len(valid_data)}")
        return (train_loader, valid_loader, test_loader)

if __name__ == '__main__':
    seed = 42
    pl.seed_everything(seed, workers=True)
    # sets seeds for numpy, torch and python.random.
    
    Train.makedir()
    
    #Train.unzip()
    
    train_loader, valid_loader, test_loader = Train.create_dataloaders()
    
    model = MyLightningModule()

    #model.populate_with_legacy_checkpoint()

    logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(deterministic=True, logger=logger, limit_train_batches=64, max_epochs=100)
    trainer.fit(model=model, train_dataloaders=train_loader)

