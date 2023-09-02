from torch.utils.data import Dataset
from torchvision import datasets, transforms
from convert_pilimage_to_onehot import convert_pilimage_to_onehot
from PIL import Image
import torch
import numpy as np

class Dataset1(Dataset):
    @classmethod
    def create_transform(cls):
        transform = transforms.Compose(
            [
                #transforms.Resize(224),
                transforms.Resize((224, 224)),
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
            ]
        )
        return transform


    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        #img_transformed = self.transform(img)
        img_transformed1 = self.transform(img)
        img_transformed2 = convert_pilimage_to_onehot(img_transformed1)
        img_transformed3 = img_transformed2.transpose(2, 0, 1)
        img_transformed = torch.from_numpy(img_transformed3.astype(np.float32))
        #print("shape a:", img_transformed.shape)

        raw_label = img_path.split("/")[-1].split(".")[0]
        label = 0
        if raw_label == "color0":
            label = 0
        if raw_label == "color1":
            label = 1
        if raw_label == "color2":
            label = 2
        if raw_label == "color3":
            label = 3
        if raw_label == "color4":
            label = 4
        if raw_label == "color5":
            label = 5
        if raw_label == "color6":
            label = 6
        if raw_label == "color7":
            label = 7
        if raw_label == "color8":
            label = 8
        if raw_label == "color9":
            label = 9
            
        return img_transformed, label
