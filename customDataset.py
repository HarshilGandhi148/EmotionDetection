import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torchvision.transforms as transforms
import torchvision

class CKEmotionDataset(Dataset):
    def __init__(self, root_dir, transform, train):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        self.anger = sorted(os.listdir(os.path.join(self.root_dir, "anger")))
        self.fear = sorted(os.listdir(os.path.join(self.root_dir, "fear")))
        self.happiness = sorted(os.listdir(os.path.join(self.root_dir, "happiness")))
        self.neutral = sorted(os.listdir(os.path.join(self.root_dir, "neutral")))
        self.sadness = sorted(os.listdir(os.path.join(self.root_dir, "sadness")))
        self.surprise = sorted(os.listdir(os.path.join(self.root_dir, "surprise")))

        self.angerl = len(self.anger)
        self.fearl = len(self.fear)
        self.happinessl = len(self.happiness)
        self.neutrall = len(self.neutral)
        self.sadnessl = len(self.sadness)
        self.surprisel = len(self.surprise)

        self.data = [self.anger, self.fear, self.happiness, self.neutral, self.sadness, self.surprise]
        self.length = self.angerl + self.fearl + self.happinessl + self.neutrall + self.sadnessl + self.surprisel


    def __len__(self):
        return self.length #455

    def sub_folder(self, index):
        if (index < self.angerl):
            return "anger", index
        elif (index >= (self.angerl) and index < (self.angerl + self.fearl)):
            return "fear", index - (self.angerl)
        elif (index >= (self.angerl + self.fearl) and index < (self.angerl + self.fearl + self.happinessl)):
            return "happiness", index - (self.angerl + self.fearl)
        elif (index >= (self.angerl+ self.fearl + self.happinessl) and index < (self.angerl + self.fearl + self.happinessl + self.neutrall)):
            return "neutral", index - (self.angerl + self.fearl + self.happinessl)
        elif (index >= (self.angerl + self.fearl + self.happinessl + self.neutrall) and index < (self.angerl + self.fearl + self.happinessl + self.neutrall + self.sadnessl)):
            return "sadness", index - (self.angerl + self.fearl + self.happinessl + self.neutrall)
        else:
            return "surprise", index - (self.angerl + self.fearl + self.happinessl + self.neutrall + self.sadnessl)

    def __getitem__(self, index):
        emotions_dict = {"anger": 0, "fear": 1, "happiness": 2, "neutral": 3, "sadness": 4, "surprise": 5}

        subfolder, mod_index = self.sub_folder(index)
        img_path = os.path.join(self.root_dir, subfolder, self.data[emotions_dict[subfolder]][mod_index])
        image = io.imread(img_path)

        target = torch.tensor(int(emotions_dict[subfolder]))

        if self.transform:
            image = self.transform(image)

        t = torchvision.transforms.Resize((48, 48))
        image = t(image)
        #image = image.expand(3, -1, -1)

        return image, target

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform, train):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.data = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(os.listdir(self.root_dir)) #28273

    def __getitem__(self, index):
        path = self.data[index]
        img_path = os.path.join(self.root_dir, path)
        image = io.imread(img_path)

        emotions_dict = {"angry": 0, "fear": 1, "happy": 2, "neutral": 3, "sad": 4, "suprise": 5}
        y_label = torch.tensor(int(emotions_dict.get(path.lower().split('-')[0])))

        if self.transform:
            image = self.transform(image)

        return image, y_label

# e = CKEmotionDataset("CK", transform=transforms.ToTensor(), train = False)
# print(e.__len__())