import os
import torch
from torch.utils.data import Dataset
from skimage import io

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