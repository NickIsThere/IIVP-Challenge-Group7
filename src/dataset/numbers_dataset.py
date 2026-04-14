import torch
from torch.utils.data import Dataset
from PIL import Image

class NumbersDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # Load image in greyscale format
        image = Image.open(row['path']).convert('L')
        label = row['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)