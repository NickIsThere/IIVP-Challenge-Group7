import os
from torch.utils.data import Dataset
from PIL import Image

class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = str(self.df.iloc[idx]['Id'])
        img_id = f"{img_id}.png"

        # Create path to the picture
        img_path = os.path.join(self.img_dir, img_id)

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)
            
        return image, self.df.iloc[idx]['Id']