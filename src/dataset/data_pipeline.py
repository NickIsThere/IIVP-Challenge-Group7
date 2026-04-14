from torch.utils.data import DataLoader
from src.dataset.numbers_dataset import NumbersDataset

class DataPipeline:
    def __init__(self, train_df, val_df, transforms):
        self.train_dataset = NumbersDataset(train_df, transforms["train"])
        self.test_dataset = NumbersDataset(val_df, transforms["validation"])

    def get_loaders(self, batch_size=64):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader