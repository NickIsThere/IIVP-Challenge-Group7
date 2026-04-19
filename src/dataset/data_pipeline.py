from torch.utils.data import DataLoader
from src.dataset.numbers_dataset import NumbersDataset

class DataPipeline:
    def __init__(self, train_df, val_df, transforms):
        self.train_dataset = NumbersDataset(train_df, transforms["train"])
        self.test_dataset = NumbersDataset(val_df, transforms["validation"])

    def get_loaders(self, batch_size=64):
        # Added num_workers to parallelize CPU augmentations and pin_memory to speed up GPU transfer
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader