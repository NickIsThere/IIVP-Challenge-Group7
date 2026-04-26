from torch.utils.data import DataLoader
from src.dataset.numbers_dataset import NumbersDataset

class DataPipeline:
    def __init__(self, train_df, val_df, transforms):
        self.train_dataset = NumbersDataset(train_df, transforms["train"])
        self.test_dataset = NumbersDataset(val_df, transforms["validation"])

    # default num_workers=0 for windows... (comment on comment, Mac loves workers < -1, then we pull all!)
    def get_loaders(self, batch_size=64, num_workers=0, pin_memory=True):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  persistent_workers=(num_workers > 0))
        val_loader = DataLoader(self.test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                persistent_workers=(num_workers > 0))
        return train_loader, val_loader
