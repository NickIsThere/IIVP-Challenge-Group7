import os
import torch
import pandas as pd
from src.dataset.data_pipeline import DataPipeline

N_FOLDS = 5
BATCH_SIZE = 64
csv_path = f'folds/train_folds_{N_FOLDS}.csv'


def training_session(model, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)

    data_transforms = {
        'train': None,
        'validation': None
    }

    train_df = df[df['fold'] != fold]
    val_df = df[df['fold'] == fold]

    pipeline = DataPipeline(train_df, val_df, transforms=data_transforms)
    train_loader, val_loader = pipeline.get_loaders(batch_size=BATCH_SIZE)

    # We will later add Factory module to get current model from model
    # We will fit the model on train_loader and val_loader
    # return accuracy

def main():
    if not os.path.exists(csv_path):
        print(f"Such file doesn't exist: {csv_path}")
        return
    
    results = []
    for fold in range(N_FOLDS):
        result = training_session(...)
        results.append(result)
    accuracy = sum(results) / len(results)
    print(f"Average accuracy: {accuracy:.5f}")

if __name__ == "__main__":
    main()