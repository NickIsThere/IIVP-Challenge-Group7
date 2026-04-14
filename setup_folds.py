import os
from src.dataset.dataframe_creator import generate_df
from src.dataset.folders_creator import FoldersCreator

N_FOLDS = 5
TRAIN_PATH = "data/train/train"

def main():
    csv_path = f'folds/train_folds_{N_FOLDS}.csv'

    if os.path.exists(csv_path):
        print("Such file already exists")
        return
    
    df = generate_df(TRAIN_PATH)

    folders_creator = FoldersCreator(df, N_FOLDS)
    folders_creator.create_folds(csv_path)
    print("Folders file was created")


if __name__ == "__main__":
    main()