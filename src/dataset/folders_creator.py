from sklearn.model_selection import StratifiedKFold

class FoldersCreator:
    def __init__(self, df, n_splits):
        self.df = df
        self.n_splits = n_splits

    def create_folds(self, output_path='data/folds.csv'):
        skfolds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.df['fold'] = -1
        folds = skfolds.split(self.df, self.df['label'])
        for fold_number, (train_idx, val_idx) in enumerate(folds):
            self.df.loc[val_idx, 'fold'] = fold_number
        # Save folds to csv file
        self.df.to_csv(output_path, index=False)
        return self.df