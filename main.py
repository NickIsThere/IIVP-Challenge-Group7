import os
import torch
import pandas as pd
from torchvision import transforms
from src.dataset.data_pipeline import DataPipeline
from src.dataset.trainer import Trainer
from src.models.factory import Factory
from src.dataset.augmentation import Augmentation
from src.models.ensemble_pipeline import run_stacking_oof_cv, default_data_transforms

N_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 15
MODEL_NAME = "ResNet18"
csv_path = f'folds/train_folds_{N_FOLDS}.csv'
MODELS = ["ResNet18", "DenseNet121", "EfficientNet"]
PATH_TO_SAVE = "main_triple_stacking_longer_swa.pth"


def training_session(df, name_model, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': transforms.Compose([
            Augmentation(prob=0.4),
            transforms.RandomApply([transforms.ElasticTransform(alpha=34.0, sigma=4.0)], p=0.4),
            transforms.ToTensor(),
        ]),
        'validation': transforms.Compose([
            transforms.ToTensor(),
        ])
    }

    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)

    pipeline = DataPipeline(train_df, val_df, transforms=data_transforms)
    train_loader, val_loader = pipeline.get_loaders(batch_size=BATCH_SIZE, num_workers=4)

    model = Factory.get_model(name_model, num_classes=10)
    model = model.to(device)

    # Add label smoothing so the model is not overconfident
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Make floating learning rate using Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    trainer = Trainer(model, criterion, optimizer, device, scheduler)

    best_accuracy, history = trainer.fit(train_loader, val_loader, EPOCHS)

    return best_accuracy

def test_model(df):
    print("cuda" if torch.cuda.is_available() else "cpu")

    # Test for one model
    results = []
    for fold in range(N_FOLDS):
        print(f"Fold {fold + 1}")
        result = training_session(df, MODEL_NAME, fold)
        results.append(result)
        print(f"Fold {fold + 1}/{N_FOLDS} | best accuracy: {result:.4f}")
    accuracy = sum(results) / len(results)
    print(f"Average accuracy: {accuracy:.5f}")

def create_ensemble(df):
    save_dir = "ensembled_models"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")

    transforms = default_data_transforms(train_augmentation_prob=0.4)

    result = run_stacking_oof_cv(
        df=df,
        base_model_names=MODELS,
        folds=[0, 1, 2, 3, 4],
        num_classes=10,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,           
        num_workers=2,       
        data_transforms=transforms,
        device=device
    )

    print("=" * 20)
    print(f"Ensemble accuracy: {result.oof_accuracy:.5f}")
    print("=" * 20)

    save_path = os.path.join(save_dir, PATH_TO_SAVE)
    torch.save(result.stacker.state_dict(), save_path)
    print(f"Model is saved as {save_path}")


def main():
    if not os.path.exists(csv_path):
        print(f"Such file doesn't exist: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)

    # test_model(df)

    create_ensemble(df)

if __name__ == "__main__":
    main()