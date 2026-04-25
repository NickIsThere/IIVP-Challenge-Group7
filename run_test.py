import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset.test_dataset import TestDataset
from src.models.ensemble_pipeline import default_data_transforms
from src.models.factory import Factory
from src.models.boosting_ensemble import BaseModelSpec
from src.models.stacking_ensemble import StackingEnsemble

TEST_CSV = "data/test.csv"
TEST_DIR = "data/test/test"
MODEL_PATH = "ensembled_models/main_triple_stacking_longer_swa.pth"
OUTPUT_CSV = "data/submission_on_main_longer_swa.csv"
MODELS = ["ResNet18", "DenseNet121", "EfficientNet"]

def make_predictions(test_loader, stacker, device):
    predictions = []
    ids = []
    with torch.inference_mode():
        progress_bar = tqdm(test_loader, desc="Inference", leave=True)

        for images, image_ids in progress_bar:
            images = images.to(device)
            output = stacker.predict_with_details(images)
            preds = output.predictions.cpu().numpy()
            
            predictions.extend(preds)
            ids.extend(image_ids.numpy() if isinstance(image_ids[0], (int, float, torch.Tensor)) else image_ids)
            
    submission_df = pd.DataFrame({
        "Id": ids,
        "Category": predictions
    })
    return submission_df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(TEST_CSV):
        print(f"Error! No such file: {TEST_CSV}")

    if not os.path.exists(MODEL_PATH):
        print(f"Error! No such file: {MODEL_PATH}")

    test_df = pd.read_csv(TEST_CSV)

    transforms = default_data_transforms(train_augmentation_prob=0)["validation"]

    test_dataset = TestDataset(test_df, img_dir=TEST_DIR, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model_specs = {}
    for model in MODELS:
        model_specs[model] = BaseModelSpec(model=Factory.get_model(model, num_classes=10))

    stacker = StackingEnsemble(model_specs, num_classes=10).to(device)

    # Initialize meta learner
    dummy_images, _ = next(iter(test_loader))
    dummy_features = stacker.build_meta_features(dummy_images.to(device))
    stacker.initialize_meta_learner(feature_dim=dummy_features.shape[1])

    stacker = stacker.to(device)

    stacker.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    stacker.eval()

    submission_df = make_predictions(test_loader, stacker, device)
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"File {OUTPUT_CSV} is created")


if __name__ == "__main__":
    main()