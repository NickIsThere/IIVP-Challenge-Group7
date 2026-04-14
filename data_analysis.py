from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd

dataset = ImageFolder(root='data/train/train')

print(f"Total images: {len(dataset)}")

data = []
for path, label in dataset.samples:
    with Image.open(path) as img:
        width, height = img.size
        data.append({
            'number': label,
            'width': width,
            'height': height
        })

df = pd.DataFrame(data)
print("Sizes:")
print(df[['width', 'height']].describe())

print("Images per number:")
print(df['number'].value_counts())

dataset_test = ImageFolder(root='data/test')
print(f"Test dataset: {len(dataset_test)} images")