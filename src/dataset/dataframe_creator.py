import pandas as pd
from torchvision.datasets import ImageFolder

def generate_df(path):
    dataset = ImageFolder(path)
    df = pd.DataFrame(dataset.samples, columns=["path", "label"])
    return df