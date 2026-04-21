from src.models.cnn import CNN
from src.models.dense_net import DenseNet
from src.models.efficient_net import EfficientNet
from src.models.resnet import ResNetCustom


class Factory:
    @staticmethod
    def get_model(model_name, num_classes=10, **model_kwargs):
        if model_name == "CNN":
            return CNN(num_classes=num_classes, **model_kwargs)
        if model_name in ["ResNet18", "ResNet34"]:
            return ResNetCustom(model_name=model_name, num_classes=num_classes, **model_kwargs)
        if model_name == "DenseNet121":
            return DenseNet(num_classes=num_classes, **model_kwargs)
        if model_name == "EfficientNet":
            return EfficientNet(num_classes=num_classes, **model_kwargs)
        raise ValueError(f"Model {model_name} not recognized")
