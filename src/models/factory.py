from src.models.cnn import CNN
from src.models.resnet import ResNetCustom
from src.models.dense_net import DenseNet

class Factory:
    @staticmethod # Make factory static class since we don't need unique instances
    def get_model(model_name, num_classes=10):
        if model_name == "CNN":
            return CNN(num_classes=num_classes)
        elif model_name in ["ResNet18", "ResNet34"]:
            return ResNetCustom(model_name=model_name, num_classes=num_classes)
        elif model_name == "DenseNet121":
            return DenseNet(num_classes=num_classes)
        else:
            raise ValueError(f"Model {model_name} not recognized")