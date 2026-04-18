from src.models.cnn import CNN

class Factory:
    @staticmethod # Make factory static class since we don't need unique instances
    def get_model(model_name, num_classes=10):
        if model_name == "CNN":
            return CNN(num_classes=num_classes)