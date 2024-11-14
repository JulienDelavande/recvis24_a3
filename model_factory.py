"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, data_transforms_224, data_transforms_224_da
from model import Net, ResNet18, ResNet50, ResNet101, EfficientNetB4, VitBase16


class ModelFactory:
    def __init__(self, model_name: str, test_mode: bool = False):
        self.test_mode = test_mode
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        if self.model_name == "resnet18":
            return ResNet18()
        if self.model_name == "resnet50":
            return ResNet50()
        if self.model_name == "resnet101":
            return ResNet101()
        if self.model_name == "efficientnet_b4":
            return EfficientNetB4()
        if self.model_name == "vit_base16":
            return VitBase16()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        if self.model_name == "resnet18":
            if self.test_mode:
                return data_transforms_224
            return data_transforms_224_da
        if self.model_name == "resnet50":
            if self.test_mode:
                return data_transforms_224
            return data_transforms_224
        if self.model_name == "resnet101":
            if self.test_mode:
                return data_transforms_224
            return data_transforms_224
        if self.model_name == "efficientnet_b4":
            if self.test_mode:
                return data_transforms_224
            return data_transforms_224
        if self.model_name == "vit_base16":
            if self.test_mode:
                return data_transforms_224
            return data_transforms_224
        
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
