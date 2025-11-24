import timm

from models.mlp import SimpleMLP


def build_model(name: str, num_classes: int, pretrained=False, image_size=32):
    name = name.lower()

    if name == 'mlp':
        return SimpleMLP(num_classes=num_classes, input_size=image_size)

    print(f"Creating model: {name} (Pretrained={pretrained})")
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model
