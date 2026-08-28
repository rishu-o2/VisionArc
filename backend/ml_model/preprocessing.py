from PIL import Image


def load_fundus_image(path):
    return Image.open(path).convert("RGB")


def preprocess_image(image):
    # Preprocessing transforms will be added during model integration.
    return image
