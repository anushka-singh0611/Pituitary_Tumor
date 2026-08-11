import torch
import numpy as np

from model import Net
from preprocess import preprocess_image


MODEL_PATH = "models/unet_model.pth"


def load_model():

    model = Net()

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    return model


def predict(image_path):

    model = load_model()

    image = preprocess_image(image_path)

    with torch.no_grad():

        output = model(image)

        probability = torch.sigmoid(output)

        prediction = (probability > 0.5).float()

    mask = prediction.squeeze().numpy()

    return mask