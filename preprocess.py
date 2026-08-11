import cv2
import numpy as np
import torch


def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (256, 256))

    image = image / 255.0

    image = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1)

    image = image.unsqueeze(0)

    return image