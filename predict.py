import torch
import numpy as np

from model import Net
from preprocess import preprocess_image


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "models/unet_model.pth"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = Net()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(image_path):

    model = load_model()

    # Preprocess MRI
    image = preprocess_image(image_path)

    # Run model
    with torch.no_grad():

        output = model(image)

        # Convert model output into probability
        probability = torch.sigmoid(output)

        # Keep the original medically meaningful
        # binary threshold of 0.5
        prediction = (
            probability > 0.5
        ).float()

    # Convert tensors to NumPy
    probability_map = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    mask = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )

    return mask, probability_map
