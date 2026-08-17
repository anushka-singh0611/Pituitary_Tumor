import torch
import numpy as np

from model import Net
from preprocess import preprocess_image


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "models/unet_model.pth"

# Visualization/testing threshold.
# Your model's probabilities are relatively low, so 0.10
# is used instead of 0.50 for this prototype.
THRESHOLD = 0.10


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

    # Model inference
    with torch.no_grad():

        output = model(image)

        # Convert logits to probabilities
        probability = torch.sigmoid(output)

    # --------------------------------------------------------
    # Convert probability tensor to numpy
    # --------------------------------------------------------

    probability_map = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Binary segmentation
    # --------------------------------------------------------

    mask = (
        probability_map >= THRESHOLD
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Return BOTH outputs
    # --------------------------------------------------------

    return mask, probability_map
