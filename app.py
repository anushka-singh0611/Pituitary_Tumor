import os

import numpy as np
import streamlit as st
from PIL import Image

from predict import predict


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pituitary Tumor AI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Pituitary Tumor AI")
st.subheader("MRI Tumor Segmentation System")

st.write(
    "Upload a pituitary MRI image to generate a U-Net-based "
    "segmentation mask."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project Information")

    st.write(
        "This application uses a U-Net deep learning model "
        "to perform pixel-level segmentation of suspected "
        "tumor regions in MRI images."
    )

    st.divider()

    st.write("**Model:** U-Net")
    st.write("**Input:** MRI image")
    st.write("**Input size:** 256 × 256")
    st.write("**Output:** Segmentation mask")

    st.divider()

    st.caption(
        "Research / Educational Prototype"
    )


# ============================================================
# DOCTOR INFORMATION
# ============================================================

st.subheader("👨‍⚕️ Doctor Information")

email = st.text_input(
    "Doctor's Email ID",
    placeholder="doctor@example.com"
)


# ============================================================
# MRI UPLOAD
# ============================================================

st.subheader("📤 Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍 Analyze MRI", type="primary"):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not email.strip():

        st.warning(
            "Please enter the doctor's email ID."
        )

        st.stop()

    if uploaded_file is None:

        st.warning(
            "Please upload an MRI image."
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------------------------

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


    # --------------------------------------------------------
    # SAVE UPLOADED MRI
    # --------------------------------------------------------

    upload_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(upload_path, "wb") as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # --------------------------------------------------------
    # LOAD ORIGINAL IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # RUN U-NET
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Analyzing MRI using U-Net..."
        ):

            mask = predict(upload_path)

    except Exception as error:

        st.error(
            "The MRI could not be processed."
        )

        st.exception(error)

        st.stop()


    # --------------------------------------------------------
    # PREPARE MASK
    # --------------------------------------------------------

    mask = np.asarray(mask)

    mask_binary = (
        mask > 0
    )

    tumor_pixels = int(
        mask_binary.sum()
    )


    # --------------------------------------------------------
    # RESIZE ORIGINAL IMAGE
    # --------------------------------------------------------

    original_resized = original_image.resize(
        (256, 256)
    )

    original_array = np.asarray(
        original_resized
    ).copy()


    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    overlay = original_array.copy()

    if tumor_pixels > 0:

        # Highlight predicted region
        # in red for visualization.

        overlay[mask_binary] = [
            255,
            0,
            0
        ]


    # --------------------------------------------------------
    # RESULTS HEADER
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📊 Analysis Results"
    )


    # --------------------------------------------------------
    # RESULT STATUS
    # --------------------------------------------------------

    if tumor_pixels == 0:

        st.warning(
            "The current model did not produce a positive "
            "segmentation region for this image."
        )

        st.caption(
            "This does NOT confirm that the MRI is tumor-free. "
            "The result depends on the current model weights "
            "and should be reviewed by a qualified professional."
        )

    else:

        st.success(
            "A segmentation region was produced by the model."
        )

        st.caption(
            "The highlighted region represents the pixels "
            "classified by the current model."
        )


    # --------------------------------------------------------
    # THREE IMAGE COLUMNS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    # Original MRI
    with col1:

        st.markdown(
            "### Original MRI"
        )

        st.image(
            original_resized,
            use_container_width=True
        )


    # Predicted Mask
    with col2:

        st.markdown(
            "### Predicted Mask"
        )

        st.image(
            mask_binary.astype(np.uint8),
            clamp=True,
            use_container_width=True
        )


    # Overlay
    with col3:

        st.markdown(
            "### MRI + Segmentation"
        )

        st.image(
            overlay,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREDICTION INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📋 Prediction Information"
    )

    info1, info2, info3 = st.columns(3)


    with info1:

        st.metric(
            "Segmentation Pixels",
            f"{tumor_pixels:,}"
        )


    with info2:

        st.metric(
            "Processed Image",
            "256 × 256"
        )


    with info3:

        st.metric(
            "Model",
            "U-Net"
        )


    # --------------------------------------------------------
    # SAVE MASK
    # --------------------------------------------------------

    mask_image = (
        mask_binary.astype(np.uint8) * 255
    )

    mask_pil = Image.fromarray(
        mask_image
    )

    mask_path = os.path.join(
        "outputs",
        "predicted_tumor_mask.png"
    )

    mask_pil.save(
        mask_path
    )


    # --------------------------------------------------------
    # DOWNLOAD MASK
    # --------------------------------------------------------

    with open(
        mask_path,
        "rb"
    ) as file:

        st.download_button(
            label="⬇️ Download Segmentation Mask",
            data=file,
            file_name="predicted_tumor_mask.png",
            mime="image/png"
        )


    st.success(
        "MRI processing completed."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.info(
    "⚠️ Research / Educational Prototype — "
    "This application is not intended for clinical diagnosis "
    "or treatment decisions. Model outputs must be reviewed "
    "by a qualified medical professional."
)