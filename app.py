import os

import numpy as np
import streamlit as st

from PIL import Image

from predict import predict


# ============================================================
# PAGE CONFIGURATION
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

st.subheader(
    "MRI Tumor Segmentation System"
)

st.write(
    "Upload a pituitary MRI image to generate "
    "a U-Net-based tumor segmentation prediction."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project Information")

    st.write(
        "This application uses a U-Net deep learning "
        "model for pixel-level segmentation of "
        "suspected tumor regions in MRI images."
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

if st.button(
    "🔍 Analyze MRI",
    type="primary"
):

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
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )


    # --------------------------------------------------------
    # SAVE INPUT MRI
    # --------------------------------------------------------

    upload_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(
        upload_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # --------------------------------------------------------
    # LOAD ORIGINAL IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    original_resized = original_image.resize(
        (256, 256)
    )

    original_array = np.asarray(
        original_resized
    ).copy()


    # --------------------------------------------------------
    # RUN U-NET
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Analyzing MRI using U-Net..."
        ):

            mask, probability_map = predict(
                upload_path
            )

    except Exception as error:

        st.error(
            "The MRI could not be processed."
        )

        st.exception(error)

        st.stop()


    # --------------------------------------------------------
    # PREPARE OUTPUTS
    # --------------------------------------------------------

    mask = np.asarray(mask)

    probability_map = np.asarray(
        probability_map
    )


    # Make sure mask is 2D
    mask = np.squeeze(mask)

    # Make sure probability map is 2D
    probability_map = np.squeeze(
        probability_map
    )


    # --------------------------------------------------------
    # RESIZE OUTPUTS IF NECESSARY
    # --------------------------------------------------------

    if mask.shape != (256, 256):

        mask_image = Image.fromarray(
            mask.astype(np.uint8)
        )

        mask_image = mask_image.resize(
            (256, 256)
        )

        mask = np.asarray(
            mask_image
        )


    if probability_map.shape != (256, 256):

        probability_image = Image.fromarray(
            probability_map.astype(np.float32),
            mode="F"
        )

        probability_image = probability_image.resize(
            (256, 256)
        )

        probability_map = np.asarray(
            probability_image
        )


    # --------------------------------------------------------
    # BINARY MASK
    # --------------------------------------------------------

    mask_binary = (
        mask > 0
    )

    tumor_pixels = int(
        mask_binary.sum()
    )


    # --------------------------------------------------------
    # PROBABILITY INFORMATION
    # --------------------------------------------------------

    max_probability = float(
        probability_map.max()
    )

    mean_probability = float(
        probability_map.mean()
    )


    # --------------------------------------------------------
    # CREATE VISIBLE PROBABILITY MAP
    # --------------------------------------------------------

    # Normalize only for visualization.
    # This DOES NOT change the model prediction.

    if max_probability > 0:

        probability_visual = (
            probability_map /
            max_probability
        )

    else:

        probability_visual = (
            np.zeros_like(
                probability_map
            )
        )


    # Convert to 8-bit image
    probability_uint8 = (
        probability_visual * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # CREATE HEATMAP
    # --------------------------------------------------------

    heatmap = np.zeros(
        (
            256,
            256,
            3
        ),
        dtype=np.uint8
    )

    # Blue → low probability
    # Red → high probability

    heatmap[:, :, 0] = probability_uint8

    heatmap[:, :, 1] = (
        255 -
        probability_uint8
    )

    heatmap[:, :, 2] = (
        255 -
        probability_uint8
    )


    # --------------------------------------------------------
    # CREATE MRI + PROBABILITY OVERLAY
    # --------------------------------------------------------

    overlay_probability = (
        0.65 * original_array +
        0.35 * heatmap
    )

    overlay_probability = (
        overlay_probability
        .clip(0, 255)
        .astype(np.uint8)
    )


    # --------------------------------------------------------
    # CREATE MRI + BINARY MASK OVERLAY
    # --------------------------------------------------------

    overlay_mask = (
        original_array.copy()
    )

    if tumor_pixels > 0:

        overlay_mask[
            mask_binary
        ] = [
            255,
            0,
            0
        ]


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📊 Analysis Results"
    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if tumor_pixels == 0:

        st.warning(
            "The current model did not produce a "
            "positive segmentation region at the "
            "0.5 prediction threshold."
        )

        st.caption(
            "The probability map below shows the model's "
            "continuous output before thresholding. "
            "An empty binary mask does not confirm that "
            "the MRI is tumor-free."
        )

    else:

        st.success(
            "A segmentation region was produced "
            "by the current model."
        )

        st.caption(
            "The red region represents pixels classified "
            "as positive by the model."
        )


    # --------------------------------------------------------
    # IMAGE RESULTS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    # ORIGINAL MRI
    with col1:

        st.markdown(
            "### Original MRI"
        )

        st.image(
            original_resized,
            use_container_width=True
        )


    # PROBABILITY MAP
    with col2:

        st.markdown(
            "### Model Probability Map"
        )

        st.image(
            heatmap,
            use_container_width=True
        )

        st.caption(
            "Brighter/red regions indicate higher "
            "model probability relative to this image."
        )


    # OVERLAY
    with col3:

        st.markdown(
            "### MRI + Model Output"
        )

        st.image(
            overlay_probability,
            use_container_width=True
        )


    # --------------------------------------------------------
    # BINARY MASK
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Binary Segmentation Mask"
    )

    mask_col1, mask_col2 = st.columns(2)


    with mask_col1:

        st.image(
            mask_binary.astype(
                np.uint8
            ),
            clamp=True,
            use_container_width=True
        )


    with mask_col2:

        if tumor_pixels == 0:

            st.info(
                "No pixels crossed the 0.5 segmentation "
                "threshold for this image."
            )

        else:

            st.success(
                f"{tumor_pixels:,} pixels were "
                "classified as segmentation."
            )


    # --------------------------------------------------------
    # PREDICTION INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📋 Prediction Information"
    )

    info1, info2, info3, info4 = st.columns(4)


    with info1:

        st.metric(
            "Segmentation Pixels",
            f"{tumor_pixels:,}"
        )


    with info2:

        st.metric(
            "Maximum Probability",
            f"{max_probability:.4f}"
        )


    with info3:

        st.metric(
            "Mean Probability",
            f"{mean_probability:.4f}"
        )


    with info4:

        st.metric(
            "Processed Image",
            "256 × 256"
        )


    # --------------------------------------------------------
    # SAVE BINARY MASK
    # --------------------------------------------------------

    mask_image = (
        mask_binary.astype(
            np.uint8
        ) * 255
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
    # DOWNLOAD BINARY MASK
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


    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    st.success(
        "MRI processing completed successfully."
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
