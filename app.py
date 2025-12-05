"""
Brain Tumor Detection - Gradio Web Interface
Authors: Fatima Azeem, Ruqqayya Bibi
Program: BS AI-S23
Instructor: Ma'am Sana Saleem

This module provides a user-friendly web interface for the brain tumor
detection system using Gradio.
"""

import gradio as gr
import numpy as np
import cv2
from brain_tumor_detection import BrainTumorDetector, create_visualization_grid


def process_mri_image(
    image,
    noise_filter,
    noise_kernel,
    contrast_method,
    morph_operation,
    morph_kernel,
    morph_iterations,
    min_area_ratio,
    boundary_thickness
):
    """
    Process an MRI image and return detection results.

    Args:
        image: Input MRI image from Gradio
        noise_filter: Noise removal method
        noise_kernel: Kernel size for noise removal
        contrast_method: Contrast enhancement method
        morph_operation: Morphological operation type
        morph_kernel: Morphological kernel size
        morph_iterations: Number of morphological iterations
        min_area_ratio: Minimum area ratio for tumor detection
        boundary_thickness: Thickness of tumor boundary

    Returns:
        Tuple of processed images and detection info
    """
    if image is None:
        return None, None, None, None, None, "Please upload an MRI image."

    try:
        # Create detector instance
        detector = BrainTumorDetector()

        # Process the image
        results = detector.process_image(
            image_input=image,
            noise_filter=noise_filter,
            noise_kernel=int(noise_kernel),
            contrast_method=contrast_method,
            morph_operation=morph_operation,
            morph_kernel=int(morph_kernel),
            morph_iterations=int(morph_iterations),
            min_area_ratio=min_area_ratio,
            boundary_color=(0, 255, 0),  # Green boundary
            boundary_thickness=int(boundary_thickness)
        )

        # Prepare output images
        original = results["original"]
        preprocessed = results["preprocessed"]
        binary_mask = results["binary_mask"]
        tumor_mask = results["tumor_mask"]
        result_with_boundary = results["result"]

        # Prepare detection info text
        tumor_info = results["tumor_info"]
        if results["tumor_detected"]:
            info_text = f"""
## Tumor Detection Results

**Status:** Tumor Detected

**Tumor Statistics:**
- Area: {tumor_info['area']} pixels
- Centroid: ({tumor_info['centroid'][0]}, {tumor_info['centroid'][1]})
- Bounding Box: x={tumor_info['bounding_box'][0]}, y={tumor_info['bounding_box'][1]},
  width={tumor_info['bounding_box'][2]}, height={tumor_info['bounding_box'][3]}
- Number of Detected Regions: {tumor_info['num_regions']}

**Processing Parameters:**
- Noise Filter: {noise_filter}
- Contrast Enhancement: {contrast_method}
- Morphological Operation: {morph_operation}
"""
        else:
            info_text = f"""
## Tumor Detection Results

**Status:** No Tumor Detected

The system did not detect any significant tumor regions in this MRI scan.
This could mean:
1. The scan is healthy (no tumor present)
2. The tumor is too small to detect with current parameters
3. Parameters may need adjustment for this specific image

**Try adjusting:**
- Decrease the minimum area ratio
- Try different noise filters
- Adjust morphological parameters
"""

        return (
            original,
            preprocessed,
            binary_mask,
            tumor_mask,
            result_with_boundary,
            info_text
        )

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return None, None, None, None, None, error_msg


def create_demo_interface():
    """Create and return the Gradio interface."""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title-text {
        text-align: center;
        color: #2c3e50;
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
    }
    """

    with gr.Blocks(css=custom_css, title="Brain Tumor Detection System") as demo:
        # Header
        gr.Markdown(
            """
            # Brain Tumor Detection Using Digital Image Processing
            ### A DIP-based system for detecting brain tumors in MRI scans

            **Authors:** Fatima Azeem, Ruqqayya Bibi | **Program:** BS AI-S23 | **Instructor:** Ma'am Sana Saleem

            ---
            """
        )

        with gr.Row():
            # Left column - Input and Parameters
            with gr.Column(scale=1):
                gr.Markdown("### Upload MRI Image")
                input_image = gr.Image(
                    label="Upload MRI Scan",
                    type="numpy",
                    height=300
                )

                gr.Markdown("### Processing Parameters")

                with gr.Accordion("Preprocessing Settings", open=True):
                    noise_filter = gr.Dropdown(
                        choices=["gaussian", "median", "bilateral"],
                        value="gaussian",
                        label="Noise Removal Filter"
                    )
                    noise_kernel = gr.Slider(
                        minimum=3,
                        maximum=15,
                        step=2,
                        value=5,
                        label="Noise Filter Kernel Size"
                    )
                    contrast_method = gr.Dropdown(
                        choices=["clahe", "histogram"],
                        value="clahe",
                        label="Contrast Enhancement Method"
                    )

                with gr.Accordion("Segmentation Settings", open=True):
                    morph_operation = gr.Dropdown(
                        choices=["closing", "opening", "dilation", "erosion"],
                        value="closing",
                        label="Morphological Operation"
                    )
                    morph_kernel = gr.Slider(
                        minimum=3,
                        maximum=15,
                        step=2,
                        value=5,
                        label="Morphological Kernel Size"
                    )
                    morph_iterations = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=2,
                        label="Morphological Iterations"
                    )

                with gr.Accordion("Detection Settings", open=True):
                    min_area_ratio = gr.Slider(
                        minimum=0.001,
                        maximum=0.1,
                        step=0.001,
                        value=0.01,
                        label="Minimum Area Ratio"
                    )
                    boundary_thickness = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=2,
                        label="Boundary Thickness"
                    )

                process_btn = gr.Button("Detect Tumor", variant="primary", size="lg")

            # Right column - Results
            with gr.Column(scale=2):
                gr.Markdown("### Detection Results")

                with gr.Row():
                    with gr.Column():
                        output_original = gr.Image(label="Original Image", height=200)
                    with gr.Column():
                        output_preprocessed = gr.Image(label="Preprocessed Image", height=200)

                with gr.Row():
                    with gr.Column():
                        output_binary = gr.Image(label="Binary Mask (Otsu)", height=200)
                    with gr.Column():
                        output_tumor_mask = gr.Image(label="Tumor Mask", height=200)

                with gr.Row():
                    output_result = gr.Image(label="Final Result with Tumor Boundary", height=300)

                with gr.Row():
                    detection_info = gr.Markdown(
                        value="Upload an MRI image and click 'Detect Tumor' to see results."
                    )

        # Footer with methodology info
        gr.Markdown(
            """
            ---
            ### Methodology

            This system uses classical Digital Image Processing techniques:

            1. **Preprocessing:** Grayscale conversion, noise removal (Gaussian/Median/Bilateral filtering),
               contrast enhancement (CLAHE/Histogram Equalization)
            2. **Segmentation:** Otsu's automatic thresholding to separate foreground from background
            3. **Refinement:** Morphological operations (closing, opening, dilation, erosion) to clean the mask
            4. **Detection:** Connected component analysis to identify and isolate tumor regions
            5. **Visualization:** Contour extraction and boundary drawing on original image

            ---
            *Digital Image Processing Project - 2024*
            """
        )

        # Connect the process button
        process_btn.click(
            fn=process_mri_image,
            inputs=[
                input_image,
                noise_filter,
                noise_kernel,
                contrast_method,
                morph_operation,
                morph_kernel,
                morph_iterations,
                min_area_ratio,
                boundary_thickness
            ],
            outputs=[
                output_original,
                output_preprocessed,
                output_binary,
                output_tumor_mask,
                output_result,
                detection_info
            ]
        )

        # Add examples if sample images exist
        gr.Markdown(
            """
            ### Tips for Best Results

            - Use high-quality MRI scans in JPEG, PNG, or similar formats
            - Adjust the noise filter based on image quality
            - CLAHE generally works better than histogram equalization for MRI images
            - If no tumor is detected, try decreasing the minimum area ratio
            - For noisy images, increase the noise filter kernel size
            """
        )

    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_demo_interface()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
