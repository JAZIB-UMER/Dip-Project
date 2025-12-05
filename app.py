"""
Brain Tumor Detection - Gradio Web Interface
Authors: Fatima Azeem, Ruqqayya Bibi
Program: BS AI-S23
Instructor: Ma'am Sana Saleem
"""

import gradio as gr
import numpy as np
import cv2
from brain_tumor_detection import BrainTumorDetector


def process_mri_image(
    image,
    noise_filter,
    noise_kernel,
    contrast_method,
    morph_operation,
    morph_kernel,
    morph_iterations,
    intensity_threshold,
    min_area_ratio,
    circularity_threshold,
    min_intensity_ratio,
    boundary_thickness
):
    """Process an MRI image and return detection results."""
    if image is None:
        return None, None, None, None, None, None, "Please upload an MRI image."

    try:
        detector = BrainTumorDetector()

        results = detector.process_image(
            image_input=image,
            noise_filter=noise_filter,
            noise_kernel=int(noise_kernel),
            contrast_method=contrast_method,
            morph_operation=morph_operation,
            morph_kernel=int(morph_kernel),
            morph_iterations=int(morph_iterations),
            intensity_threshold=intensity_threshold,
            min_area_ratio=min_area_ratio,
            circularity_threshold=circularity_threshold,
            min_intensity_ratio=min_intensity_ratio,
            boundary_color=(0, 255, 0),
            boundary_thickness=int(boundary_thickness)
        )

        original = results["original"]
        preprocessed = results["preprocessed"]
        brain_mask = results["brain_mask"]
        binary_mask = results["binary_mask"]
        tumor_mask = results["tumor_mask"]
        result_with_boundary = results["result"]

        tumor_info = results["tumor_info"]
        brain_stats = results.get("brain_stats", {})

        if results["tumor_detected"]:
            info_text = f"""
## TUMOR DETECTED

**Confidence Level:** {tumor_info['confidence']}

### Tumor Statistics:
| Metric | Value |
|--------|-------|
| Area | {tumor_info['area']} pixels |
| Centroid | ({tumor_info['centroid'][0]}, {tumor_info['centroid'][1]}) |
| Circularity | {tumor_info['circularity']} |
| Intensity Ratio | {tumor_info['intensity_ratio']}x brighter than median |
| Candidate Regions | {tumor_info['num_regions']} |

### Why Detected:
- Region is **{tumor_info['intensity_ratio']}x brighter** than brain median (threshold: {min_intensity_ratio}x)
- Circularity of **{tumor_info['circularity']}** meets threshold ({circularity_threshold})
- Area within valid range
"""
        else:
            info_text = f"""
## NO TUMOR DETECTED

The scan appears **healthy** - no abnormal regions meeting tumor criteria were found.

### Brain Statistics:
| Metric | Value |
|--------|-------|
| Median Intensity | {brain_stats.get('median', 'N/A'):.1f} |
| Std Deviation | {brain_stats.get('std', 'N/A'):.1f} |
| 95th Percentile | {brain_stats.get('p95', 'N/A'):.1f} |

### Detection Criteria Used:
- Minimum intensity ratio: **{min_intensity_ratio}x** brighter than median
- Minimum circularity: **{circularity_threshold}**
- Minimum area: **{min_area_ratio*100:.1f}%** of brain
- Intensity threshold: **{intensity_threshold*100:.0f}th** percentile

*If you believe there should be a detection, try lowering the intensity ratio or thresholds.*
"""

        return (
            original,
            preprocessed,
            brain_mask,
            binary_mask,
            tumor_mask,
            result_with_boundary,
            info_text
        )

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return None, None, None, None, None, None, error_msg


def create_demo_interface():
    """Create and return the Gradio interface."""

    with gr.Blocks(title="Brain Tumor Detection System") as demo:
        gr.Markdown(
            """
            # Brain Tumor Detection Using Digital Image Processing
            ### Classical DIP-based system for detecting brain tumors in MRI scans

            **Authors:** Fatima Azeem, Ruqqayya Bibi | **Program:** BS AI-S23 | **Instructor:** Ma'am Sana Saleem

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload MRI Image")
                input_image = gr.Image(
                    label="Upload MRI Scan",
                    type="numpy",
                    height=250
                )

                gr.Markdown("### Processing Parameters")

                with gr.Accordion("Preprocessing", open=False):
                    noise_filter = gr.Dropdown(
                        choices=["gaussian", "median", "bilateral"],
                        value="gaussian",
                        label="Noise Filter"
                    )
                    noise_kernel = gr.Slider(
                        minimum=3, maximum=15, step=2, value=5,
                        label="Filter Kernel Size"
                    )
                    contrast_method = gr.Dropdown(
                        choices=["clahe", "histogram"],
                        value="clahe",
                        label="Contrast Enhancement"
                    )

                with gr.Accordion("Morphological Operations", open=False):
                    morph_operation = gr.Dropdown(
                        choices=["closing", "opening", "dilation", "erosion"],
                        value="closing",
                        label="Operation Type"
                    )
                    morph_kernel = gr.Slider(
                        minimum=3, maximum=15, step=2, value=5,
                        label="Kernel Size"
                    )
                    morph_iterations = gr.Slider(
                        minimum=1, maximum=10, step=1, value=2,
                        label="Iterations"
                    )

                with gr.Accordion("Detection Thresholds (Important)", open=True):
                    intensity_threshold = gr.Slider(
                        minimum=0.7, maximum=0.98, step=0.01, value=0.85,
                        label="Intensity Percentile Threshold",
                        info="Only consider pixels above this percentile"
                    )
                    min_intensity_ratio = gr.Slider(
                        minimum=1.1, maximum=2.0, step=0.05, value=1.4,
                        label="Min Intensity Ratio",
                        info="Tumor must be this much brighter than brain median"
                    )
                    min_area_ratio = gr.Slider(
                        minimum=0.005, maximum=0.1, step=0.005, value=0.01,
                        label="Min Area Ratio",
                        info="Minimum tumor size as % of brain"
                    )
                    circularity_threshold = gr.Slider(
                        minimum=0.1, maximum=0.7, step=0.05, value=0.3,
                        label="Min Circularity",
                        info="Shape compactness (1.0 = perfect circle)"
                    )

                with gr.Accordion("Display", open=False):
                    boundary_thickness = gr.Slider(
                        minimum=1, maximum=5, step=1, value=2,
                        label="Boundary Thickness"
                    )

                process_btn = gr.Button("Detect Tumor", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Results")

                with gr.Row():
                    output_original = gr.Image(label="Original", height=160)
                    output_preprocessed = gr.Image(label="Preprocessed", height=160)
                    output_brain_mask = gr.Image(label="Brain Mask", height=160)

                with gr.Row():
                    output_binary = gr.Image(label="Abnormal Regions", height=160)
                    output_tumor_mask = gr.Image(label="Tumor Mask", height=160)
                    output_result = gr.Image(label="Final Result", height=160)

                detection_info = gr.Markdown(
                    value="Upload an MRI image and click **Detect Tumor** to analyze."
                )

        gr.Markdown(
            """
            ---
            ### How It Works

            1. **Brain Extraction** - Isolates brain from background using Otsu's method
            2. **Statistical Analysis** - Calculates intensity distribution (mean, median, percentiles)
            3. **Abnormality Detection** - Finds regions above the intensity percentile threshold
            4. **Validation** - Filters candidates by:
               - **Intensity ratio**: Must be significantly brighter than brain median
               - **Size**: Must be within reasonable tumor size range
               - **Shape**: Must be reasonably circular/compact

            ### Key Insight
            A healthy brain has relatively **uniform intensity distribution**. Tumors appear as
            **abnormally bright regions** that are significantly brighter than surrounding tissue.
            The system requires a region to be at least **{min_intensity_ratio}x brighter** than
            the median brain intensity to be considered a tumor.
            """
        )

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
                intensity_threshold,
                min_area_ratio,
                circularity_threshold,
                min_intensity_ratio,
                boundary_thickness
            ],
            outputs=[
                output_original,
                output_preprocessed,
                output_brain_mask,
                output_binary,
                output_tumor_mask,
                output_result,
                detection_info
            ]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
