"""
Brain Tumor Detection - Command Line Interface
Authors: Fatima Azeem, Ruqqayya Bibi
Program: BS AI-S23

A simple command-line interface for processing MRI images.
"""

import argparse
import os
import cv2
import matplotlib.pyplot as plt
from brain_tumor_detection import BrainTumorDetector


def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Detection using Digital Image Processing"
    )
    parser.add_argument(
        "image_path",
        help="Path to the MRI image"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for the result image"
    )
    parser.add_argument(
        "--noise-filter",
        choices=["gaussian", "median", "bilateral"],
        default="gaussian",
        help="Noise removal filter (default: gaussian)"
    )
    parser.add_argument(
        "--contrast",
        choices=["clahe", "histogram"],
        default="clahe",
        help="Contrast enhancement method (default: clahe)"
    )
    parser.add_argument(
        "--morph-op",
        choices=["closing", "opening", "dilation", "erosion"],
        default="closing",
        help="Morphological operation (default: closing)"
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Display all processing steps"
    )

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return

    # Create detector and process image
    detector = BrainTumorDetector()

    print(f"Processing: {args.image_path}")
    print("-" * 50)

    results = detector.process_image(
        image_input=args.image_path,
        noise_filter=args.noise_filter,
        contrast_method=args.contrast,
        morph_operation=args.morph_op
    )

    # Print results
    if results["tumor_detected"]:
        info = results["tumor_info"]
        print("TUMOR DETECTED!")
        print(f"  Area: {info['area']} pixels")
        print(f"  Centroid: {info['centroid']}")
        print(f"  Bounding Box: {info['bounding_box']}")
    else:
        print("No tumor detected in the image.")

    # Save or display result
    if args.output:
        cv2.imwrite(args.output, cv2.cvtColor(results["result"], cv2.COLOR_RGB2BGR))
        print(f"\nResult saved to: {args.output}")

    if args.show_steps:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle("Brain Tumor Detection - Processing Steps", fontsize=14)

        axes[0, 0].imshow(results["original"])
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(results["grayscale"], cmap="gray")
        axes[0, 1].set_title("Grayscale")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(results["preprocessed"], cmap="gray")
        axes[0, 2].set_title("Preprocessed")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(results["binary_mask"], cmap="gray")
        axes[1, 0].set_title("Binary Mask (Otsu)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(results["tumor_mask"], cmap="gray")
        axes[1, 1].set_title("Tumor Mask")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(results["result"])
        axes[1, 2].set_title("Result with Boundary")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
