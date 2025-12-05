"""
Brain Tumor Detection Using Digital Image Processing Techniques
Authors: Fatima Azeem, Ruqqayya Bibi
Program: BS AI-S23
Instructor: Ma'am Sana Saleem

This module implements classical DIP techniques for brain tumor detection
from MRI images without using deep learning.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class BrainTumorDetector:
    """
    A class to detect brain tumors in MRI images using classical
    Digital Image Processing techniques.
    """

    def __init__(self):
        """Initialize the detector with default parameters."""
        self.original_image = None
        self.grayscale_image = None
        self.preprocessed_image = None
        self.binary_mask = None
        self.tumor_mask = None
        self.result_image = None
        self.tumor_detected = False
        self.tumor_area = 0
        self.tumor_contours = []

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from the given path.

        Args:
            image_path: Path to the MRI image

        Returns:
            Loaded image as numpy array
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return self.original_image

    def load_image_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Load an image from a numpy array (for Gradio compatibility).

        Args:
            image_array: Image as numpy array (RGB format from Gradio)

        Returns:
            Loaded image as numpy array
        """
        if image_array is None:
            raise ValueError("No image provided")

        # Gradio provides RGB, convert to BGR for OpenCV processing
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            self.original_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            self.original_image = image_array

        return self.original_image

    def convert_to_grayscale(self) -> np.ndarray:
        """
        Convert the loaded image to grayscale.

        Returns:
            Grayscale image
        """
        if self.original_image is None:
            raise ValueError("No image loaded. Call load_image() first.")

        if len(self.original_image.shape) == 3:
            self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            self.grayscale_image = self.original_image.copy()

        return self.grayscale_image

    def apply_noise_removal(self, method: str = "gaussian", kernel_size: int = 5) -> np.ndarray:
        """
        Apply noise removal filter to the grayscale image.

        Args:
            method: Filtering method - "gaussian", "median", or "bilateral"
            kernel_size: Size of the filter kernel (must be odd)

        Returns:
            Filtered image
        """
        if self.grayscale_image is None:
            raise ValueError("No grayscale image. Call convert_to_grayscale() first.")

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        if method == "gaussian":
            self.preprocessed_image = cv2.GaussianBlur(
                self.grayscale_image, (kernel_size, kernel_size), 0
            )
        elif method == "median":
            self.preprocessed_image = cv2.medianBlur(self.grayscale_image, kernel_size)
        elif method == "bilateral":
            self.preprocessed_image = cv2.bilateralFilter(
                self.grayscale_image, kernel_size, 75, 75
            )
        else:
            raise ValueError(f"Unknown filtering method: {method}")

        return self.preprocessed_image

    def enhance_contrast(self, method: str = "clahe") -> np.ndarray:
        """
        Enhance contrast of the preprocessed image.

        Args:
            method: Enhancement method - "histogram" or "clahe"

        Returns:
            Contrast-enhanced image
        """
        if self.preprocessed_image is None:
            # If no preprocessing done, use grayscale
            if self.grayscale_image is None:
                raise ValueError("No image to enhance. Call preprocessing steps first.")
            self.preprocessed_image = self.grayscale_image.copy()

        if method == "histogram":
            self.preprocessed_image = cv2.equalizeHist(self.preprocessed_image)
        elif method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.preprocessed_image = clahe.apply(self.preprocessed_image)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")

        return self.preprocessed_image

    def apply_otsu_thresholding(self) -> np.ndarray:
        """
        Apply Otsu's thresholding to segment the image.

        Returns:
            Binary mask after thresholding
        """
        if self.preprocessed_image is None:
            raise ValueError("No preprocessed image. Run preprocessing steps first.")

        # Apply Otsu's thresholding
        _, self.binary_mask = cv2.threshold(
            self.preprocessed_image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return self.binary_mask

    def apply_morphological_operations(
        self,
        operation: str = "closing",
        kernel_size: int = 5,
        iterations: int = 2
    ) -> np.ndarray:
        """
        Apply morphological operations to refine the binary mask.

        Args:
            operation: Type of operation - "closing", "opening", "dilation", "erosion"
            kernel_size: Size of the structuring element
            iterations: Number of times to apply the operation

        Returns:
            Refined binary mask
        """
        if self.binary_mask is None:
            raise ValueError("No binary mask. Call apply_otsu_thresholding() first.")

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        if operation == "closing":
            self.tumor_mask = cv2.morphologyEx(
                self.binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations
            )
        elif operation == "opening":
            self.tumor_mask = cv2.morphologyEx(
                self.binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations
            )
        elif operation == "dilation":
            self.tumor_mask = cv2.dilate(
                self.binary_mask, kernel, iterations=iterations
            )
        elif operation == "erosion":
            self.tumor_mask = cv2.erode(
                self.binary_mask, kernel, iterations=iterations
            )
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")

        return self.tumor_mask

    def detect_tumor_region(self, min_area_ratio: float = 0.01) -> Tuple[bool, Dict]:
        """
        Detect and isolate the tumor region using connected component analysis.

        Args:
            min_area_ratio: Minimum area ratio (relative to image) to consider as tumor

        Returns:
            Tuple of (tumor_detected, tumor_info_dict)
        """
        if self.tumor_mask is None:
            raise ValueError("No tumor mask. Run morphological operations first.")

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.tumor_mask, connectivity=8
        )

        # Calculate image area
        image_area = self.tumor_mask.shape[0] * self.tumor_mask.shape[1]
        min_area = image_area * min_area_ratio

        # Find the largest component (excluding background at index 0)
        tumor_info = {
            "detected": False,
            "area": 0,
            "centroid": (0, 0),
            "bounding_box": (0, 0, 0, 0),
            "num_regions": 0
        }

        if num_labels > 1:
            # Get areas of all components (excluding background)
            areas = stats[1:, cv2.CC_STAT_AREA]

            # Filter components by minimum area
            valid_indices = np.where(areas >= min_area)[0] + 1  # +1 because we excluded background

            if len(valid_indices) > 0:
                # Find the largest valid component
                largest_idx = valid_indices[np.argmax(areas[valid_indices - 1])]

                # Create mask for the tumor region
                self.tumor_mask = np.uint8(labels == largest_idx) * 255

                # Get tumor statistics
                tumor_info["detected"] = True
                tumor_info["area"] = stats[largest_idx, cv2.CC_STAT_AREA]
                tumor_info["centroid"] = tuple(centroids[largest_idx].astype(int))
                tumor_info["bounding_box"] = (
                    stats[largest_idx, cv2.CC_STAT_LEFT],
                    stats[largest_idx, cv2.CC_STAT_TOP],
                    stats[largest_idx, cv2.CC_STAT_WIDTH],
                    stats[largest_idx, cv2.CC_STAT_HEIGHT]
                )
                tumor_info["num_regions"] = len(valid_indices)

        self.tumor_detected = tumor_info["detected"]
        self.tumor_area = tumor_info["area"]

        return tumor_info["detected"], tumor_info

    def extract_contours(self) -> list:
        """
        Extract contours from the tumor mask.

        Returns:
            List of contours
        """
        if self.tumor_mask is None:
            raise ValueError("No tumor mask available.")

        contours, _ = cv2.findContours(
            self.tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.tumor_contours = contours
        return contours

    def draw_tumor_boundary(
        self,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw the tumor boundary on the original image.

        Args:
            color: BGR color for the boundary
            thickness: Line thickness

        Returns:
            Image with tumor boundary drawn
        """
        if self.original_image is None:
            raise ValueError("No original image loaded.")

        if not self.tumor_contours:
            self.extract_contours()

        # Create a copy of the original image
        self.result_image = self.original_image.copy()

        # Draw contours
        cv2.drawContours(self.result_image, self.tumor_contours, -1, color, thickness)

        return self.result_image

    def process_image(
        self,
        image_input,
        noise_filter: str = "gaussian",
        noise_kernel: int = 5,
        contrast_method: str = "clahe",
        morph_operation: str = "closing",
        morph_kernel: int = 5,
        morph_iterations: int = 2,
        min_area_ratio: float = 0.01,
        boundary_color: Tuple[int, int, int] = (0, 255, 0),
        boundary_thickness: int = 2
    ) -> Dict:
        """
        Complete processing pipeline for brain tumor detection.

        Args:
            image_input: Either a file path (str) or numpy array
            noise_filter: Noise removal method
            noise_kernel: Kernel size for noise removal
            contrast_method: Contrast enhancement method
            morph_operation: Morphological operation type
            morph_kernel: Kernel size for morphological operations
            morph_iterations: Number of morphological iterations
            min_area_ratio: Minimum area ratio for tumor detection
            boundary_color: Color for tumor boundary
            boundary_thickness: Thickness of boundary line

        Returns:
            Dictionary containing all processing results
        """
        # Load image
        if isinstance(image_input, str):
            self.load_image(image_input)
        else:
            self.load_image_from_array(image_input)

        # Preprocessing
        self.convert_to_grayscale()
        self.apply_noise_removal(method=noise_filter, kernel_size=noise_kernel)
        self.enhance_contrast(method=contrast_method)

        # Segmentation
        self.apply_otsu_thresholding()
        self.apply_morphological_operations(
            operation=morph_operation,
            kernel_size=morph_kernel,
            iterations=morph_iterations
        )

        # Tumor detection
        detected, tumor_info = self.detect_tumor_region(min_area_ratio=min_area_ratio)

        # Draw boundary
        self.draw_tumor_boundary(color=boundary_color, thickness=boundary_thickness)

        # Prepare results
        results = {
            "original": cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB),
            "grayscale": self.grayscale_image,
            "preprocessed": self.preprocessed_image,
            "binary_mask": self.binary_mask,
            "tumor_mask": self.tumor_mask,
            "result": cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB),
            "tumor_detected": detected,
            "tumor_info": tumor_info
        }

        return results


def create_visualization_grid(results: Dict) -> np.ndarray:
    """
    Create a grid visualization of all processing steps.

    Args:
        results: Dictionary from process_image()

    Returns:
        Combined visualization image
    """
    # Get image dimensions
    h, w = results["grayscale"].shape[:2]

    # Resize all images to same dimensions
    def resize_and_convert(img, target_h, target_w):
        resized = cv2.resize(img, (target_w, target_h))
        if len(resized.shape) == 2:
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return resized

    # Create grid
    row1 = np.hstack([
        resize_and_convert(results["original"], h, w),
        resize_and_convert(results["grayscale"], h, w),
        resize_and_convert(results["preprocessed"], h, w)
    ])

    row2 = np.hstack([
        resize_and_convert(results["binary_mask"], h, w),
        resize_and_convert(results["tumor_mask"], h, w),
        resize_and_convert(results["result"], h, w)
    ])

    grid = np.vstack([row1, row2])

    return grid
