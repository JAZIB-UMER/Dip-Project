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
        self.brain_mask = None
        self.binary_mask = None
        self.tumor_mask = None
        self.result_image = None
        self.tumor_detected = False
        self.tumor_area = 0
        self.tumor_contours = []
        self.brain_stats = {}

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from the given path."""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return self.original_image

    def load_image_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """Load an image from a numpy array (for Gradio compatibility)."""
        if image_array is None:
            raise ValueError("No image provided")

        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            self.original_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            self.original_image = image_array

        return self.original_image

    def convert_to_grayscale(self) -> np.ndarray:
        """Convert the loaded image to grayscale."""
        if self.original_image is None:
            raise ValueError("No image loaded. Call load_image() first.")

        if len(self.original_image.shape) == 3:
            self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            self.grayscale_image = self.original_image.copy()

        return self.grayscale_image

    def apply_noise_removal(self, method: str = "gaussian", kernel_size: int = 5) -> np.ndarray:
        """Apply noise removal filter to the grayscale image."""
        if self.grayscale_image is None:
            raise ValueError("No grayscale image. Call convert_to_grayscale() first.")

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
        """Enhance contrast of the preprocessed image."""
        if self.preprocessed_image is None:
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

    def extract_brain_region(self) -> np.ndarray:
        """Extract the brain region from the MRI image (skull stripping)."""
        if self.preprocessed_image is None:
            raise ValueError("No preprocessed image. Run preprocessing steps first.")

        # Use Otsu to get initial brain region
        _, brain_thresh = cv2.threshold(
            self.preprocessed_image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        brain_thresh = cv2.morphologyEx(brain_thresh, cv2.MORPH_CLOSE, kernel)
        brain_thresh = cv2.morphologyEx(brain_thresh, cv2.MORPH_OPEN, kernel)

        # Find the largest connected component (the brain)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(brain_thresh)

        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            self.brain_mask = np.uint8(labels == largest_label) * 255
        else:
            self.brain_mask = brain_thresh

        # Fill holes in the brain mask
        contours, _ = cv2.findContours(self.brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(self.brain_mask, contours, -1, 255, -1)

        # Calculate and store brain statistics for later use
        brain_pixels = self.preprocessed_image[self.brain_mask > 0]
        if len(brain_pixels) > 0:
            self.brain_stats = {
                "mean": np.mean(brain_pixels),
                "std": np.std(brain_pixels),
                "median": np.median(brain_pixels),
                "max": np.max(brain_pixels),
                "min": np.min(brain_pixels),
                "p75": np.percentile(brain_pixels, 75),
                "p90": np.percentile(brain_pixels, 90),
                "p95": np.percentile(brain_pixels, 95),
                "p99": np.percentile(brain_pixels, 99)
            }

        return self.brain_mask

    def detect_abnormal_regions(self, intensity_threshold: float = 0.85) -> np.ndarray:
        """
        Detect abnormally bright regions within the brain that could be tumors.
        Uses percentile-based thresholding for more robust detection.

        A region is considered abnormal only if it's in the top percentile
        of brightness AND significantly brighter than the median.
        """
        if self.brain_mask is None:
            self.extract_brain_region()

        brain_pixels = self.preprocessed_image[self.brain_mask > 0]

        if len(brain_pixels) == 0:
            self.binary_mask = np.zeros_like(self.preprocessed_image)
            return self.binary_mask

        # Use percentile-based threshold
        # intensity_threshold of 0.85 means we look at pixels above 85th percentile
        percentile_value = intensity_threshold * 100
        threshold_value = np.percentile(brain_pixels, percentile_value)

        # Additional constraint: threshold must be significantly above median
        # This prevents false positives in uniform images
        median_val = self.brain_stats.get("median", np.median(brain_pixels))
        std_val = self.brain_stats.get("std", np.std(brain_pixels))

        # Threshold should be at least median + 2*std to be considered abnormal
        min_threshold = median_val + 2 * std_val
        threshold_value = max(threshold_value, min_threshold)

        # Apply brain mask first
        brain_only = cv2.bitwise_and(self.preprocessed_image, self.preprocessed_image,
                                      mask=self.brain_mask)

        # Create binary mask for abnormally bright regions
        _, self.binary_mask = cv2.threshold(
            brain_only, threshold_value, 255, cv2.THRESH_BINARY
        )

        return self.binary_mask

    def apply_morphological_operations(
        self,
        operation: str = "closing",
        kernel_size: int = 5,
        iterations: int = 2
    ) -> np.ndarray:
        """Apply morphological operations to refine the binary mask."""
        if self.binary_mask is None:
            raise ValueError("No binary mask. Call detect_abnormal_regions() first.")

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # First apply strong opening to remove small noise
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        temp_mask = cv2.morphologyEx(self.binary_mask, cv2.MORPH_OPEN, small_kernel, iterations=2)

        if operation == "closing":
            self.tumor_mask = cv2.morphologyEx(
                temp_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations
            )
        elif operation == "opening":
            self.tumor_mask = cv2.morphologyEx(
                temp_mask, cv2.MORPH_OPEN, kernel, iterations=iterations
            )
        elif operation == "dilation":
            self.tumor_mask = cv2.dilate(
                temp_mask, kernel, iterations=iterations
            )
        elif operation == "erosion":
            self.tumor_mask = cv2.erode(
                temp_mask, kernel, iterations=iterations
            )
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")

        return self.tumor_mask

    def detect_tumor_region(
        self,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.25,
        circularity_threshold: float = 0.3,
        min_intensity_ratio: float = 1.4
    ) -> Tuple[bool, Dict]:
        """
        Detect and isolate the tumor region with strict validation criteria.

        Args:
            min_area_ratio: Minimum area ratio (relative to brain)
            max_area_ratio: Maximum area ratio
            circularity_threshold: Minimum circularity (0-1)
            min_intensity_ratio: Region must be this much brighter than brain median
        """
        if self.tumor_mask is None:
            raise ValueError("No tumor mask. Run morphological operations first.")

        brain_area = np.sum(self.brain_mask > 0) if self.brain_mask is not None else \
                     self.tumor_mask.shape[0] * self.tumor_mask.shape[1]

        min_area = brain_area * min_area_ratio
        max_area = brain_area * max_area_ratio

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.tumor_mask, connectivity=8
        )

        tumor_info = {
            "detected": False,
            "area": 0,
            "centroid": (0, 0),
            "bounding_box": (0, 0, 0, 0),
            "num_regions": 0,
            "circularity": 0,
            "intensity_ratio": 0,
            "confidence": "N/A"
        }

        valid_tumors = []
        brain_median = self.brain_stats.get("median", 128)
        brain_p75 = self.brain_stats.get("p75", 150)

        if num_labels > 1:
            for label_idx in range(1, num_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]

                # Check area constraints
                if area < min_area or area > max_area:
                    continue

                # Create mask for this component
                component_mask = np.uint8(labels == label_idx) * 255

                # Calculate circularity
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)

                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    continue

                # Check circularity - tumors should be reasonably compact
                if circularity < circularity_threshold:
                    continue

                # CRITICAL: Check intensity ratio
                # The region must be significantly brighter than normal brain tissue
                region_pixels = self.preprocessed_image[component_mask > 0]
                if len(region_pixels) == 0:
                    continue

                region_mean = np.mean(region_pixels)
                intensity_ratio = region_mean / brain_median if brain_median > 0 else 0

                # Must be significantly brighter (e.g., 40% brighter than median)
                if intensity_ratio < min_intensity_ratio:
                    continue

                # Additional check: region mean should be above 75th percentile of brain
                if region_mean < brain_p75:
                    continue

                valid_tumors.append({
                    "label": label_idx,
                    "area": area,
                    "centroid": tuple(centroids[label_idx].astype(int)),
                    "bounding_box": (
                        stats[label_idx, cv2.CC_STAT_LEFT],
                        stats[label_idx, cv2.CC_STAT_TOP],
                        stats[label_idx, cv2.CC_STAT_WIDTH],
                        stats[label_idx, cv2.CC_STAT_HEIGHT]
                    ),
                    "circularity": circularity,
                    "intensity_ratio": intensity_ratio,
                    "region_mean": region_mean
                })

        if valid_tumors:
            # Select the most likely tumor (highest intensity ratio * area score)
            best_tumor = max(valid_tumors,
                           key=lambda x: x["intensity_ratio"] * np.sqrt(x["area"]))

            # Create final tumor mask
            self.tumor_mask = np.uint8(labels == best_tumor["label"]) * 255

            # Determine confidence based on how abnormal the region is
            if best_tumor["intensity_ratio"] > 1.8 and best_tumor["circularity"] > 0.5:
                confidence = "High"
            elif best_tumor["intensity_ratio"] > 1.5 and best_tumor["circularity"] > 0.35:
                confidence = "Medium"
            else:
                confidence = "Low"

            tumor_info = {
                "detected": True,
                "area": best_tumor["area"],
                "centroid": best_tumor["centroid"],
                "bounding_box": best_tumor["bounding_box"],
                "num_regions": len(valid_tumors),
                "circularity": round(best_tumor["circularity"], 3),
                "intensity_ratio": round(best_tumor["intensity_ratio"], 2),
                "confidence": confidence
            }
        else:
            # No tumor found - create empty mask
            self.tumor_mask = np.zeros_like(self.tumor_mask)

        self.tumor_detected = tumor_info["detected"]
        self.tumor_area = tumor_info["area"]

        return tumor_info["detected"], tumor_info

    def extract_contours(self) -> list:
        """Extract contours from the tumor mask."""
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
        """Draw the tumor boundary on the original image."""
        if self.original_image is None:
            raise ValueError("No original image loaded.")

        self.extract_contours()
        self.result_image = self.original_image.copy()

        # Only draw contours if tumor was detected
        if self.tumor_detected and self.tumor_contours:
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
        intensity_threshold: float = 0.85,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.25,
        circularity_threshold: float = 0.3,
        min_intensity_ratio: float = 1.4,
        boundary_color: Tuple[int, int, int] = (0, 255, 0),
        boundary_thickness: int = 2
    ) -> Dict:
        """
        Complete processing pipeline for brain tumor detection.
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

        # Brain extraction
        self.extract_brain_region()

        # Detect abnormal regions
        self.detect_abnormal_regions(intensity_threshold=intensity_threshold)

        # Refine with morphological operations
        self.apply_morphological_operations(
            operation=morph_operation,
            kernel_size=morph_kernel,
            iterations=morph_iterations
        )

        # Tumor detection with strict validation
        detected, tumor_info = self.detect_tumor_region(
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            circularity_threshold=circularity_threshold,
            min_intensity_ratio=min_intensity_ratio
        )

        # Draw boundary
        self.draw_tumor_boundary(color=boundary_color, thickness=boundary_thickness)

        # Prepare results
        results = {
            "original": cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB),
            "grayscale": self.grayscale_image,
            "preprocessed": self.preprocessed_image,
            "brain_mask": self.brain_mask,
            "binary_mask": self.binary_mask,
            "tumor_mask": self.tumor_mask,
            "result": cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB),
            "tumor_detected": detected,
            "tumor_info": tumor_info,
            "brain_stats": self.brain_stats
        }

        return results


def create_visualization_grid(results: Dict) -> np.ndarray:
    """Create a grid visualization of all processing steps."""
    h, w = results["grayscale"].shape[:2]

    def resize_and_convert(img, target_h, target_w):
        if img is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        resized = cv2.resize(img, (target_w, target_h))
        if len(resized.shape) == 2:
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return resized

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
