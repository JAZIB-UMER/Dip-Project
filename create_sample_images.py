"""
Create sample MRI-like images for testing the brain tumor detection system.
These are synthetic images for demonstration purposes only.
"""

import cv2
import numpy as np
import os


def create_sample_mri_with_tumor(output_path: str, size: int = 256):
    """
    Create a synthetic MRI-like image with a simulated tumor.

    Args:
        output_path: Path to save the image
        size: Image size (square)
    """
    # Create base brain-like elliptical shape
    image = np.zeros((size, size), dtype=np.uint8)

    # Draw brain outline (skull)
    center = (size // 2, size // 2)
    axes = (size // 2 - 20, size // 2 - 30)
    cv2.ellipse(image, center, axes, 0, 0, 360, 80, -1)

    # Add brain texture with random noise
    noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
    brain_mask = image > 0
    image[brain_mask] = image[brain_mask] + noise[brain_mask]

    # Add some internal brain structure
    cv2.ellipse(image, (size // 2, size // 2 - 20), (60, 40), 0, 0, 360, 100, -1)
    cv2.ellipse(image, (size // 2, size // 2 + 30), (50, 35), 0, 0, 360, 90, -1)

    # Add simulated tumor (bright region)
    tumor_center = (size // 2 + 40, size // 2 - 20)
    tumor_radius = 25
    cv2.circle(image, tumor_center, tumor_radius, 200, -1)

    # Add some gradient to tumor
    for i in range(tumor_radius, 0, -5):
        intensity = 180 + (tumor_radius - i) * 2
        cv2.circle(image, tumor_center, i, min(intensity, 255), -1)

    # Apply Gaussian blur for more realistic appearance
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Created sample MRI with tumor: {output_path}")


def create_sample_mri_healthy(output_path: str, size: int = 256):
    """
    Create a synthetic MRI-like image without tumor (healthy).

    Args:
        output_path: Path to save the image
        size: Image size (square)
    """
    # Create base brain-like elliptical shape
    image = np.zeros((size, size), dtype=np.uint8)

    # Draw brain outline
    center = (size // 2, size // 2)
    axes = (size // 2 - 20, size // 2 - 30)
    cv2.ellipse(image, center, axes, 0, 0, 360, 80, -1)

    # Add brain texture
    noise = np.random.randint(0, 25, (size, size), dtype=np.uint8)
    brain_mask = image > 0
    image[brain_mask] = image[brain_mask] + noise[brain_mask]

    # Add internal brain structures (ventricles, etc.)
    cv2.ellipse(image, (size // 2 - 20, size // 2), (30, 20), 0, 0, 360, 100, -1)
    cv2.ellipse(image, (size // 2 + 20, size // 2), (30, 20), 0, 0, 360, 100, -1)
    cv2.ellipse(image, (size // 2, size // 2 + 40), (40, 25), 0, 0, 360, 85, -1)

    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Created sample healthy MRI: {output_path}")


def create_sample_mri_large_tumor(output_path: str, size: int = 256):
    """
    Create a synthetic MRI-like image with a large tumor.

    Args:
        output_path: Path to save the image
        size: Image size (square)
    """
    # Create base brain-like elliptical shape
    image = np.zeros((size, size), dtype=np.uint8)

    # Draw brain outline
    center = (size // 2, size // 2)
    axes = (size // 2 - 20, size // 2 - 30)
    cv2.ellipse(image, center, axes, 0, 0, 360, 75, -1)

    # Add brain texture
    noise = np.random.randint(0, 20, (size, size), dtype=np.uint8)
    brain_mask = image > 0
    image[brain_mask] = image[brain_mask] + noise[brain_mask]

    # Add large tumor
    tumor_center = (size // 2 - 30, size // 2 + 10)
    cv2.ellipse(image, tumor_center, (45, 35), 20, 0, 360, 220, -1)

    # Add irregular edges to tumor
    for _ in range(5):
        offset = (np.random.randint(-15, 15), np.random.randint(-15, 15))
        cv2.circle(image,
                   (tumor_center[0] + offset[0], tumor_center[1] + offset[1]),
                   np.random.randint(10, 20), 200, -1)

    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Created sample MRI with large tumor: {output_path}")


def main():
    """Create all sample images."""
    # Create samples directory
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Create sample images
    create_sample_mri_with_tumor(os.path.join(samples_dir, "mri_tumor_1.png"))
    create_sample_mri_healthy(os.path.join(samples_dir, "mri_healthy.png"))
    create_sample_mri_large_tumor(os.path.join(samples_dir, "mri_tumor_large.png"))

    # Create additional variations
    create_sample_mri_with_tumor(os.path.join(samples_dir, "mri_tumor_2.png"), size=300)

    print(f"\nSample images created in: {samples_dir}")
    print("You can use these images to test the brain tumor detection system.")
    print("\nNote: These are synthetic images for testing only.")
    print("For real applications, use actual MRI scan data from medical datasets.")


if __name__ == "__main__":
    main()
