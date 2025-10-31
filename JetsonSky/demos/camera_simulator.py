"""
Camera Simulator for Demo Applications

This module simulates a ZWO ASI camera without requiring actual hardware.
Used for testing and demonstrating the Phase 1 refactoring.
"""

import numpy as np
from typing import Optional, Tuple
import time


class SimulatedCamera:
    """
    Simulated ZWO ASI camera for demo purposes.

    Mimics the behavior of a real camera without requiring hardware.
    Generates synthetic images with configurable noise and settings.
    """

    def __init__(self, model: str, width: int, height: int, bit_depth: int):
        """
        Initialize simulated camera.

        Args:
            model: Camera model name (e.g., "ZWO ASI178MC")
            width: Image width in pixels
            height: Image height in pixels
            bit_depth: Bit depth (12, 14, or 16)
        """
        self.model = model
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.max_value = 2 ** bit_depth - 1

        # Camera state
        self.exposition = 1000  # microseconds
        self.gain = 100
        self.is_capturing = False
        self.frame_count = 0

        # Simulate sensor noise
        self.read_noise = 5.0
        self.dark_current = 0.01  # e-/pixel/sec

        print(f"✓ Simulated camera initialized: {model}")
        print(f"  Resolution: {width}x{height}, {bit_depth}-bit")

    def set_exposition(self, exposition_us: int):
        """Set exposure time in microseconds."""
        self.exposition = exposition_us
        print(f"  Exposition set to {exposition_us} µs")

    def set_gain(self, gain: int):
        """Set gain value."""
        self.gain = gain
        print(f"  Gain set to {gain}")

    def start_capture(self):
        """Start image capture."""
        self.is_capturing = True
        self.frame_count = 0
        print("  Camera capturing started")

    def stop_capture(self):
        """Stop image capture."""
        self.is_capturing = False
        print("  Camera capturing stopped")

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.

        Returns:
            Numpy array with simulated image data, or None if not capturing
        """
        if not self.is_capturing:
            return None

        # Simulate exposure time delay
        time.sleep(self.exposition / 1000000.0)  # Convert µs to seconds

        # Generate synthetic image
        image = self._generate_synthetic_image()

        self.frame_count += 1
        return image

    def _generate_synthetic_image(self) -> np.ndarray:
        """
        Generate synthetic image with realistic characteristics.

        Returns:
            Numpy array with simulated image
        """
        # Base signal level (depends on exposition and gain)
        signal_level = (self.exposition / 1000) * (self.gain / 100) * 100

        # Create gradient background (simulates vignetting)
        y, x = np.ogrid[:self.height, :self.width]
        center_x, center_y = self.width / 2, self.height / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        vignetting = 1.0 - 0.3 * (distance / max_distance)

        # Base image with vignetting
        image = signal_level * vignetting

        # Add some "stars" (bright spots)
        num_stars = np.random.randint(50, 200)
        for _ in range(num_stars):
            star_x = np.random.randint(0, self.width)
            star_y = np.random.randint(0, self.height)
            star_brightness = np.random.uniform(500, self.max_value * 0.8)
            star_size = np.random.randint(1, 4)

            # Add gaussian star
            y_coords, x_coords = np.ogrid[
                max(0, star_y-star_size*2):min(self.height, star_y+star_size*2),
                max(0, star_x-star_size*2):min(self.width, star_x+star_size*2)
            ]
            star_mask = np.exp(-((x_coords - star_x)**2 + (y_coords - star_y)**2) / (2 * star_size**2))

            y_start = max(0, star_y-star_size*2)
            y_end = min(self.height, star_y+star_size*2)
            x_start = max(0, star_x-star_size*2)
            x_end = min(self.width, star_x+star_size*2)

            image[y_start:y_end, x_start:x_end] += star_brightness * star_mask

        # Add read noise
        noise = np.random.normal(0, self.read_noise, (self.height, self.width))
        image += noise

        # Add dark current (depends on exposition)
        dark = self.dark_current * (self.exposition / 1000000.0)
        image += dark

        # Clip to valid range
        image = np.clip(image, 0, self.max_value)

        # Convert to uint16
        return image.astype(np.uint16)

    def get_stats(self) -> dict:
        """
        Get camera statistics.

        Returns:
            Dictionary with camera stats
        """
        return {
            'model': self.model,
            'resolution': f"{self.width}x{self.height}",
            'bit_depth': self.bit_depth,
            'exposition': self.exposition,
            'gain': self.gain,
            'frames_captured': self.frame_count,
            'is_capturing': self.is_capturing,
        }


def create_simulated_camera(camera_model: str, resolution: Tuple[int, int], bit_depth: int) -> SimulatedCamera:
    """
    Factory function to create a simulated camera.

    Args:
        camera_model: Camera model name
        resolution: (width, height) tuple
        bit_depth: Bit depth (12, 14, or 16)

    Returns:
        SimulatedCamera instance
    """
    return SimulatedCamera(camera_model, resolution[0], resolution[1], bit_depth)
