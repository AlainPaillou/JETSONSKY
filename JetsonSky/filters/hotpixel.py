"""
Hot pixel removal filters.

Hot pixels are bright defective pixels common in astronomy imaging,
especially with long exposures or high gain.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .base import Filter, FilterConfig, UniversalFilter


@dataclass
class HotPixelConfig(FilterConfig):
    """Configuration for hot pixel filter."""
    threshold: float = 0.9  # Detection threshold (0.0 to 1.0)
    kernel_size: int = 3    # Median filter kernel size

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            'threshold': self.threshold,
            'kernel_size': self.kernel_size,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'HotPixelConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'HotPixelFilter'),
            threshold=data.get('threshold', 0.9),
            kernel_size=data.get('kernel_size', 3),
        )


class HotPixelFilter(UniversalFilter):
    """
    Remove hot pixels using median filtering.

    Hot pixels are significantly brighter than their neighbors.
    This filter detects and replaces them with the median of surrounding pixels.

    Replaces: filter_enabled_hotpix global

    Example:
        >>> filter = HotPixelFilter(threshold=0.9)
        >>> result = filter.process(image)
    """

    def __init__(self, threshold: float = 0.9, kernel_size: int = 3, enabled: bool = True):
        """
        Initialize hot pixel filter.

        Args:
            threshold: Detection sensitivity (0.0 to 1.0, higher = more aggressive)
            kernel_size: Median filter kernel size (3, 5, or 7)
            enabled: Whether filter is enabled
        """
        config = HotPixelConfig(
            enabled=enabled,
            name="HotPixelFilter",
            threshold=threshold,
            kernel_size=kernel_size,
        )
        super().__init__(config)
        self.config: HotPixelConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply hot pixel removal.

        Algorithm:
        1. Apply median filter to get smoothed image
        2. Calculate difference between original and smoothed
        3. Pixels with large positive difference are hot pixels
        4. Replace hot pixels with median-filtered values

        Args:
            image: Input image (2D or 3D)

        Returns:
            Image with hot pixels removed
        """
        try:
            import cv2
        except ImportError:
            return image

        # Apply median filter
        median_filtered = cv2.medianBlur(image, self.config.kernel_size)

        # Calculate difference
        if image.dtype == np.uint8:
            max_val = 255
        elif image.dtype == np.uint16:
            max_val = 65535
        else:
            max_val = image.max()

        # Convert to float for comparison
        image_float = image.astype(np.float32)
        median_float = median_filtered.astype(np.float32)

        # Calculate deviation
        diff = image_float - median_float

        # Threshold for hot pixel detection
        threshold_val = self.config.threshold * max_val * 0.1

        # Create mask of hot pixels
        hot_pixel_mask = diff > threshold_val

        # Replace hot pixels with median values
        result = image.copy()
        result[hot_pixel_mask] = median_filtered[hot_pixel_mask]

        return result

    def set_threshold(self, threshold: float) -> None:
        """Set detection threshold."""
        self.config.threshold = max(0.0, min(1.0, threshold))

    def set_kernel_size(self, kernel_size: int) -> None:
        """Set kernel size (must be odd)."""
        # Ensure odd number
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.config.kernel_size = max(3, min(7, kernel_size))

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"HotPixelFilter(threshold={self.config.threshold:.2f}, kernel={self.config.kernel_size}, {status})"


class DeadPixelFilter(UniversalFilter):
    """
    Remove dead pixels (stuck at low values).

    Similar to hot pixel filter but detects pixels that are too dark.

    Example:
        >>> filter = DeadPixelFilter(threshold=0.9)
        >>> result = filter.process(image)
    """

    def __init__(self, threshold: float = 0.9, kernel_size: int = 3, enabled: bool = True):
        """
        Initialize dead pixel filter.

        Args:
            threshold: Detection sensitivity (0.0 to 1.0)
            kernel_size: Median filter kernel size (3, 5, or 7)
            enabled: Whether filter is enabled
        """
        config = HotPixelConfig(  # Reuse config structure
            enabled=enabled,
            name="DeadPixelFilter",
            threshold=threshold,
            kernel_size=kernel_size,
        )
        super().__init__(config)
        self.config: HotPixelConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply dead pixel removal.

        Args:
            image: Input image (2D or 3D)

        Returns:
            Image with dead pixels removed
        """
        try:
            import cv2
        except ImportError:
            return image

        # Apply median filter
        median_filtered = cv2.medianBlur(image, self.config.kernel_size)

        # Calculate difference (negative for dead pixels)
        image_float = image.astype(np.float32)
        median_float = median_filtered.astype(np.float32)
        diff = median_float - image_float

        # Determine max value
        if image.dtype == np.uint8:
            max_val = 255
        elif image.dtype == np.uint16:
            max_val = 65535
        else:
            max_val = image.max()

        # Threshold for dead pixel detection
        threshold_val = self.config.threshold * max_val * 0.1

        # Create mask of dead pixels
        dead_pixel_mask = diff > threshold_val

        # Replace dead pixels with median values
        result = image.copy()
        result[dead_pixel_mask] = median_filtered[dead_pixel_mask]

        return result

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"DeadPixelFilter(threshold={self.config.threshold:.2f}, kernel={self.config.kernel_size}, {status})"


class BadPixelMapFilter(UniversalFilter):
    """
    Remove bad pixels using a pre-defined bad pixel map.

    Useful when you have a known map of defective pixels.

    Example:
        >>> bad_pixel_map = np.zeros((1080, 1920), dtype=bool)
        >>> bad_pixel_map[100, 200] = True  # Mark pixel as bad
        >>> filter = BadPixelMapFilter(bad_pixel_map)
        >>> result = filter.process(image)
    """

    def __init__(self, bad_pixel_map: np.ndarray = None, enabled: bool = True):
        """
        Initialize bad pixel map filter.

        Args:
            bad_pixel_map: Boolean array where True = bad pixel
            enabled: Whether filter is enabled
        """
        config = FilterConfig(enabled=enabled, name="BadPixelMapFilter")
        super().__init__(config)
        self.bad_pixel_map = bad_pixel_map

    def validate(self, image: np.ndarray) -> bool:
        """Validate image and bad pixel map compatibility."""
        super().validate(image)

        if self.bad_pixel_map is None:
            # No map provided, just return original
            return True

        # Check dimensions match
        if len(image.shape) == 2:
            # Grayscale
            if image.shape != self.bad_pixel_map.shape:
                raise ValueError(
                    f"Bad pixel map shape {self.bad_pixel_map.shape} doesn't match "
                    f"image shape {image.shape}"
                )
        else:
            # Color - map should match first two dimensions
            if image.shape[:2] != self.bad_pixel_map.shape:
                raise ValueError(
                    f"Bad pixel map shape {self.bad_pixel_map.shape} doesn't match "
                    f"image shape {image.shape[:2]}"
                )

        return True

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply bad pixel map correction.

        Replaces bad pixels with median of 8 surrounding neighbors.

        Args:
            image: Input image (2D or 3D)

        Returns:
            Image with bad pixels corrected
        """
        if self.bad_pixel_map is None:
            return image.copy()

        try:
            import cv2
        except ImportError:
            return image

        # Apply median filter
        median_filtered = cv2.medianBlur(image, 3)

        # Replace bad pixels
        result = image.copy()
        if len(image.shape) == 2:
            # Grayscale
            result[self.bad_pixel_map] = median_filtered[self.bad_pixel_map]
        else:
            # Color - apply to all channels
            for c in range(image.shape[2]):
                result[:, :, c][self.bad_pixel_map] = median_filtered[:, :, c][self.bad_pixel_map]

        return result

    def set_bad_pixel_map(self, bad_pixel_map: np.ndarray) -> None:
        """Set or update bad pixel map."""
        self.bad_pixel_map = bad_pixel_map

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        if self.bad_pixel_map is not None:
            bad_count = np.sum(self.bad_pixel_map)
            return f"BadPixelMapFilter({bad_count} bad pixels, {status})"
        else:
            return f"BadPixelMapFilter(no map, {status})"
