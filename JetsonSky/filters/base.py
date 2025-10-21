"""
Base classes for image processing filters.

This module provides the foundation for the filter system, replacing the
monolithic filter functions with modular, testable filter classes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

# Optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


@dataclass
class FilterConfig:
    """Configuration for a filter instance."""
    enabled: bool = True
    name: str = ""

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'enabled': self.enabled,
            'name': self.name
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FilterConfig':
        """Create config from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', '')
        )


class Filter(ABC):
    """
    Abstract base class for all image processing filters.

    All filters must implement the apply() method and can optionally
    override validate() for parameter validation.
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize filter with configuration.

        Args:
            config: Filter configuration. If None, uses default config.
        """
        self.config = config or FilterConfig(name=self.__class__.__name__)

    @abstractmethod
    def apply(self, image, **kwargs):
        """
        Apply the filter to an image.

        Args:
            image: Input image array (can be 2D grayscale or 3D color)
            **kwargs: Additional parameters specific to the filter

        Returns:
            Processed image array

        Raises:
            ValueError: If image format is invalid
        """
        pass

    def validate(self, image) -> bool:
        """
        Validate input image format.

        Args:
            image: Input image to validate

        Returns:
            True if image is valid for this filter

        Raises:
            ValueError: If image is invalid
        """
        if image is None:
            raise ValueError(f"{self.config.name}: Image cannot be None")

        if HAS_NUMPY:
            if not isinstance(image, np.ndarray):
                raise ValueError(f"{self.config.name}: Image must be numpy array")

            if image.size == 0:
                raise ValueError(f"{self.config.name}: Image cannot be empty")
        else:
            # Basic validation without numpy
            if not hasattr(image, 'shape'):
                raise ValueError(f"{self.config.name}: Image must have 'shape' attribute")

        return True

    def process(self, image, **kwargs):
        """
        Process image with validation and enabled check.

        This is the main entry point for applying a filter. It handles
        validation and checks if the filter is enabled.

        Args:
            image: Input image array
            **kwargs: Additional parameters for apply()

        Returns:
            Processed image (or original if filter disabled)
        """
        # Skip if disabled
        if not self.config.enabled:
            return image

        # Validate input
        self.validate(image)

        # Apply filter
        return self.apply(image, **kwargs)

    def enable(self) -> None:
        """Enable this filter."""
        self.config.enabled = True

    def disable(self) -> None:
        """Disable this filter."""
        self.config.enabled = False

    def is_enabled(self) -> bool:
        """Check if filter is enabled."""
        return self.config.enabled

    def get_name(self) -> str:
        """Get filter name."""
        return self.config.name

    def __repr__(self) -> str:
        """String representation of filter."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"{self.config.name}({status})"


class ColorFilter(Filter):
    """
    Base class for filters that operate on color images.

    Validates that input is a 3-channel color image.
    """

    def validate(self, image) -> bool:
        """Validate image is color (3 channels)."""
        super().validate(image)

        if not hasattr(image, 'shape'):
            raise ValueError(f"{self.config.name}: Image must have 'shape' attribute")

        if len(image.shape) != 3:
            raise ValueError(
                f"{self.config.name}: Expected 3D color image, got {len(image.shape)}D"
            )

        if image.shape[2] != 3:
            raise ValueError(
                f"{self.config.name}: Expected 3 channels, got {image.shape[2]}"
            )

        return True


class MonoFilter(Filter):
    """
    Base class for filters that operate on monochrome images.

    Validates that input is a 2D grayscale image.
    """

    def validate(self, image) -> bool:
        """Validate image is monochrome (2D)."""
        super().validate(image)

        if not hasattr(image, 'shape'):
            raise ValueError(f"{self.config.name}: Image must have 'shape' attribute")

        if len(image.shape) != 2:
            raise ValueError(
                f"{self.config.name}: Expected 2D monochrome image, got {len(image.shape)}D"
            )

        return True


class UniversalFilter(Filter):
    """
    Base class for filters that work on both color and mono images.

    No additional validation beyond base Filter class.
    """
    pass


class GPUFilter(Filter):
    """
    Base class for GPU-accelerated filters using CuPy.

    Handles CPU/GPU array conversion and fallback to CPU if CuPy unavailable.
    """

    def __init__(self, config: Optional[FilterConfig] = None, use_gpu: bool = True):
        """
        Initialize GPU filter.

        Args:
            config: Filter configuration
            use_gpu: Whether to use GPU acceleration if available
        """
        super().__init__(config)
        self.use_gpu = use_gpu
        self.cupy_available = False

        # Try to import CuPy
        if use_gpu and HAS_NUMPY:
            try:
                import cupy as cp
                self.cp = cp
                self.cupy_available = True
            except ImportError:
                self.cupy_available = False

    def to_gpu(self, image):
        """Transfer image to GPU if available."""
        if self.cupy_available and self.use_gpu:
            return self.cp.asarray(image)
        return image

    def to_cpu(self, image):
        """Transfer image to CPU if it's on GPU."""
        if self.cupy_available and self.use_gpu:
            if isinstance(image, self.cp.ndarray):
                return self.cp.asnumpy(image)
        return image

    def process(self, image, **kwargs):
        """
        Process with automatic GPU/CPU conversion.

        Transfers to GPU, applies filter, transfers back to CPU.
        """
        if not self.config.enabled:
            return image

        self.validate(image)

        # Transfer to GPU
        gpu_image = self.to_gpu(image)

        # Apply filter (subclass implements this)
        result = self.apply(gpu_image, **kwargs)

        # Transfer back to CPU
        return self.to_cpu(result)
