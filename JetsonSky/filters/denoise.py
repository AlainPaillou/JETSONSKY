"""
Denoising filters for image quality enhancement.

Includes multiple denoising algorithms: KNN, Paillou, etc.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .base import Filter, FilterConfig, UniversalFilter, GPUFilter


@dataclass
class DenoiseKNNConfig(FilterConfig):
    """Configuration for KNN denoise filter."""
    strength: float = 0.4  # Denoise strength (0.0 to 1.0)
    knn_strength: float = 0.2  # KNN-specific strength
    template_window_size: int = 7
    search_window_size: int = 21

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            'strength': self.strength,
            'knn_strength': self.knn_strength,
            'template_window_size': self.template_window_size,
            'search_window_size': self.search_window_size,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'DenoiseKNNConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'DenoiseKNNFilter'),
            strength=data.get('strength', 0.4),
            knn_strength=data.get('knn_strength', 0.2),
            template_window_size=data.get('template_window_size', 7),
            search_window_size=data.get('search_window_size', 21),
        )


class DenoiseKNNFilter(UniversalFilter):
    """
    K-Nearest Neighbors denoising filter.

    Uses OpenCV's fastNlMeansDenoising for grayscale or
    fastNlMeansDenoisingColored for color images.

    Replaces: val_denoise_KNN, filter_enabled_denoise_KNN globals

    Example:
        >>> filter = DenoiseKNNFilter(strength=0.4, knn_strength=0.2)
        >>> result = filter.process(image)
    """

    def __init__(
        self,
        strength: float = 0.4,
        knn_strength: float = 0.2,
        template_window_size: int = 7,
        search_window_size: int = 21,
        enabled: bool = True
    ):
        """
        Initialize KNN denoise filter.

        Args:
            strength: Overall denoise strength (0.0 to 1.0)
            knn_strength: KNN-specific strength (0.0 to 1.0)
            template_window_size: Size of template patch (odd number)
            search_window_size: Size of search area (odd number)
            enabled: Whether filter is enabled
        """
        config = DenoiseKNNConfig(
            enabled=enabled,
            name="DenoiseKNNFilter",
            strength=strength,
            knn_strength=knn_strength,
            template_window_size=template_window_size,
            search_window_size=search_window_size,
        )
        super().__init__(config)
        self.config: DenoiseKNNConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply KNN denoising.

        Args:
            image: Input image (2D grayscale or 3D color)

        Returns:
            Denoised image
        """
        try:
            import cv2
        except ImportError:
            # If OpenCV not available, return original
            return image

        # Convert strength (0.0-1.0) to OpenCV h parameter (typically 3-30)
        h = self.config.strength * 30.0
        h_color = self.config.knn_strength * 30.0

        # Ensure image is right dtype for OpenCV
        if image.dtype != np.uint8 and image.dtype != np.uint16:
            # Normalize to uint8
            image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_normalized = image

        # Apply appropriate denoise based on image type
        if len(image.shape) == 2:
            # Grayscale
            result = cv2.fastNlMeansDenoising(
                image_normalized,
                None,
                h=h,
                templateWindowSize=self.config.template_window_size,
                searchWindowSize=self.config.search_window_size
            )
        else:
            # Color
            result = cv2.fastNlMeansDenoisingColored(
                image_normalized,
                None,
                h=h,
                hColor=h_color,
                templateWindowSize=self.config.template_window_size,
                searchWindowSize=self.config.search_window_size
            )

        # If input was different dtype, convert back
        if image.dtype != result.dtype:
            if image.dtype == np.uint16:
                result = (result.astype(np.float32) / 255.0 * 65535).astype(np.uint16)

        return result

    def set_strength(self, strength: float) -> None:
        """Set denoise strength (0.0 to 1.0)."""
        self.config.strength = max(0.0, min(1.0, strength))

    def set_knn_strength(self, knn_strength: float) -> None:
        """Set KNN strength (0.0 to 1.0)."""
        self.config.knn_strength = max(0.0, min(1.0, knn_strength))

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"DenoiseKNNFilter(strength={self.config.strength:.2f}, {status})"


@dataclass
class DenoisePaillouConfig(FilterConfig):
    """Configuration for Paillou denoise filter."""
    strength: float = 0.4

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d['strength'] = self.strength
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'DenoisePaillouConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'DenoisePaillouFilter'),
            strength=data.get('strength', 0.4),
        )


class DenoisePaillouFilter(UniversalFilter):
    """
    Paillou edge-preserving denoising filter.

    Uses bilateral filtering to denoise while preserving edges.

    Replaces: val_denoise, filter_enabled_denoise_Paillou globals

    Example:
        >>> filter = DenoisePaillouFilter(strength=0.4)
        >>> result = filter.process(image)
    """

    def __init__(self, strength: float = 0.4, enabled: bool = True):
        """
        Initialize Paillou denoise filter.

        Args:
            strength: Denoise strength (0.0 to 1.0)
            enabled: Whether filter is enabled
        """
        config = DenoisePaillouConfig(
            enabled=enabled,
            name="DenoisePaillouFilter",
            strength=strength,
        )
        super().__init__(config)
        self.config: DenoisePaillouConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Paillou denoising (bilateral filter).

        Args:
            image: Input image (2D or 3D)

        Returns:
            Denoised image
        """
        try:
            import cv2
        except ImportError:
            return image

        # Convert strength to bilateral filter parameters
        d = int(5 + self.config.strength * 10)  # Diameter (5-15)
        sigma_color = 25 + self.config.strength * 100  # Color sigma (25-125)
        sigma_space = 25 + self.config.strength * 100  # Space sigma (25-125)

        # Ensure proper dtype
        if image.dtype != np.uint8 and image.dtype != np.uint16:
            image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_normalized = image

        # Apply bilateral filter
        result = cv2.bilateralFilter(
            image_normalized,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )

        # Convert back if necessary
        if image.dtype != result.dtype:
            if image.dtype == np.uint16:
                result = (result.astype(np.float32) / 255.0 * 65535).astype(np.uint16)

        return result

    def set_strength(self, strength: float) -> None:
        """Set denoise strength (0.0 to 1.0)."""
        self.config.strength = max(0.0, min(1.0, strength))

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"DenoisePaillouFilter(strength={self.config.strength:.2f}, {status})"


@dataclass
class DenoiseGaussianConfig(FilterConfig):
    """Configuration for Gaussian denoise filter."""
    sigma: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d['sigma'] = self.sigma
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'DenoiseGaussianConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'DenoiseGaussianFilter'),
            sigma=data.get('sigma', 1.0),
        )


class DenoiseGaussianFilter(UniversalFilter):
    """
    Gaussian blur denoising filter.

    Simple Gaussian blur for noise reduction.

    Example:
        >>> filter = DenoiseGaussianFilter(sigma=1.5)
        >>> result = filter.process(image)
    """

    def __init__(self, sigma: float = 1.0, enabled: bool = True):
        """
        Initialize Gaussian denoise filter.

        Args:
            sigma: Standard deviation for Gaussian kernel
            enabled: Whether filter is enabled
        """
        config = DenoiseGaussianConfig(
            enabled=enabled,
            name="DenoiseGaussianFilter",
            sigma=sigma,
        )
        super().__init__(config)
        self.config: DenoiseGaussianConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Gaussian blur.

        Args:
            image: Input image (2D or 3D)

        Returns:
            Blurred image
        """
        try:
            import cv2
        except ImportError:
            return image

        # Calculate kernel size from sigma (should be odd)
        ksize = int(2 * np.ceil(3 * self.config.sigma) + 1)

        result = cv2.GaussianBlur(
            image,
            ksize=(ksize, ksize),
            sigmaX=self.config.sigma,
            sigmaY=self.config.sigma
        )

        return result

    def set_sigma(self, sigma: float) -> None:
        """Set Gaussian sigma."""
        self.config.sigma = max(0.1, sigma)

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"DenoiseGaussianFilter(sigma={self.config.sigma:.1f}, {status})"
