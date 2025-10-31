"""
Contrast enhancement filters.

Includes CLAHE, histogram equalization, and gamma correction.
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

from .base import Filter, FilterConfig, UniversalFilter, ColorFilter, MonoFilter


@dataclass
class CLAHEConfig(FilterConfig):
    """Configuration for CLAHE filter."""
    clip_limit: float = 2.0  # Contrast limiting
    grid_size: int = 8       # Tile grid size

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            'clip_limit': self.clip_limit,
            'grid_size': self.grid_size,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'CLAHEConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'CLAHEFilter'),
            clip_limit=data.get('clip_limit', 2.0),
            grid_size=data.get('grid_size', 8),
        )


class CLAHEFilter(UniversalFilter):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances local contrast while limiting noise amplification.

    Replaces: contrast_CLAHE, Grid_CLAHE, filter_enabled_CLAHE globals

    Example:
        >>> filter = CLAHEFilter(clip_limit=2.0, grid_size=8)
        >>> result = filter.process(image)
    """

    def __init__(self, clip_limit: float = 2.0, grid_size: int = 8, enabled: bool = True):
        """
        Initialize CLAHE filter.

        Args:
            clip_limit: Threshold for contrast limiting (1.0 to 10.0)
            grid_size: Size of grid for local equalization (4 to 16)
            enabled: Whether filter is enabled
        """
        config = CLAHEConfig(
            enabled=enabled,
            name="CLAHEFilter",
            clip_limit=clip_limit,
            grid_size=grid_size,
        )
        super().__init__(config)
        self.config: CLAHEConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply CLAHE.

        Args:
            image: Input image (2D grayscale or 3D color)

        Returns:
            Contrast-enhanced image
        """
        try:
            import cv2
        except ImportError:
            return image

        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=(self.config.grid_size, self.config.grid_size)
        )

        # Ensure proper dtype (CLAHE works on uint8)
        needs_conversion = False
        original_dtype = image.dtype

        if image.dtype != np.uint8:
            needs_conversion = True
            # Normalize to uint8
            image_min, image_max = image.min(), image.max()
            if image_max > image_min:
                image_normalized = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image_normalized = np.zeros_like(image, dtype=np.uint8)
        else:
            image_normalized = image

        # Apply CLAHE based on image type
        if len(image.shape) == 2:
            # Grayscale - direct application
            result = clahe.apply(image_normalized)
        else:
            # Color - convert to LAB, apply to L channel
            lab = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Convert back to original dtype if needed
        if needs_conversion:
            if original_dtype == np.uint16:
                result = (result.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
            elif original_dtype in [np.float32, np.float64]:
                result = result.astype(np.float32) / 255.0

        return result

    def set_clip_limit(self, clip_limit: float) -> None:
        """Set CLAHE clip limit."""
        self.config.clip_limit = max(1.0, min(10.0, clip_limit))

    def set_grid_size(self, grid_size: int) -> None:
        """Set grid size."""
        self.config.grid_size = max(2, min(32, grid_size))

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"CLAHEFilter(clip={self.config.clip_limit:.1f}, grid={self.config.grid_size}, {status})"


class HistogramEqualizeFilter(UniversalFilter):
    """
    Global histogram equalization.

    Enhances contrast by equalizing the histogram.

    Example:
        >>> filter = HistogramEqualizeFilter()
        >>> result = filter.process(image)
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize histogram equalization filter.

        Args:
            enabled: Whether filter is enabled
        """
        config = FilterConfig(enabled=enabled, name="HistogramEqualizeFilter")
        super().__init__(config)

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply histogram equalization.

        Args:
            image: Input image (2D or 3D)

        Returns:
            Equalized image
        """
        try:
            import cv2
        except ImportError:
            return image

        # Ensure uint8
        needs_conversion = False
        original_dtype = image.dtype

        if image.dtype != np.uint8:
            needs_conversion = True
            image_min, image_max = image.min(), image.max()
            if image_max > image_min:
                image_normalized = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image_normalized = np.zeros_like(image, dtype=np.uint8)
        else:
            image_normalized = image

        # Apply equalization
        if len(image.shape) == 2:
            result = cv2.equalizeHist(image_normalized)
        else:
            # For color, convert to YCrCb and equalize Y channel
            ycrcb = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        # Convert back
        if needs_conversion:
            if original_dtype == np.uint16:
                result = (result.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
            elif original_dtype in [np.float32, np.float64]:
                result = result.astype(np.float32) / 255.0

        return result


@dataclass
class GammaCorrectionConfig(FilterConfig):
    """Configuration for gamma correction filter."""
    gamma: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d['gamma'] = self.gamma
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'GammaCorrectionConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'GammaCorrectionFilter'),
            gamma=data.get('gamma', 1.0),
        )


class GammaCorrectionFilter(UniversalFilter):
    """
    Gamma correction for brightness adjustment.

    Gamma < 1.0: Brighten
    Gamma = 1.0: No change
    Gamma > 1.0: Darken

    Example:
        >>> filter = GammaCorrectionFilter(gamma=1.2)
        >>> result = filter.process(image)
    """

    def __init__(self, gamma: float = 1.0, enabled: bool = True):
        """
        Initialize gamma correction filter.

        Args:
            gamma: Gamma value (0.1 to 5.0)
            enabled: Whether filter is enabled
        """
        config = GammaCorrectionConfig(
            enabled=enabled,
            name="GammaCorrectionFilter",
            gamma=gamma,
        )
        super().__init__(config)
        self.config: GammaCorrectionConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply gamma correction.

        Args:
            image: Input image (2D or 3D)

        Returns:
            Gamma-corrected image
        """
        # Skip if gamma is 1.0
        if abs(self.config.gamma - 1.0) < 0.001:
            return image.copy()

        # Build lookup table for efficient computation
        if image.dtype == np.uint8:
            inv_gamma = 1.0 / self.config.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            return cv2.LUT(image, table) if 'cv2' in dir() else self._apply_gamma_direct(image)
        elif image.dtype == np.uint16:
            # For 16-bit, normalize, apply gamma, denormalize
            normalized = image.astype(np.float32) / 65535.0
            corrected = np.power(normalized, 1.0 / self.config.gamma)
            return (corrected * 65535.0).astype(np.uint16)
        else:
            # For float images
            normalized = (image - image.min()) / (image.max() - image.min())
            corrected = np.power(normalized, 1.0 / self.config.gamma)
            return corrected * (image.max() - image.min()) + image.min()

    def _apply_gamma_direct(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma directly without lookup table."""
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, 1.0 / self.config.gamma)
        return (corrected * 255.0).astype(np.uint8)

    def set_gamma(self, gamma: float) -> None:
        """Set gamma value."""
        self.config.gamma = max(0.1, min(5.0, gamma))

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"GammaCorrectionFilter(gamma={self.config.gamma:.2f}, {status})"
