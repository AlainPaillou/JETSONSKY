"""
Sharpening filters for image enhancement.

Includes unsharp mask and various sharpening techniques.
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
class SharpenConfig(FilterConfig):
    """Configuration for sharpen filter."""
    amount: float = 1.5  # Sharpening strength
    sigma: float = 1.0   # Gaussian sigma for unsharp mask

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            'amount': self.amount,
            'sigma': self.sigma,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'SharpenConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'SharpenFilter'),
            amount=data.get('amount', 1.5),
            sigma=data.get('sigma', 1.0),
        )


class SharpenFilter(UniversalFilter):
    """
    Unsharp mask sharpening filter.

    Enhances edges by subtracting a blurred version of the image.

    Replaces: val_sharpen, filter_enabled_sharpen1 globals

    Example:
        >>> filter = SharpenFilter(amount=1.5, sigma=1.0)
        >>> result = filter.process(image)
    """

    def __init__(self, amount: float = 1.5, sigma: float = 1.0, enabled: bool = True):
        """
        Initialize sharpen filter.

        Args:
            amount: Sharpening strength (0.0 to 5.0, typically 1.0-2.0)
            sigma: Gaussian sigma for blur (typically 0.5-2.0)
            enabled: Whether filter is enabled
        """
        config = SharpenConfig(
            enabled=enabled,
            name="SharpenFilter",
            amount=amount,
            sigma=sigma,
        )
        super().__init__(config)
        self.config: SharpenConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply unsharp mask sharpening.

        Algorithm:
        1. Create blurred version of image
        2. Subtract blur from original to get edges
        3. Add scaled edges back to original

        Args:
            image: Input image (2D or 3D)

        Returns:
            Sharpened image
        """
        try:
            import cv2
        except ImportError:
            return image

        # Calculate kernel size from sigma
        ksize = int(2 * np.ceil(3 * self.config.sigma) + 1)

        # Create blurred version
        blurred = cv2.GaussianBlur(
            image,
            ksize=(ksize, ksize),
            sigmaX=self.config.sigma,
            sigmaY=self.config.sigma
        )

        # Convert to float for arithmetic
        image_float = image.astype(np.float32)
        blurred_float = blurred.astype(np.float32)

        # Unsharp mask: original + amount * (original - blurred)
        sharpened = image_float + self.config.amount * (image_float - blurred_float)

        # Clip to valid range
        if image.dtype == np.uint8:
            sharpened = np.clip(sharpened, 0, 255)
        elif image.dtype == np.uint16:
            sharpened = np.clip(sharpened, 0, 65535)
        else:
            sharpened = np.clip(sharpened, image.min(), image.max())

        # Convert back to original dtype
        return sharpened.astype(image.dtype)

    def set_amount(self, amount: float) -> None:
        """Set sharpening amount."""
        self.config.amount = max(0.0, min(5.0, amount))

    def set_sigma(self, sigma: float) -> None:
        """Set Gaussian sigma."""
        self.config.sigma = max(0.1, sigma)

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"SharpenFilter(amount={self.config.amount:.1f}, sigma={self.config.sigma:.1f}, {status})"


@dataclass
class LaplacianSharpenConfig(FilterConfig):
    """Configuration for Laplacian sharpen filter."""
    strength: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d['strength'] = self.strength
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'LaplacianSharpenConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'LaplacianSharpenFilter'),
            strength=data.get('strength', 1.0),
        )


class LaplacianSharpenFilter(UniversalFilter):
    """
    Laplacian edge detection sharpening.

    Uses Laplacian operator to detect edges and enhance them.

    Example:
        >>> filter = LaplacianSharpenFilter(strength=1.0)
        >>> result = filter.process(image)
    """

    def __init__(self, strength: float = 1.0, enabled: bool = True):
        """
        Initialize Laplacian sharpen filter.

        Args:
            strength: Sharpening strength (0.0 to 3.0)
            enabled: Whether filter is enabled
        """
        config = LaplacianSharpenConfig(
            enabled=enabled,
            name="LaplacianSharpenFilter",
            strength=strength,
        )
        super().__init__(config)
        self.config: LaplacianSharpenConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Laplacian sharpening.

        Args:
            image: Input image (2D or 3D)

        Returns:
            Sharpened image
        """
        try:
            import cv2
        except ImportError:
            return image

        # For color images, convert to grayscale for Laplacian
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Convert to float for arithmetic
        image_float = image.astype(np.float32)

        # Add scaled Laplacian
        if len(image.shape) == 3:
            # Expand Laplacian to 3 channels
            laplacian_3ch = np.stack([laplacian] * 3, axis=2)
            sharpened = image_float - self.config.strength * laplacian_3ch
        else:
            sharpened = image_float - self.config.strength * laplacian

        # Clip to valid range
        if image.dtype == np.uint8:
            sharpened = np.clip(sharpened, 0, 255)
        elif image.dtype == np.uint16:
            sharpened = np.clip(sharpened, 0, 65535)
        else:
            sharpened = np.clip(sharpened, image.min(), image.max())

        return sharpened.astype(image.dtype)

    def set_strength(self, strength: float) -> None:
        """Set sharpening strength."""
        self.config.strength = max(0.0, min(3.0, strength))

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"LaplacianSharpenFilter(strength={self.config.strength:.1f}, {status})"
