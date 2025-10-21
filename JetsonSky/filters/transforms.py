"""
Basic image transformation filters.

Includes flip, rotate, negative, and other simple transformations.
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
class FlipConfig(FilterConfig):
    """Configuration for flip filter."""
    vertical: bool = False
    horizontal: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            'vertical': self.vertical,
            'horizontal': self.horizontal,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'FlipConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'FlipFilter'),
            vertical=data.get('vertical', False),
            horizontal=data.get('horizontal', False),
        )


class FlipFilter(UniversalFilter):
    """
    Flip image vertically and/or horizontally.

    Replaces the FlipV and FlipH global variable logic.

    Example:
        >>> filter = FlipFilter(vertical=True, horizontal=False)
        >>> result = filter.process(image)
    """

    def __init__(self, vertical: bool = False, horizontal: bool = False, enabled: bool = True):
        """
        Initialize flip filter.

        Args:
            vertical: Flip vertically (upside down)
            horizontal: Flip horizontally (left-right mirror)
            enabled: Whether filter is enabled
        """
        config = FlipConfig(
            enabled=enabled,
            name="FlipFilter",
            vertical=vertical,
            horizontal=horizontal
        )
        super().__init__(config)
        self.config: FlipConfig  # Type hint for IDE support

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply flip transformation.

        Args:
            image: Input image (2D or 3D)

        Returns:
            Flipped image
        """
        result = image.copy()

        # Flip vertically (axis 0)
        if self.config.vertical:
            result = np.flip(result, axis=0)

        # Flip horizontally (axis 1)
        if self.config.horizontal:
            result = np.flip(result, axis=1)

        return result

    def set_vertical(self, vertical: bool) -> None:
        """Set vertical flip state."""
        self.config.vertical = vertical

    def set_horizontal(self, horizontal: bool) -> None:
        """Set horizontal flip state."""
        self.config.horizontal = horizontal

    def __repr__(self) -> str:
        """String representation."""
        flips = []
        if self.config.vertical:
            flips.append("V")
        if self.config.horizontal:
            flips.append("H")
        flip_str = "+".join(flips) if flips else "none"
        status = "enabled" if self.config.enabled else "disabled"
        return f"FlipFilter({flip_str}, {status})"


class NegativeFilter(UniversalFilter):
    """
    Invert image values (negative).

    Replaces the image_negative global variable logic.

    For 8-bit images: new_value = 255 - old_value
    For 16-bit images: new_value = 65535 - old_value

    Example:
        >>> filter = NegativeFilter()
        >>> result = filter.process(image)
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize negative filter.

        Args:
            enabled: Whether filter is enabled
        """
        config = FilterConfig(enabled=enabled, name="NegativeFilter")
        super().__init__(config)

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply negative transformation.

        Args:
            image: Input image (2D or 3D)

        Returns:
            Inverted image
        """
        # Determine bit depth from dtype
        if image.dtype == np.uint8:
            max_value = 255
        elif image.dtype == np.uint16:
            max_value = 65535
        else:
            # For float images, assume 0-1 range
            if image.max() <= 1.0:
                return 1.0 - image
            else:
                max_value = int(image.max())

        return max_value - image

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"NegativeFilter({status})"


@dataclass
class RotateConfig(FilterConfig):
    """Configuration for rotate filter."""
    angle: int = 0  # 0, 90, 180, 270

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d['angle'] = self.angle
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'RotateConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'RotateFilter'),
            angle=data.get('angle', 0),
        )


class RotateFilter(UniversalFilter):
    """
    Rotate image by 90-degree increments.

    Example:
        >>> filter = RotateFilter(angle=90)
        >>> result = filter.process(image)
    """

    def __init__(self, angle: int = 0, enabled: bool = True):
        """
        Initialize rotate filter.

        Args:
            angle: Rotation angle (0, 90, 180, or 270)
            enabled: Whether filter is enabled
        """
        if angle not in [0, 90, 180, 270]:
            raise ValueError("Angle must be 0, 90, 180, or 270")

        config = RotateConfig(
            enabled=enabled,
            name="RotateFilter",
            angle=angle
        )
        super().__init__(config)
        self.config: RotateConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply rotation.

        Args:
            image: Input image (2D or 3D)

        Returns:
            Rotated image
        """
        if self.config.angle == 0:
            return image.copy()
        elif self.config.angle == 90:
            return np.rot90(image, k=1)
        elif self.config.angle == 180:
            return np.rot90(image, k=2)
        elif self.config.angle == 270:
            return np.rot90(image, k=3)
        else:
            return image.copy()

    def set_angle(self, angle: int) -> None:
        """
        Set rotation angle.

        Args:
            angle: Rotation angle (0, 90, 180, or 270)
        """
        if angle not in [0, 90, 180, 270]:
            raise ValueError("Angle must be 0, 90, 180, or 270")
        self.config.angle = angle

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"RotateFilter({self.config.angle}°, {status})"
