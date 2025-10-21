"""
Color manipulation filters.

Includes saturation, white balance, color correction, etc.
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

from .base import Filter, FilterConfig, ColorFilter


@dataclass
class SaturationConfig(FilterConfig):
    """Configuration for saturation filter."""
    saturation: float = 1.0  # 0.0 = grayscale, 1.0 = normal, 2.0 = double

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d['saturation'] = self.saturation
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'SaturationConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'SaturationFilter'),
            saturation=data.get('saturation', 1.0),
        )


class SaturationFilter(ColorFilter):
    """
    Adjust image color saturation.

    Replaces: val_SAT, filter_enabled_SAT globals

    Example:
        >>> filter = SaturationFilter(saturation=1.5)
        >>> result = filter.process(image)
    """

    def __init__(self, saturation: float = 1.0, enabled: bool = True):
        """
        Initialize saturation filter.

        Args:
            saturation: Saturation multiplier (0.0 to 3.0)
                       0.0 = grayscale, 1.0 = no change, 2.0 = double saturation
            enabled: Whether filter is enabled
        """
        config = SaturationConfig(
            enabled=enabled,
            name="SaturationFilter",
            saturation=saturation,
        )
        super().__init__(config)
        self.config: SaturationConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply saturation adjustment.

        Args:
            image: Input color image (3 channels)

        Returns:
            Saturation-adjusted image
        """
        # Skip if saturation is 1.0 (no change)
        if abs(self.config.saturation - 1.0) < 0.001:
            return image.copy()

        try:
            import cv2
        except ImportError:
            return image

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Adjust saturation channel
        hsv[:, :, 1] = hsv[:, :, 1] * self.config.saturation

        # Clip to valid range
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        # Convert back to BGR
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result

    def set_saturation(self, saturation: float) -> None:
        """Set saturation multiplier."""
        self.config.saturation = max(0.0, min(3.0, saturation))

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"SaturationFilter(saturation={self.config.saturation:.2f}, {status})"


@dataclass
class WhiteBalanceConfig(FilterConfig):
    """Configuration for white balance filter."""
    red_balance: int = 50    # Red channel multiplier (0-100)
    blue_balance: int = 50   # Blue channel multiplier (0-100)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            'red_balance': self.red_balance,
            'blue_balance': self.blue_balance,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'WhiteBalanceConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'WhiteBalanceFilter'),
            red_balance=data.get('red_balance', 50),
            blue_balance=data.get('blue_balance', 50),
        )


class WhiteBalanceFilter(ColorFilter):
    """
    Adjust white balance by scaling red and blue channels.

    Replaces: red_equilibrium, blue_equilibrium, val_red_balance, val_blue_balance globals

    Example:
        >>> filter = WhiteBalanceFilter(red_balance=63, blue_balance=74)
        >>> result = filter.process(image)
    """

    def __init__(self, red_balance: int = 50, blue_balance: int = 50, enabled: bool = True):
        """
        Initialize white balance filter.

        Args:
            red_balance: Red channel balance (0-100, 50 = no change)
            blue_balance: Blue channel balance (0-100, 50 = no change)
            enabled: Whether filter is enabled
        """
        config = WhiteBalanceConfig(
            enabled=enabled,
            name="WhiteBalanceFilter",
            red_balance=red_balance,
            blue_balance=blue_balance,
        )
        super().__init__(config)
        self.config: WhiteBalanceConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply white balance adjustment.

        Args:
            image: Input color image (3 channels BGR)

        Returns:
            White-balanced image
        """
        # Skip if both balances are at neutral (50)
        if self.config.red_balance == 50 and self.config.blue_balance == 50:
            return image.copy()

        # Convert balance values (0-100) to multipliers (0.5-1.5)
        red_mult = 0.5 + (self.config.red_balance / 100.0)
        blue_mult = 0.5 + (self.config.blue_balance / 100.0)

        # Work with float to avoid overflow
        result = image.astype(np.float32)

        # Apply multipliers to B and R channels (BGR format)
        result[:, :, 0] = result[:, :, 0] * blue_mult   # Blue channel
        result[:, :, 2] = result[:, :, 2] * red_mult    # Red channel

        # Clip to valid range
        if image.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            result = np.clip(result, 0, 65535).astype(np.uint16)

        return result

    def set_red_balance(self, balance: int) -> None:
        """Set red channel balance (0-100)."""
        self.config.red_balance = max(0, min(100, balance))

    def set_blue_balance(self, balance: int) -> None:
        """Set blue channel balance (0-100)."""
        self.config.blue_balance = max(0, min(100, balance))

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"WhiteBalanceFilter(R={self.config.red_balance}, B={self.config.blue_balance}, {status})"


class AutoWhiteBalanceFilter(ColorFilter):
    """
    Automatic white balance using gray world assumption.

    Example:
        >>> filter = AutoWhiteBalanceFilter()
        >>> result = filter.process(image)
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize auto white balance filter.

        Args:
            enabled: Whether filter is enabled
        """
        config = FilterConfig(enabled=enabled, name="AutoWhiteBalanceFilter")
        super().__init__(config)

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply automatic white balance.

        Uses gray world algorithm: assumes average color should be gray.

        Args:
            image: Input color image (3 channels)

        Returns:
            White-balanced image
        """
        # Calculate average of each channel
        b_avg = np.mean(image[:, :, 0])
        g_avg = np.mean(image[:, :, 1])
        r_avg = np.mean(image[:, :, 2])

        # Calculate gray value
        gray_avg = (b_avg + g_avg + r_avg) / 3.0

        # Avoid division by zero
        if b_avg == 0 or g_avg == 0 or r_avg == 0:
            return image.copy()

        # Calculate scaling factors
        b_scale = gray_avg / b_avg
        g_scale = gray_avg / g_avg
        r_scale = gray_avg / r_avg

        # Apply scaling
        result = image.astype(np.float32)
        result[:, :, 0] = result[:, :, 0] * b_scale
        result[:, :, 1] = result[:, :, 1] * g_scale
        result[:, :, 2] = result[:, :, 2] * r_scale

        # Clip and convert back
        if image.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            result = np.clip(result, 0, 65535).astype(np.uint16)

        return result


@dataclass
class ColorTemperatureConfig(FilterConfig):
    """Configuration for color temperature filter."""
    temperature: float = 6500  # Color temperature in Kelvin

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d['temperature'] = self.temperature
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ColorTemperatureConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            name=data.get('name', 'ColorTemperatureFilter'),
            temperature=data.get('temperature', 6500),
        )


class ColorTemperatureFilter(ColorFilter):
    """
    Adjust color temperature (warm/cool).

    Example:
        >>> filter = ColorTemperatureFilter(temperature=5500)  # Warmer
        >>> result = filter.process(image)
    """

    def __init__(self, temperature: float = 6500, enabled: bool = True):
        """
        Initialize color temperature filter.

        Args:
            temperature: Color temperature in Kelvin (2000-12000)
                        Lower = warmer (orange), Higher = cooler (blue)
            enabled: Whether filter is enabled
        """
        config = ColorTemperatureConfig(
            enabled=enabled,
            name="ColorTemperatureFilter",
            temperature=temperature,
        )
        super().__init__(config)
        self.config: ColorTemperatureConfig

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply color temperature adjustment.

        Args:
            image: Input color image (3 channels)

        Returns:
            Temperature-adjusted image
        """
        # Calculate RGB multipliers based on temperature
        # This is a simplified approximation
        temp = self.config.temperature / 100.0

        # Red channel
        if temp <= 66:
            r = 255
        else:
            r = temp - 60
            r = 329.698727446 * (r ** -0.1332047592)
            r = max(0, min(255, r))

        # Green channel
        if temp <= 66:
            g = temp
            g = 99.4708025861 * np.log(g) - 161.1195681661
        else:
            g = temp - 60
            g = 288.1221695283 * (g ** -0.0755148492)
        g = max(0, min(255, g))

        # Blue channel
        if temp >= 66:
            b = 255
        elif temp <= 19:
            b = 0
        else:
            b = temp - 10
            b = 138.5177312231 * np.log(b) - 305.0447927307
            b = max(0, min(255, b))

        # Normalize multipliers
        r_mult = r / 255.0
        g_mult = g / 255.0
        b_mult = b / 255.0

        # Apply to image
        result = image.astype(np.float32)
        result[:, :, 0] = result[:, :, 0] * b_mult  # Blue
        result[:, :, 1] = result[:, :, 1] * g_mult  # Green
        result[:, :, 2] = result[:, :, 2] * r_mult  # Red

        # Clip and convert
        if image.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            result = np.clip(result, 0, 65535).astype(np.uint16)

        return result

    def set_temperature(self, temperature: float) -> None:
        """Set color temperature."""
        self.config.temperature = max(2000, min(12000, temperature))

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.config.enabled else "disabled"
        return f"ColorTemperatureFilter(temp={self.config.temperature:.0f}K, {status})"
