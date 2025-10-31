"""
Image processing filters module.

This module provides modular, testable filter classes that replace the
monolithic application_filtrage_color() and application_filtrage_mono() functions.

Example usage:
    >>> from filters import FilterPipeline, DenoiseKNNFilter, SharpenFilter
    >>> pipeline = FilterPipeline()
    >>> pipeline.add_filter(DenoiseKNNFilter(strength=0.3))
    >>> pipeline.add_filter(SharpenFilter(amount=1.5))
    >>> result = pipeline.apply(image)
"""

from .base import (
    Filter,
    FilterConfig,
    ColorFilter,
    MonoFilter,
    UniversalFilter,
    GPUFilter,
)

from .pipeline import (
    FilterPipeline,
    ColorPipeline,
    MonoPipeline,
    create_default_color_pipeline,
    create_default_mono_pipeline,
)

# Import filter implementations
from .denoise import (
    DenoiseKNNFilter,
    DenoisePaillouFilter,
    DenoiseGaussianFilter,
)

from .sharpen import (
    SharpenFilter,
    LaplacianSharpenFilter,
)

from .contrast import (
    CLAHEFilter,
    HistogramEqualizeFilter,
    GammaCorrectionFilter,
)

from .color import (
    SaturationFilter,
    WhiteBalanceFilter,
    AutoWhiteBalanceFilter,
    ColorTemperatureFilter,
)

from .hotpixel import (
    HotPixelFilter,
    DeadPixelFilter,
    BadPixelMapFilter,
)

from .transforms import (
    FlipFilter,
    NegativeFilter,
    RotateFilter,
)


__all__ = [
    # Base classes
    'Filter',
    'FilterConfig',
    'ColorFilter',
    'MonoFilter',
    'UniversalFilter',
    'GPUFilter',

    # Pipeline
    'FilterPipeline',
    'ColorPipeline',
    'MonoPipeline',
    'create_default_color_pipeline',
    'create_default_mono_pipeline',

    # Denoise filters
    'DenoiseKNNFilter',
    'DenoisePaillouFilter',
    'DenoiseGaussianFilter',

    # Sharpen filters
    'SharpenFilter',
    'LaplacianSharpenFilter',

    # Contrast filters
    'CLAHEFilter',
    'HistogramEqualizeFilter',
    'GammaCorrectionFilter',

    # Color filters
    'SaturationFilter',
    'WhiteBalanceFilter',
    'AutoWhiteBalanceFilter',
    'ColorTemperatureFilter',

    # Hot pixel filters
    'HotPixelFilter',
    'DeadPixelFilter',
    'BadPixelMapFilter',

    # Transform filters
    'FlipFilter',
    'NegativeFilter',
    'RotateFilter',
]
