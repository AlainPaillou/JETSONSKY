#!/usr/bin/env python3
"""
Filter System Usage Examples - Phase 2

This file demonstrates how to use the new modular filter system that replaces
the monolithic application_filtrage_color() and application_filtrage_mono() functions.

The new system provides:
- Modular, testable filter classes
- FilterPipeline for sequential processing
- Easy enable/disable of individual filters
- Performance statistics collection
- GPU acceleration support (with CuPy)
"""

import sys
import os

# Add JetsonSky directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check for dependencies
try:
    import numpy as np
    import cv2
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


def example_1_basic_filter_usage():
    """Example 1: Basic filter usage."""
    print("=" * 80)
    print("EXAMPLE 1: BASIC FILTER USAGE")
    print("=" * 80)

    if not HAS_DEPENDENCIES:
        print("⚠ NumPy/OpenCV not available - skipping this example")
        return

    from filters import FlipFilter, NegativeFilter

    # Create test image
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    print(f"Created test image: {test_image.shape}, dtype={test_image.dtype}")

    # Apply flip filter
    flip_filter = FlipFilter(vertical=True, horizontal=False)
    flipped = flip_filter.process(test_image)
    print(f"✓ Vertical flip applied: {flipped.shape}")

    # Apply negative filter
    negative_filter = NegativeFilter()
    negated = negative_filter.process(test_image)
    print(f"✓ Negative filter applied: {negated.shape}")

    # Disable and re-enable
    flip_filter.disable()
    result = flip_filter.process(test_image)
    print(f"✓ Disabled filter returns original image: {np.array_equal(result, test_image)}")

    flip_filter.enable()
    result = flip_filter.process(test_image)
    print(f"✓ Re-enabled filter works again")

    print()


def example_2_filter_pipeline():
    """Example 2: Building a filter pipeline."""
    print("=" * 80)
    print("EXAMPLE 2: FILTER PIPELINE")
    print("=" * 80)

    if not HAS_DEPENDENCIES:
        print("⚠ NumPy/OpenCV not available - skipping this example")
        return

    from filters import (
        FilterPipeline,
        HotPixelFilter,
        DenoiseKNNFilter,
        SharpenFilter,
        CLAHEFilter,
    )

    # Create test image
    test_image = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
    # Add some hot pixels
    test_image[10:15, 10:15] = 255

    print(f"Created test image with hot pixels: {test_image.shape}")

    # Build pipeline (typical astronomy processing chain)
    pipeline = FilterPipeline()
    pipeline.add_filter(HotPixelFilter(threshold=0.9))
    pipeline.add_filter(DenoiseKNNFilter(strength=0.3))
    pipeline.add_filter(SharpenFilter(amount=1.5))
    pipeline.add_filter(CLAHEFilter(clip_limit=2.0))

    print(f"✓ Created pipeline with {len(pipeline)} filters")
    print(pipeline)

    # Apply pipeline
    result = pipeline.apply(test_image)
    print(f"✓ Pipeline applied: {result.shape}")

    # Apply with statistics
    result = pipeline.apply(test_image, collect_stats=True)
    print(f"✓ Pipeline with stats:")
    pipeline.print_stats()

    print()


def example_3_old_vs_new_comparison():
    """Example 3: OLD monolithic code vs NEW modular approach."""
    print("=" * 80)
    print("EXAMPLE 3: OLD VS NEW COMPARISON")
    print("=" * 80)

    print("\nOLD WAY (Monolithic):")
    print("-" * 80)
    print("""
def application_filtrage_color(image):
    # ~1,000 lines of if-elif-else checking globals
    global filter_enabled_hotpix, filter_enabled_denoise_KNN
    global filter_enabled_sharpen1, filter_enabled_CLAHE
    global val_denoise_KNN, val_sharpen, contrast_CLAHE

    if filter_enabled_hotpix == 1:
        # 50 lines of hot pixel removal
        ...

    if filter_enabled_denoise_KNN == 1:
        # 50 lines of KNN denoise
        ...

    if filter_enabled_sharpen1 == 1:
        # 50 lines of sharpening
        ...

    if filter_enabled_CLAHE == 1:
        # 50 lines of CLAHE
        ...

    return image

# Problems:
# - 300+ global variables
# - 1,000+ lines in one function
# - Hard to test individual filters
# - Hard to add/remove filters
# - No performance profiling
# - No GPU support
    """)

    print("\nNEW WAY (Modular):")
    print("-" * 80)
    print("""
from filters import (
    FilterPipeline,
    HotPixelFilter,
    DenoiseKNNFilter,
    SharpenFilter,
    CLAHEFilter,
)

# Create pipeline - clean, readable
pipeline = FilterPipeline()
pipeline.add_filter(HotPixelFilter(threshold=0.9))
pipeline.add_filter(DenoiseKNNFilter(strength=0.3))
pipeline.add_filter(SharpenFilter(amount=1.5))
pipeline.add_filter(CLAHEFilter(clip_limit=2.0))

# Apply - one line!
result = pipeline.apply(image, collect_stats=True)

# Benefits:
# ✓ 0 global variables
# ✓ Each filter is 50-100 lines, testable
# ✓ Easy to add/remove/reorder filters
# ✓ Built-in performance profiling
# ✓ GPU support with CuPy
# ✓ Dynamic enable/disable
# ✓ JSON serializable configuration
    """)

    print()


def example_4_color_image_processing():
    """Example 4: Color image processing pipeline."""
    print("=" * 80)
    print("EXAMPLE 4: COLOR IMAGE PROCESSING")
    print("=" * 80)

    if not HAS_DEPENDENCIES:
        print("⚠ NumPy/OpenCV not available - skipping this example")
        return

    from filters import (
        ColorPipeline,
        DenoiseKNNFilter,
        SharpenFilter,
        CLAHEFilter,
        SaturationFilter,
        WhiteBalanceFilter,
    )

    # Create color test image
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    print(f"Created color test image: {test_image.shape}")

    # Build color processing pipeline
    pipeline = ColorPipeline()
    pipeline.add_filter(DenoiseKNNFilter(strength=0.2))
    pipeline.add_filter(WhiteBalanceFilter(red_balance=63, blue_balance=74))
    pipeline.add_filter(SharpenFilter(amount=1.2))
    pipeline.add_filter(CLAHEFilter(clip_limit=2.0))
    pipeline.add_filter(SaturationFilter(saturation=1.3))

    print(f"✓ Created color pipeline with {len(pipeline)} filters")

    # Apply pipeline
    result = pipeline.apply(test_image, collect_stats=True)
    print(f"✓ Processed color image: {result.shape}")

    print("\nPerformance breakdown:")
    pipeline.print_stats()

    print()


def example_5_dynamic_filter_control():
    """Example 5: Dynamically controlling filters."""
    print("=" * 80)
    print("EXAMPLE 5: DYNAMIC FILTER CONTROL")
    print("=" * 80)

    if not HAS_DEPENDENCIES:
        print("⚠ NumPy/OpenCV not available - skipping this example")
        return

    from filters import (
        FilterPipeline,
        DenoiseKNNFilter,
        SharpenFilter,
        CLAHEFilter,
    )

    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # Create pipeline
    pipeline = FilterPipeline()
    pipeline.add_filter(DenoiseKNNFilter(strength=0.4))
    pipeline.add_filter(SharpenFilter(amount=1.5))
    pipeline.add_filter(CLAHEFilter(clip_limit=2.0))

    print(f"Created pipeline: {len(pipeline)} filters, {pipeline.get_enabled_count()} enabled")

    # Disable sharpening
    pipeline.disable_filter("SharpenFilter")
    print(f"✓ Disabled sharpening: {pipeline.get_enabled_count()} enabled")

    # Re-enable
    pipeline.enable_filter("SharpenFilter")
    print(f"✓ Re-enabled sharpening: {pipeline.get_enabled_count()} enabled")

    # Get specific filter and modify parameters
    sharpen = pipeline.get_filter("SharpenFilter")
    if sharpen:
        sharpen.set_amount(2.0)
        print(f"✓ Modified sharpen amount to 2.0")

    # Remove a filter
    pipeline.remove_filter("DenoiseKNNFilter")
    print(f"✓ Removed denoise: {len(pipeline)} filters remaining")

    print()


def example_6_custom_filter():
    """Example 6: Creating a custom filter."""
    print("=" * 80)
    print("EXAMPLE 6: CREATING CUSTOM FILTERS")
    print("=" * 80)

    if not HAS_DEPENDENCIES:
        print("⚠ NumPy/OpenCV not available - skipping this example")
        return

    from filters.base import UniversalFilter, FilterConfig
    from filters import FilterPipeline
    from dataclasses import dataclass

    # Define custom filter configuration
    @dataclass
    class BrightnessConfig(FilterConfig):
        brightness: int = 0  # -100 to +100

        def to_dict(self):
            d = super().to_dict()
            d['brightness'] = self.brightness
            return d

    # Define custom filter
    class BrightnessFilter(UniversalFilter):
        """Custom filter to adjust brightness."""

        def __init__(self, brightness: int = 0, enabled: bool = True):
            config = BrightnessConfig(
                enabled=enabled,
                name="BrightnessFilter",
                brightness=brightness
            )
            super().__init__(config)
            self.config: BrightnessConfig

        def apply(self, image, **kwargs):
            """Apply brightness adjustment."""
            # Convert to float for calculation
            result = image.astype(np.float32)
            result += self.config.brightness

            # Clip to valid range
            if image.dtype == np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
            elif image.dtype == np.uint16:
                result = np.clip(result, 0, 65535).astype(np.uint16)

            return result

    # Use custom filter
    test_image = np.ones((100, 100), dtype=np.uint8) * 128
    brightness_filter = BrightnessFilter(brightness=50)

    result = brightness_filter.process(test_image)
    print(f"✓ Custom brightness filter applied")
    print(f"  Original average: {np.mean(test_image):.1f}")
    print(f"  Result average: {np.mean(result):.1f}")

    # Use in pipeline
    pipeline = FilterPipeline()
    pipeline.add_filter(BrightnessFilter(brightness=30))
    pipeline.add_filter(brightness_filter)

    print(f"✓ Custom filter works in pipeline!")

    print()


def example_7_astronomy_workflow():
    """Example 7: Complete astronomy imaging workflow."""
    print("=" * 80)
    print("EXAMPLE 7: ASTRONOMY IMAGING WORKFLOW")
    print("=" * 80)

    if not HAS_DEPENDENCIES:
        print("⚠ NumPy/OpenCV not available - skipping this example")
        return

    from filters import (
        FilterPipeline,
        HotPixelFilter,
        DenoiseKNNFilter,
        DenoisePaillouFilter,
        SharpenFilter,
        CLAHEFilter,
        GammaCorrectionFilter,
    )

    # Simulate raw astronomy image
    np.random.seed(42)
    raw_image = np.random.randint(100, 150, (512, 512), dtype=np.uint16)
    # Add some stars (bright spots)
    for _ in range(50):
        x, y = np.random.randint(10, 502, 2)
        raw_image[y-2:y+2, x-2:x+2] = 50000
    # Add hot pixels
    hot_pixels = np.random.randint(0, 512, (20, 2))
    for x, y in hot_pixels:
        raw_image[y, x] = 65000

    print(f"Created simulated astronomy image: {raw_image.shape}, {raw_image.dtype}")
    print(f"  Value range: {raw_image.min()} - {raw_image.max()}")

    # Build professional processing pipeline
    pipeline = FilterPipeline()

    # 1. Remove hot pixels
    pipeline.add_filter(HotPixelFilter(
        threshold=0.95,
        kernel_size=3
    ))

    # 2. First denoise pass (KNN)
    pipeline.add_filter(DenoiseKNNFilter(
        strength=0.2,
        knn_strength=0.15
    ))

    # 3. Second denoise pass (Paillou/bilateral)
    pipeline.add_filter(DenoisePaillouFilter(
        strength=0.25
    ))

    # 4. Sharpen to enhance star detail
    pipeline.add_filter(SharpenFilter(
        amount=1.8,
        sigma=0.8
    ))

    # 5. Enhance local contrast
    pipeline.add_filter(CLAHEFilter(
        clip_limit=2.5,
        grid_size=8
    ))

    # 6. Gamma correction for visualization
    pipeline.add_filter(GammaCorrectionFilter(
        gamma=0.8  # Brighten
    ))

    print(f"\n✓ Created {len(pipeline)}-stage processing pipeline:")
    for i, f in enumerate(pipeline.filters, 1):
        print(f"  {i}. {f.get_name()}")

    # Process image with timing
    print("\nProcessing...")
    result = pipeline.apply(raw_image, collect_stats=True)

    print(f"\n✓ Processed image: {result.shape}, {result.dtype}")
    print(f"  Value range: {result.min()} - {result.max()}")

    print("\nPerformance breakdown:")
    pipeline.print_stats()

    print("\n✓ Complete astronomy workflow executed successfully!")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" FILTER SYSTEM USAGE EXAMPLES - PHASE 2")
    print("=" * 80)
    print()

    if not HAS_DEPENDENCIES:
        print("⚠ WARNING: NumPy and/or OpenCV not installed")
        print("           Most examples will be skipped")
        print("           Install with: pip install numpy opencv-python")
        print()

    examples = [
        example_1_basic_filter_usage,
        example_2_filter_pipeline,
        example_3_old_vs_new_comparison,
        example_4_color_image_processing,
        example_5_dynamic_filter_control,
        example_6_custom_filter,
        example_7_astronomy_workflow,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"✗ Example failed: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print(" ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print()

    print("Key Takeaways:")
    print("  ✓ Modular filters replace 1,000+ line monolithic functions")
    print("  ✓ FilterPipeline provides clean, testable processing chain")
    print("  ✓ Easy to enable/disable/reorder filters dynamically")
    print("  ✓ Built-in performance profiling")
    print("  ✓ Custom filters are easy to create")
    print("  ✓ Professional astronomy workflows are now maintainable")
    print()


if __name__ == "__main__":
    main()
