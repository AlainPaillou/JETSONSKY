#!/usr/bin/env python3
"""
Test script for Phase 2 refactoring - Filter modules.

This script verifies that the filter system works correctly.
"""

import sys
import os

# Add JetsonSky directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all filter modules can be imported."""
    print("=" * 70)
    print("TESTING FILTER IMPORTS")
    print("=" * 70)

    try:
        from filters import (
            Filter, FilterConfig, FilterPipeline,
            ColorFilter, MonoFilter, UniversalFilter,
            DenoiseKNNFilter, DenoisePaillouFilter,
            SharpenFilter, CLAHEFilter,
            SaturationFilter, WhiteBalanceFilter,
            HotPixelFilter, FlipFilter, NegativeFilter
        )
        print("✓ All filter modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import filter modules: {e}")
        return False

    print()
    return True


def test_base_filter_classes():
    """Test filter base classes."""
    print("=" * 70)
    print("TESTING BASE FILTER CLASSES")
    print("=" * 70)

    from filters.base import Filter, FilterConfig, UniversalFilter

    # Test FilterConfig
    config = FilterConfig(enabled=True, name="TestFilter")
    assert config.enabled == True
    assert config.name == "TestFilter"
    print("✓ FilterConfig creation works")

    # Test config serialization
    config_dict = config.to_dict()
    assert config_dict['enabled'] == True
    assert config_dict['name'] == "TestFilter"
    print("✓ FilterConfig serialization works")

    # Test config deserialization
    config2 = FilterConfig.from_dict(config_dict)
    assert config2.enabled == config.enabled
    assert config2.name == config.name
    print("✓ FilterConfig deserialization works")

    # Create simple test filter
    class SimpleFilter(UniversalFilter):
        def apply(self, image, **kwargs):
            # Return a mock processed image
            return MockImage(image.width, image.height)

    # Create mock image class for testing without numpy
    class MockImage:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.shape = (height, width)

    # Test filter
    test_filter = SimpleFilter()
    test_image = MockImage(100, 100)
    result = test_filter.process(test_image)
    assert result.shape == test_image.shape
    print("✓ Basic filter processing works")

    # Test enable/disable
    test_filter.disable()
    assert not test_filter.is_enabled()
    result = test_filter.process(test_image)
    assert result == test_image  # Should return original when disabled
    print("✓ Filter enable/disable works")

    test_filter.enable()
    assert test_filter.is_enabled()
    print("✓ Filter enable works")

    print()
    return True


def test_filter_pipeline():
    """Test FilterPipeline functionality."""
    print("=" * 70)
    print("TESTING FILTER PIPELINE")
    print("=" * 70)

    from filters import FilterPipeline
    from filters.base import UniversalFilter

    # Create mock image class
    class MockImage:
        def __init__(self, value):
            self.value = value
            self.shape = (10, 10)

        def __add__(self, other):
            return MockImage(self.value + other)

        def __mul__(self, other):
            return MockImage(self.value * other)

        def __eq__(self, other):
            if isinstance(other, MockImage):
                return self.value == other.value
            return False

    # Create test filters
    class AddFilter(UniversalFilter):
        def __init__(self, value):
            super().__init__()
            self.value = value
            self.config.name = f"Add{value}"

        def apply(self, image, **kwargs):
            return image + self.value

    class MultiplyFilter(UniversalFilter):
        def __init__(self, value):
            super().__init__()
            self.value = value
            self.config.name = f"Multiply{value}"

        def apply(self, image, **kwargs):
            return image * self.value

    # Create pipeline
    pipeline = FilterPipeline()
    assert len(pipeline) == 0
    print("✓ Empty pipeline created")

    # Add filters
    pipeline.add_filter(AddFilter(10))
    pipeline.add_filter(MultiplyFilter(2))
    assert len(pipeline) == 2
    print("✓ Filters added to pipeline")

    # Test pipeline application
    test_image = MockImage(1)
    result = pipeline.apply(test_image)
    # Should be: (1 + 10) * 2 = 22
    assert result.value == 22
    print("✓ Pipeline applies filters in correct order")

    # Test disabling a filter
    pipeline.disable_filter("Add10")
    result = pipeline.apply(test_image)
    # Should be: 1 * 2 = 2
    assert result.value == 2
    print("✓ Pipeline filter disable works")

    # Test re-enabling
    pipeline.enable_filter("Add10")
    result = pipeline.apply(test_image)
    assert result.value == 22
    print("✓ Pipeline filter enable works")

    # Test removing filter
    pipeline.remove_filter("Add10")
    assert len(pipeline) == 1
    print("✓ Filter removal works")

    # Test clear
    pipeline.clear()
    assert len(pipeline) == 0
    print("✓ Pipeline clear works")

    print()
    return True


def test_transform_filters():
    """Test transform filters (flip, negative, rotate)."""
    print("=" * 70)
    print("TESTING TRANSFORM FILTERS")
    print("=" * 70)

    try:
        import numpy as np
    except ImportError:
        print("⚠ NumPy not available - skipping transform filter tests")
        return True

    from filters import FlipFilter, NegativeFilter, RotateFilter

    # Create test image
    test_image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.uint8)

    # Test FlipFilter - vertical
    flip_v = FlipFilter(vertical=True, horizontal=False)
    result = flip_v.process(test_image)
    expected = np.array([
        [7, 8, 9],
        [4, 5, 6],
        [1, 2, 3]
    ], dtype=np.uint8)
    assert np.array_equal(result, expected)
    print("✓ Vertical flip works")

    # Test FlipFilter - horizontal
    flip_h = FlipFilter(vertical=False, horizontal=True)
    result = flip_h.process(test_image)
    expected = np.array([
        [3, 2, 1],
        [6, 5, 4],
        [9, 8, 7]
    ], dtype=np.uint8)
    assert np.array_equal(result, expected)
    print("✓ Horizontal flip works")

    # Test NegativeFilter
    negative = NegativeFilter()
    result = negative.process(test_image)
    expected = 255 - test_image
    assert np.array_equal(result, expected)
    print("✓ Negative filter works")

    # Test RotateFilter
    rotate_90 = RotateFilter(angle=90)
    result = rotate_90.process(test_image)
    assert result.shape == (3, 3)
    print("✓ Rotate 90° works")

    rotate_180 = RotateFilter(angle=180)
    result = rotate_180.process(test_image)
    expected = np.array([
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ], dtype=np.uint8)
    assert np.array_equal(result, expected)
    print("✓ Rotate 180° works")

    print()
    return True


def test_denoise_filters():
    """Test denoise filters."""
    print("=" * 70)
    print("TESTING DENOISE FILTERS")
    print("=" * 70)

    try:
        import numpy as np
        import cv2
        has_opencv = True
    except ImportError:
        print("⚠ OpenCV not available - skipping denoise filter tests")
        return True

    from filters import DenoiseKNNFilter, DenoisePaillouFilter, DenoiseGaussianFilter

    # Create noisy test image
    np.random.seed(42)
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # Test DenoiseKNNFilter
    denoise_knn = DenoiseKNNFilter(strength=0.4)
    result = denoise_knn.process(test_image)
    assert result.shape == test_image.shape
    assert result.dtype == test_image.dtype
    print("✓ DenoiseKNNFilter works")

    # Test DenoisePaillouFilter
    denoise_paillou = DenoisePaillouFilter(strength=0.4)
    result = denoise_paillou.process(test_image)
    assert result.shape == test_image.shape
    print("✓ DenoisePaillouFilter works")

    # Test DenoiseGaussianFilter
    denoise_gaussian = DenoiseGaussianFilter(sigma=1.0)
    result = denoise_gaussian.process(test_image)
    assert result.shape == test_image.shape
    print("✓ DenoiseGaussianFilter works")

    print()
    return True


def test_sharpen_filters():
    """Test sharpen filters."""
    print("=" * 70)
    print("TESTING SHARPEN FILTERS")
    print("=" * 70)

    try:
        import numpy as np
        import cv2
    except ImportError:
        print("⚠ OpenCV not available - skipping sharpen filter tests")
        return True

    from filters import SharpenFilter, LaplacianSharpenFilter

    # Create test image
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # Test SharpenFilter
    sharpen = SharpenFilter(amount=1.5, sigma=1.0)
    result = sharpen.process(test_image)
    assert result.shape == test_image.shape
    assert result.dtype == test_image.dtype
    print("✓ SharpenFilter works")

    # Test LaplacianSharpenFilter
    laplacian = LaplacianSharpenFilter(strength=1.0)
    result = laplacian.process(test_image)
    assert result.shape == test_image.shape
    print("✓ LaplacianSharpenFilter works")

    print()
    return True


def test_contrast_filters():
    """Test contrast filters."""
    print("=" * 70)
    print("TESTING CONTRAST FILTERS")
    print("=" * 70)

    try:
        import numpy as np
        import cv2
    except ImportError:
        print("⚠ OpenCV not available - skipping contrast filter tests")
        return True

    from filters import CLAHEFilter, HistogramEqualizeFilter, GammaCorrectionFilter

    # Create test image
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # Test CLAHEFilter
    clahe = CLAHEFilter(clip_limit=2.0, grid_size=8)
    result = clahe.process(test_image)
    assert result.shape == test_image.shape
    assert result.dtype == test_image.dtype
    print("✓ CLAHEFilter works")

    # Test HistogramEqualizeFilter
    hist_eq = HistogramEqualizeFilter()
    result = hist_eq.process(test_image)
    assert result.shape == test_image.shape
    print("✓ HistogramEqualizeFilter works")

    # Test GammaCorrectionFilter
    gamma = GammaCorrectionFilter(gamma=1.2)
    result = gamma.process(test_image)
    assert result.shape == test_image.shape
    print("✓ GammaCorrectionFilter works")

    print()
    return True


def test_color_filters():
    """Test color filters."""
    print("=" * 70)
    print("TESTING COLOR FILTERS")
    print("=" * 70)

    try:
        import numpy as np
        import cv2
    except ImportError:
        print("⚠ OpenCV not available - skipping color filter tests")
        return True

    from filters import SaturationFilter, WhiteBalanceFilter, AutoWhiteBalanceFilter

    # Create test color image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Test SaturationFilter
    saturation = SaturationFilter(saturation=1.5)
    result = saturation.process(test_image)
    assert result.shape == test_image.shape
    assert result.dtype == test_image.dtype
    print("✓ SaturationFilter works")

    # Test WhiteBalanceFilter
    wb = WhiteBalanceFilter(red_balance=63, blue_balance=74)
    result = wb.process(test_image)
    assert result.shape == test_image.shape
    print("✓ WhiteBalanceFilter works")

    # Test AutoWhiteBalanceFilter
    auto_wb = AutoWhiteBalanceFilter()
    result = auto_wb.process(test_image)
    assert result.shape == test_image.shape
    print("✓ AutoWhiteBalanceFilter works")

    print()
    return True


def test_hotpixel_filters():
    """Test hot pixel filters."""
    print("=" * 70)
    print("TESTING HOT PIXEL FILTERS")
    print("=" * 70)

    try:
        import numpy as np
        import cv2
    except ImportError:
        print("⚠ OpenCV not available - skipping hot pixel filter tests")
        return True

    from filters import HotPixelFilter, DeadPixelFilter, BadPixelMapFilter

    # Create test image with hot pixels
    test_image = np.random.randint(50, 100, (100, 100), dtype=np.uint8)
    # Add some hot pixels
    test_image[10, 10] = 255
    test_image[50, 50] = 255

    # Test HotPixelFilter
    hotpixel = HotPixelFilter(threshold=0.9)
    result = hotpixel.process(test_image)
    assert result.shape == test_image.shape
    assert result[10, 10] < test_image[10, 10]  # Hot pixel should be reduced
    print("✓ HotPixelFilter works")

    # Test DeadPixelFilter
    # Create image with dead pixels
    test_image2 = np.random.randint(150, 200, (100, 100), dtype=np.uint8)
    test_image2[20, 20] = 0
    deadpixel = DeadPixelFilter(threshold=0.9)
    result = deadpixel.process(test_image2)
    assert result.shape == test_image2.shape
    assert result[20, 20] > test_image2[20, 20]  # Dead pixel should be increased
    print("✓ DeadPixelFilter works")

    # Test BadPixelMapFilter
    bad_pixel_map = np.zeros((100, 100), dtype=bool)
    bad_pixel_map[30, 30] = True
    badpixel_map = BadPixelMapFilter(bad_pixel_map)
    result = badpixel_map.process(test_image)
    assert result.shape == test_image.shape
    print("✓ BadPixelMapFilter works")

    print()
    return True


def test_full_pipeline():
    """Test complete filter pipeline."""
    print("=" * 70)
    print("TESTING FULL FILTER PIPELINE")
    print("=" * 70)

    try:
        import numpy as np
        import cv2
    except ImportError:
        print("⚠ OpenCV not available - skipping full pipeline test")
        return True

    from filters import (
        FilterPipeline,
        HotPixelFilter,
        DenoiseKNNFilter,
        SharpenFilter,
        CLAHEFilter,
    )

    # Create test image
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # Build pipeline (simulating typical astronomy processing)
    pipeline = FilterPipeline()
    pipeline.add_filter(HotPixelFilter(threshold=0.9))
    pipeline.add_filter(DenoiseKNNFilter(strength=0.3))
    pipeline.add_filter(SharpenFilter(amount=1.5))
    pipeline.add_filter(CLAHEFilter(clip_limit=2.0))

    assert len(pipeline) == 4
    assert pipeline.get_enabled_count() == 4
    print(f"✓ Pipeline created with {len(pipeline)} filters")

    # Apply pipeline
    result = pipeline.apply(test_image)
    assert result.shape == test_image.shape
    print("✓ Pipeline processes image correctly")

    # Apply with stats
    result = pipeline.apply(test_image, collect_stats=True)
    stats = pipeline.get_stats()
    assert len(stats) == 4
    print("✓ Pipeline statistics collection works")

    # Disable a filter
    pipeline.disable_filter("CLAHEFilter")
    assert pipeline.get_enabled_count() == 3
    result = pipeline.apply(test_image)
    print("✓ Pipeline with disabled filter works")

    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" PHASE 2 FILTER MODULE TESTING")
    print("=" * 70)
    print()

    tests = [
        ("Imports", test_imports),
        ("Base Filter Classes", test_base_filter_classes),
        ("Filter Pipeline", test_filter_pipeline),
        ("Transform Filters", test_transform_filters),
        ("Denoise Filters", test_denoise_filters),
        ("Sharpen Filters", test_sharpen_filters),
        ("Contrast Filters", test_contrast_filters),
        ("Color Filters", test_color_filters),
        ("Hot Pixel Filters", test_hotpixel_filters),
        ("Full Pipeline", test_full_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} test failed")
        except Exception as e:
            failed += 1
            print(f"✗ {name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed}/{len(tests)} tests FAILED")
    else:
        print("         ALL TESTS PASSED! ✓")
    print("=" * 70)
    print()

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
