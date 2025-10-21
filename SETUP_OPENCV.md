# OpenCV Setup Guide for JETSONSKY

This guide explains how to add OpenCV capability to JETSONSKY for full filter functionality.

## Why OpenCV?

The Phase 2 filter system uses OpenCV for:
- **Advanced denoising**: KNN, bilateral filtering
- **Contrast enhancement**: CLAHE, histogram equalization
- **Image processing**: Gaussian blur, Laplacian sharpening
- **Color manipulation**: Saturation, white balance
- **Hot pixel removal**: Median filtering

Without OpenCV, most filters will return the original image unchanged.

## Quick Installation

### Method 1: Automated Script (Recommended)

```bash
# Run the installation script
./install_dependencies.sh
```

This script will:
1. Detect your operating system
2. Install NumPy and OpenCV
3. Verify the installation
4. Run tests to confirm everything works

### Method 2: Manual Installation

```bash
# Install core dependencies
pip install numpy opencv-python

# Verify installation
python3 -c "import cv2; print(f'OpenCV {cv2.__version__} installed successfully')"
```

### Method 3: Using requirements.txt

```bash
pip install -r requirements.txt
```

## Platform-Specific Instructions

### Ubuntu/Debian Linux

```bash
# Option 1: System packages (may be older version)
sudo apt-get update
sudo apt-get install python3-numpy python3-opencv

# Option 2: pip (latest version)
pip3 install numpy opencv-python

# For headless systems (no GUI)
pip3 install opencv-python-headless
```

### Windows

```bash
# Using pip
pip install numpy opencv-python

# Or with Anaconda
conda install -c conda-forge opencv
```

### macOS

```bash
# Using pip
pip3 install numpy opencv-python

# Or with Homebrew
brew install opencv
pip3 install opencv-python
```

### Jetson Nano / Jetson Xavier (ARM)

```bash
# Jetson comes with OpenCV pre-installed, but you may need to install Python bindings

# Check if OpenCV is available
python3 -c "import cv2; print(cv2.__version__)"

# If not available, install via pip
pip3 install opencv-python

# For better performance on Jetson, build from source with CUDA support:
# See: https://github.com/mdegans/nano_build_opencv
```

## Verifying OpenCV Installation

### Check Version

```bash
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

Expected output:
```
OpenCV version: 4.8.0 (or similar)
```

### Check Build Information

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())"
```

Look for:
- **CUDA**: YES (if GPU support enabled)
- **FFmpeg**: YES (for video support)
- **GUI**: QT or GTK (for windowing)

### Run Filter Tests

```bash
cd JetsonSky
python3 test_phase2.py
```

Expected output:
```
======================================================================
TESTING DENOISE FILTERS
======================================================================
✓ DenoiseKNNFilter works
✓ DenoisePaillouFilter works
✓ DenoiseGaussianFilter works

RESULTS: 10/10 tests passed
         ALL TESTS PASSED! ✓
```

## GPU Acceleration (Optional)

For 5-20x performance improvement on NVIDIA GPUs:

### Install CuPy

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# Verify installation
python3 -c "import cupy; print(f'CuPy {cupy.__version__} with {cupy.cuda.runtime.getDeviceCount()} GPU(s)')"
```

### Enable GPU Filters

```python
from filters.base import GPUFilter

# GPU filters automatically use CUDA if available
class MyGPUFilter(GPUFilter):
    def apply(self, image, **kwargs):
        # Automatically on GPU if CuPy available
        if self.cupy_available:
            result = self.cp.median_filter(image, size=3)
        else:
            # CPU fallback
            import cv2
            result = cv2.medianBlur(image, 3)
        return result
```

## Testing OpenCV Functionality

### Test Script

Create `test_opencv.py`:

```python
#!/usr/bin/env python3
"""Test OpenCV functionality for JETSONSKY filters."""

import sys

# Test imports
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not available")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError:
    print("✗ OpenCV not available")
    sys.exit(1)

# Test basic operations
print("\nTesting OpenCV operations:")

# Create test image
img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
print(f"✓ Created test image: {img.shape}")

# Test Gaussian blur
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
print(f"✓ GaussianBlur works: {blurred.shape}")

# Test median blur
median = cv2.medianBlur(img, 3)
print(f"✓ MedianBlur works: {median.shape}")

# Test CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(img)
print(f"✓ CLAHE works: {enhanced.shape}")

# Test bilateral filter
bilateral = cv2.bilateralFilter(img, 5, 75, 75)
print(f"✓ BilateralFilter works: {bilateral.shape}")

# Test color operations
img_color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
print(f"✓ Color conversion works: {hsv.shape}")

print("\n✓ All OpenCV operations working correctly!")
print("\nYou can now use all JETSONSKY filters.")
```

Run it:
```bash
python3 test_opencv.py
```

## Troubleshooting

### ImportError: No module named 'cv2'

**Solution:**
```bash
pip install opencv-python
```

### ImportError: libGL.so.1: cannot open shared object file

This occurs on headless systems (no display).

**Solution:**
```bash
# Install headless version
pip uninstall opencv-python
pip install opencv-python-headless
```

### ImportError: numpy.core.multiarray failed to import

NumPy version mismatch.

**Solution:**
```bash
pip install --upgrade numpy
pip install --force-reinstall opencv-python
```

### Slow performance / Filter timeout

**Solutions:**
1. Install GPU acceleration:
   ```bash
   pip install cupy-cuda11x  # or cupy-cuda12x
   ```

2. Reduce image size before processing:
   ```python
   # Bin image 2x2
   binned = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
   ```

3. Disable expensive filters:
   ```python
   pipeline.disable_filter("DenoiseKNNFilter")
   ```

### OpenCV built without GUI support

**Solution:**
```bash
# Install full version (not headless)
pip uninstall opencv-python-headless
pip install opencv-python
```

## Checking What's Available

### Check Filter Requirements

```bash
cd JetsonSky
python3 -c "
from filters import *
import sys

try:
    import cv2
    print(f'✓ OpenCV {cv2.__version__} available')
    print('  All filters will work')
except ImportError:
    print('✗ OpenCV not available')
    print('  Limited filter functionality:')
    print('    ✓ FlipFilter, NegativeFilter, RotateFilter')
    print('    ✗ Denoise, Sharpen, Contrast, Color, HotPixel filters')
    sys.exit(1)
"
```

## Performance Benchmarks

### With OpenCV (NumPy + OpenCV)
- Flip/Negative: < 1ms
- Hot Pixel Removal: ~10ms
- Denoise (KNN): ~100ms
- Sharpen: ~20ms
- CLAHE: ~15ms
- **Full Pipeline**: ~150ms

### With GPU (NumPy + OpenCV + CuPy)
- Flip/Negative: < 1ms
- Hot Pixel Removal: ~2ms
- Denoise (KNN): ~10ms
- Sharpen: ~3ms
- CLAHE: ~2ms
- **Full Pipeline**: ~20ms (7-8x faster)

### Without OpenCV (NumPy only)
- Flip/Negative: < 1ms
- All other filters: Return original (0ms)
- **Full Pipeline**: < 5ms (but no processing)

## Next Steps

After installing OpenCV:

1. **Run tests**:
   ```bash
   cd JetsonSky
   python3 test_phase2.py
   ```

2. **Try examples**:
   ```bash
   python3 FILTER_USAGE_EXAMPLES.py
   ```

3. **Read filter documentation**:
   ```bash
   cat filters/README.md
   ```

4. **Build your own pipeline**:
   ```python
   from filters import FilterPipeline, HotPixelFilter, DenoiseKNNFilter

   pipeline = FilterPipeline()
   pipeline.add_filter(HotPixelFilter())
   pipeline.add_filter(DenoiseKNNFilter())

   result = pipeline.apply(my_astronomy_image)
   ```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `python3 test_opencv.py` to diagnose
3. Check OpenCV installation: `python3 -c "import cv2; print(cv2.getBuildInformation())"`
4. Verify NumPy: `python3 -c "import numpy; print(numpy.__version__)"`
5. See filter test results: `python3 JetsonSky/test_phase2.py`

## Summary

- **Minimum**: `pip install numpy opencv-python`
- **Recommended**: `pip install numpy opencv-python cupy-cuda11x`
- **Quick test**: `python3 test_opencv.py`
- **Full test**: `python3 JetsonSky/test_phase2.py`

**Once installed, all JETSONSKY filters will work at full capability!**
