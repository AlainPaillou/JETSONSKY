#!/bin/bash
# JETSONSKY Dependency Installation Script

echo "=========================================="
echo "JETSONSKY Dependency Installation"
echo "=========================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="Windows"
else
    OS="Unknown"
fi

echo "Detected OS: $OS"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Function to check if package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Install core dependencies
echo "Installing core dependencies..."
echo "  - numpy"
echo "  - opencv-python"
echo ""

pip3 install -r requirements.txt

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

# Check NumPy
if check_package numpy; then
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
    echo "✓ NumPy installed: $NUMPY_VERSION"
else
    echo "✗ NumPy installation FAILED"
fi

# Check OpenCV
if check_package cv2; then
    CV_VERSION=$(python3 -c "import cv2; print(cv2.__version__)")
    echo "✓ OpenCV installed: $CV_VERSION"
else
    echo "✗ OpenCV installation FAILED"
fi

# Check CuPy (optional)
if check_package cupy; then
    CUPY_VERSION=$(python3 -c "import cupy; print(cupy.__version__)")
    CUDA_DEVICES=$(python3 -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())" 2>/dev/null || echo "0")
    echo "✓ CuPy installed: $CUPY_VERSION (GPU devices: $CUDA_DEVICES)"
else
    echo "⚠ CuPy not installed (optional - for GPU acceleration)"
    echo "  Install with: pip3 install cupy-cuda11x  # or cupy-cuda12x"
fi

echo ""
echo "=========================================="
echo "Testing Filter System"
echo "=========================================="
echo ""

cd JetsonSky
python3 test_phase2.py

echo ""
echo "=========================================="
echo "Installation Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run examples: python3 JetsonSky/FILTER_USAGE_EXAMPLES.py"
echo "  2. Run tests: python3 JetsonSky/test_phase2.py"
echo "  3. Read docs: cat JetsonSky/filters/README.md"
echo ""
