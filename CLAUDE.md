# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JetsonSky is an astronomy live video capture and image processing application for ZWO cameras. It leverages NVIDIA GPU computing (CUDA/CuPy) for real-time image processing and supports both Linux (ARM64/x86_64) and Windows platforms.

**Current Version**: V53_07RC

**License**: Free for personal/non-commercial use. NOT free for commercial or professional use.

## Key Features

- Live video capture from ZWO ASI cameras (178, 183, 224, 290, 294, 385, 462, 482, 485, 533, 585, 662, 676, 678, 715, 1600 series)
- Real-time GPU-accelerated image processing
- AI-based crater and satellite detection using YOLOv8 models
- Support for 16-bit SER file format reading/writing (via Serfile library)
- Image stabilization and atmospheric turbulence reduction
- Multiple debayering patterns (RAW, RGGB, BGGR, GRBG, GBRG)
- HDR processing (Mertens and Mean methods)
- Post-processing of existing videos/images (no camera required)

## Architecture

### Main Components

1. **JetsonSky_Linux_Windows_V53_07RC.py** - Main application (10,000+ lines)
   - Tkinter-based GUI
   - Camera control and acquisition loops
   - Real-time filter pipeline orchestration
   - SER file video I/O
   - Video/image post-processing modes

2. **cuda_kernels.py** - CUDA/CuPy kernel definitions
   - Contains ~30+ custom CUDA kernels for image processing
   - Categories: FNR (Frame Noise Reduction), HDR, binning, debayering, dead pixel removal, contrast enhancement, denoising (NLM2, KNN, AANR), saturation, star amplification, histogram operations

3. **gui_widgets.py** - GUI widget definitions
   - Returns string templates for Tkinter widget creation
   - Organized into logical groups (top row, various widgets, filters, etc.)
   - Uses namespace injection pattern for dynamic GUI construction

4. **image_utils.py** - Image conversion utilities
   - NumPy ↔ CuPy conversions
   - RGB image ↔ separate RGB channels conversions
   - Gaussian blur, negative, image quality estimation

5. **astronomy_utils.py** - Astronomical calculations
   - `AstronomyCalculator` class for celestial mechanics
   - Angle conversions (degrees/minutes/seconds)
   - Julian day and sidereal time calculations
   - AltAz ↔ RA/Dec coordinate conversions
   - Telescope mount calibration using Polaris

### Support Libraries

- **Serfile/** - 16-bit SER file format library (modified by Alain Paillou from Jean-Baptiste Butet's original)
- **zwoasi_cupy/** - Python binding for ZWO SDK (camera control)
- **zwoefw/** - Python binding for ZWO EFW Mini filter wheel
- **synscan/** - Telescope mount control (optional)
- **Lib/** - Native ZWO libraries (.dll for Windows, .so for Linux ARM64)
- **AI_models/** - YOLOv8 models for crater and satellite detection

### Processing Pipeline Order

The filter pipeline processes frames in this fixed order:

1. RGB software adjustment
2. Image negative
3. Luminance estimate (mono from color)
4. 2-5 images SUM or MEAN stacking
5. Reduce consecutive variation
6. 3FNR1 front (3-frame noise removal)
7. AANR front (Adaptive Absorber Noise Removal) - High/Low dynamic
8. NR P1 (Noise Removal Paillou 1)
9. NR P2 (Noise Removal Paillou 2)
10. NLM2 (Non-Local Means 2)
11. KNN (K-Nearest Neighbors)
12. Luminance adjust
13. Image amplification (Linear/Gaussian)
14. Star amplification
15. Gradient/vignetting management
16. CLL (Contrast Low Light)
17. CLAHE contrast
18. Color saturation enhancement (2-pass option)
19. 3FNR2 back
20. AANR back (High dynamic only)
21. Sharpen 1
22. Sharpen 2

## Development Setup

### System Requirements

- **Linux**: ARM64 (Jetson) or x86_64 with NVIDIA GPU
- **Windows**: 64-bit Windows 10/11 with NVIDIA GPU
- **GPU**: NVIDIA GPU with CUDA support (required)
- **Python**: 3.10 recommended (NOT Python 3.11)

### Required Libraries

Core dependencies:
```
cupy (GPU acceleration)
opencv-python (with CUDA support preferred)
numpy
pillow
torch (for YOLOv8)
ultralytics (YOLOv8)
pynput (Linux keyboard input)
keyboard (Windows keyboard input)
psutil (Windows priority setting)
tkinter (GUI)
```

Optional:
```
GStreamer (hardware-accelerated video encoding)
```

### Platform-Specific Setup

**Linux**:
- Install ZWO SDK libraries from https://astronomy-imaging-camera.com/software-drivers
- Install udev rules: `sudo install Lib/asi.rules /lib/udev/rules.d` (or `/etc/udev/rules.d`)
- Set `keyboard_layout` in main script (line 99): "AZERTY" or "QWERTY"

**Windows**:
- Install ZWO ASI camera drivers
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- If using GStreamer, configure paths (lines 174-177 in main script)
- Set `keyboard_layout` in main script (line 99): "AZERTY" or "QWERTY"

### Running the Application

**With camera**:
```bash
python JetsonSky_Linux_Windows_V53_07RC.py
```

The software auto-detects:
- If camera connected → camera acquisition mode
- If no camera → video/image post-processing mode

### Important Configuration

**Main script settings** (JetsonSky_Linux_Windows_V53_07RC.py):
- Line 99-100: Keyboard layout ("AZERTY" or "QWERTY")
- Line 157-159: AI model paths (craters, satellites)
- Line 283: Main window font size (5-7, system-dependent)
- Line 288-297: Image/video directories and SDK library paths

## Development Patterns

### GPU Acceleration

- All heavy image processing uses CuPy (cp.ndarray) for GPU acceleration
- Minimize CPU↔GPU transfers: keep data on GPU between operations
- Use `cupy_context = cp.cuda.Stream(non_blocking=True)` for async operations
- Custom CUDA kernels in cuda_kernels.py are compiled at runtime via CuPy RawKernel

### Image Data Flow

1. Camera → NumPy array → CuPy array (GPU)
2. Process on GPU using CUDA kernels
3. CuPy array → NumPy array → PIL Image → Tkinter display
4. For video save: keep on GPU until batch write

### Widget Separation Pattern

GUI widgets use namespace injection:
```python
# In gui_widgets.py
def create_widgets():
    return '''
    # Tkinter widget code as string
    '''

# In main script
exec(gui_widgets.create_widgets())
```

This allows GUI definitions to be separated while maintaining access to main script's namespace.

### Recent Refactoring

Recent commits show modularization effort:
- CUDA kernels separated to `cuda_kernels.py` (raw CUDA C code as CuPy RawKernels)
- GUI widgets separated to `gui_widgets.py` (string templates with `exec()` for namespace injection)
- Image utilities separated to `image_utils.py` (pure utility functions)
- Astronomy calculations separated to `astronomy_utils.py` (OOP design with `AstronomyCalculator` class)

**Refactoring Pattern**:
1. Identify self-contained, reusable code blocks
2. Extract to separate module with clear dependencies
3. Add comprehensive docstrings
4. Update imports in main file
5. Update function calls to use new module
6. Remove old code from main file
7. Test thoroughly

When adding features, follow this pattern: separate reusable code into modules.

## Common Operations

### Testing Changes

No automated test suite exists. Test manually:
1. Launch with test camera or use existing SER video
2. Enable various filters via GUI checkboxes
3. Verify FPS and visual output
4. Check for CUDA errors in terminal

### Adding New Filters

1. Define CUDA kernel in `cuda_kernels.py`
2. Add kernel invocation in main processing loop
3. Create GUI widgets in `gui_widgets.py`
4. Add callbacks in main script
5. Insert in correct position in filter pipeline (see order above)

### Debugging GPU Operations

- CuPy errors often indicate memory issues or kernel launch failures
- Use `cp.cuda.Stream.synchronize()` to force GPU completion for debugging
- Check kernel thread dimensions: `nb_ThreadsX=32, nb_ThreadsY=32`
- Grid dimensions calculated from image resolution

### Working with Serfile

For 16-bit SER files:
- Import: `import Serfile as Serfile`
- Modified version specific to JetsonSky (line 5 mentions update)
- Handles RAW 16-bit mono/color video formats

## Keyboard Controls

Vary by layout (AZERTY vs QWERTY), but generally:
- **ZSQD/WSAD**: Image stabilization positioning
- **Arrows**: Zoom navigation
- **Shift+R then TGHFV**: Various functions
- **Shift+B then OLMK;/OL:K,**: Additional functions

## Known Issues

- Jetson SBC may have issues with certain versions (V53_03RC had problems, V51_05RC recommended for Jetson)
- Python threading is minimal by design (V52+ reduced threads for stability)
- Linux version may have more bugs than Windows version
- GStreamer paths must be correctly set on Windows or errors occur

## Camera Support

Supported ZWO cameras (as of V53_07RC):
ASI120, 178, 183, 224, 290, 294, 385, 462, 482, 485, 533, 585, 662, 676, 678, 715, 1600 series

Both mono (MM) and color (MC) variants, including Pro versions.

## Video Format

- Output: Uncompressed AVI or GStreamer-encoded MP4 (if available)
- Input: SER format (8-bit and 16-bit), AVI
- 16-bit support for HDR and small signal enhancement
