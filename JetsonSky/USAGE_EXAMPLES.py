"""
Usage Examples for Phase 1 Refactoring

This file demonstrates how to use the new core and utils modules
created in Phase 1 of the refactoring. These examples show the
transformation from the old monolithic approach to the new modular design.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# EXAMPLE 1: Camera Configuration (OLD vs NEW)
# =============================================================================

def example_1_camera_config():
    """
    Example 1: Getting camera configuration

    OLD WAY: 1,500-line if-elif chain in init_camera()
    NEW WAY: Simple dictionary lookup
    """
    print("="*70)
    print("EXAMPLE 1: Camera Configuration")
    print("="*70)

    from core import get_camera_config, get_supported_cameras

    # Get list of all supported cameras
    cameras = get_supported_cameras()
    print(f"\n📷 {len(cameras)} cameras supported")
    print(f"   First 5: {cameras[:5]}")

    # Get configuration for a specific camera
    camera_model = "ZWO ASI178MC"
    config = get_camera_config(camera_model)

    print(f"\n✓ Camera: {config.model}")
    print(f"  • Resolution: {config.resolution_x} x {config.resolution_y}")
    print(f"  • Sensor: {config.sensor_factor} ({config.sensor_bits}-bit)")
    print(f"  • Bayer: {config.bayer_pattern}")
    print(f"  • Max Gain: {config.max_gain}")
    print(f"  • Display: {config.display_x} x {config.display_y}")

    # Access resolution arrays
    print(f"\n  BIN1 Resolutions ({len(config.supported_resolutions_bin1)}):")
    for i, (w, h) in enumerate(config.supported_resolutions_bin1[:3]):
        print(f"    [{i}] {w} x {h}")
    print(f"    ...")

    print(f"\n  BIN2 Resolutions ({len(config.supported_resolutions_bin2)}):")
    for i, (w, h) in enumerate(config.supported_resolutions_bin2[:3]):
        print(f"    [{i}] {w} x {h}")
    print(f"    ...")

    print("\n💡 OLD WAY:")
    print("   if cameras_found[0] == 'ZWO ASI178MC':")
    print("       res_cam_x = 3096")
    print("       res_cam_y = 2080")
    print("       # ... 50 more lines of hardcoded values")
    print("   elif cameras_found[0] == 'ZWO ASI294MC':")
    print("       # ... another 50 lines")
    print("   # ... repeat 20+ times = 1,500 lines!")

    print("\n✨ NEW WAY:")
    print("   config = get_camera_config('ZWO ASI178MC')")
    print("   # All values in one clean object!")
    print()


# =============================================================================
# EXAMPLE 2: Application State Management (OLD vs NEW)
# =============================================================================

def example_2_application_state():
    """
    Example 2: Managing application state

    OLD WAY: 300+ scattered global variables
    NEW WAY: Organized AppState dataclass
    """
    print("="*70)
    print("EXAMPLE 2: Application State Management")
    print("="*70)

    from core import AppState, get_camera_config

    # Create application state
    app = AppState()

    print("\n✓ Created AppState")
    print(f"  • Acquisition running: {app.acquisition_running}")
    print(f"  • Camera connected: {app.camera_connected}")
    print(f"  • Resolution mode: {app.resolution_mode}")
    print(f"  • Binning mode: {app.binning_mode}")

    # Configure camera
    app.camera_config = get_camera_config("ZWO ASI178MC")
    app.camera_connected = True

    print(f"\n✓ Camera configured: {app.camera_config.model}")

    # Get current resolution based on mode
    resolution = app.get_current_resolution()
    print(f"  • Current resolution: {resolution[0]} x {resolution[1]}")

    # Change resolution mode
    app.resolution_mode = 0  # Highest resolution
    resolution = app.get_current_resolution()
    print(f"  • Mode 0 (highest): {resolution[0]} x {resolution[1]}")

    app.resolution_mode = 2  # Medium resolution
    resolution = app.get_current_resolution()
    print(f"  • Mode 2 (medium): {resolution[0]} x {resolution[1]}")

    # Configure processing parameters
    app.processing.exposition = 2000
    app.processing.gain = 150
    app.processing.flip_vertical = True
    app.processing.denoise_knn = 0.3
    app.processing.sharpen_amount = 1.5

    print("\n✓ Processing configured:")
    print(f"  • Exposition: {app.processing.exposition} µs")
    print(f"  • Gain: {app.processing.gain}")
    print(f"  • Vertical flip: {app.processing.flip_vertical}")
    print(f"  • Denoise: {app.processing.denoise_knn}")
    print(f"  • Sharpen: {app.processing.sharpen_amount}")

    # Enable filters
    app.processing.filter_enabled_sharpen1 = True
    app.processing.filter_enabled_clahe = True
    app.processing.filter_enabled_sat = True

    enabled_filters = []
    if app.processing.filter_enabled_sharpen1:
        enabled_filters.append("Sharpen")
    if app.processing.filter_enabled_clahe:
        enabled_filters.append("CLAHE Contrast")
    if app.processing.filter_enabled_sat:
        enabled_filters.append("Saturation")

    print(f"\n✓ Enabled filters: {', '.join(enabled_filters)}")

    print("\n💡 OLD WAY (300+ global variables):")
    print("   val_exposition = 2000")
    print("   val_gain = 150")
    print("   FlipV = 1")
    print("   val_denoise_KNN = 0.3")
    print("   val_sharpen = 1.5")
    print("   # ... 295 more scattered globals!")

    print("\n✨ NEW WAY (organized state):")
    print("   app = AppState()")
    print("   app.processing.exposition = 2000")
    print("   app.processing.gain = 150")
    print("   # All state in one place, organized by category!")
    print()


# =============================================================================
# EXAMPLE 3: Using Constants (OLD vs NEW)
# =============================================================================

def example_3_constants():
    """
    Example 3: Using centralized constants

    OLD WAY: Magic numbers scattered everywhere
    NEW WAY: Named constants from utils.constants
    """
    print("="*70)
    print("EXAMPLE 3: Using Constants")
    print("="*70)

    from utils.constants import (
        DEFAULT_EXPOSITION,
        DEFAULT_GAIN,
        MAX_16BIT_VALUE,
        COLOR_TURQUOISE,
        COLOR_BLUE,
        NB_THREADS_X,
        NB_THREADS_Y,
        BAYER_RGGB,
        HDR_METHOD_MERTENS,
        STACKING_MODE_MEAN,
        get_usb_bandwidth_for_platform,
        get_max_value_for_bits,
        PLATFORM_WINDOWS,
        PLATFORM_LINUX,
    )

    # Camera defaults
    print("\n📸 Camera Defaults:")
    print(f"  • Default exposition: {DEFAULT_EXPOSITION} µs")
    print(f"  • Default gain: {DEFAULT_GAIN}")
    print(f"  • Max 16-bit value: {MAX_16BIT_VALUE}")

    # GUI colors
    print("\n🎨 GUI Colors:")
    print(f"  • Turquoise: {COLOR_TURQUOISE}")
    print(f"  • Blue: {COLOR_BLUE}")

    # CUDA settings
    print("\n⚡ CUDA Settings:")
    print(f"  • Thread block X: {NB_THREADS_X}")
    print(f"  • Thread block Y: {NB_THREADS_Y}")

    # Platform-specific settings
    usb_win = get_usb_bandwidth_for_platform(PLATFORM_WINDOWS)
    usb_linux = get_usb_bandwidth_for_platform(PLATFORM_LINUX)

    print("\n💻 Platform Settings:")
    print(f"  • USB bandwidth (Windows): {usb_win}")
    print(f"  • USB bandwidth (Linux): {usb_linux}")

    # Bit depth helpers
    print("\n🔢 Bit Depth Values:")
    print(f"  • 12-bit max: {get_max_value_for_bits(12)}")
    print(f"  • 14-bit max: {get_max_value_for_bits(14)}")
    print(f"  • 16-bit max: {get_max_value_for_bits(16)}")

    # Processing constants
    print("\n🎛️ Processing Constants:")
    print(f"  • Bayer RGGB: {BAYER_RGGB}")
    print(f"  • HDR method: {HDR_METHOD_MERTENS}")
    print(f"  • Stacking mode: {STACKING_MODE_MEAN}")

    print("\n💡 OLD WAY:")
    print("   val_exposition = 1000  # Magic number!")
    print("   threshold_16bits = 65535  # What is this?")
    print("   USBCam = 95  # Windows? Linux? Who knows!")

    print("\n✨ NEW WAY:")
    print("   from utils.constants import DEFAULT_EXPOSITION, MAX_16BIT_VALUE")
    print("   exposition = DEFAULT_EXPOSITION  # Self-documenting!")
    print("   usb = get_usb_bandwidth_for_platform(platform)")
    print()


# =============================================================================
# EXAMPLE 4: Practical Camera Initialization
# =============================================================================

def example_4_practical_initialization():
    """
    Example 4: Practical camera initialization workflow

    Shows a complete initialization sequence using the new modules.
    """
    print("="*70)
    print("EXAMPLE 4: Practical Camera Initialization")
    print("="*70)

    from core import AppState, get_camera_config, is_camera_supported
    from utils.constants import (
        DEFAULT_EXPOSITION,
        DEFAULT_GAIN,
        DEFAULT_USB_BANDWIDTH_LINUX,
        get_usb_bandwidth_for_platform,
        PLATFORM_LINUX,
    )

    # Step 1: Check if camera is supported
    camera_model = "ZWO ASI178MC"

    print(f"\n1️⃣ Checking camera support...")
    if not is_camera_supported(camera_model):
        print(f"   ✗ {camera_model} is not supported!")
        return
    print(f"   ✓ {camera_model} is supported")

    # Step 2: Create application state
    print("\n2️⃣ Creating application state...")
    app = AppState()
    print("   ✓ AppState created")

    # Step 3: Load camera configuration
    print(f"\n3️⃣ Loading camera configuration...")
    app.camera_config = get_camera_config(camera_model)
    print(f"   ✓ Loaded config for {app.camera_config.model}")
    print(f"     • Sensor: {app.camera_config.resolution_x}x{app.camera_config.resolution_y}")
    print(f"     • Bit depth: {app.camera_config.sensor_bits}-bit")

    # Step 4: Initialize camera settings with defaults
    print("\n4️⃣ Initializing camera settings...")
    app.processing.exposition = DEFAULT_EXPOSITION
    app.processing.gain = DEFAULT_GAIN
    app.processing.usb_bandwidth = get_usb_bandwidth_for_platform(PLATFORM_LINUX)
    print(f"   ✓ Exposition: {app.processing.exposition} µs")
    print(f"   ✓ Gain: {app.processing.gain}")
    print(f"   ✓ USB bandwidth: {app.processing.usb_bandwidth}")

    # Step 5: Set resolution mode
    print("\n5️⃣ Setting resolution mode...")
    app.resolution_mode = 1  # Medium-high resolution
    app.binning_mode = 1     # BIN1 mode
    resolution = app.get_current_resolution()
    print(f"   ✓ Resolution: {resolution[0]}x{resolution[1]}")
    print(f"   ✓ Binning: BIN{app.binning_mode}")

    # Step 6: Configure initial filters
    print("\n6️⃣ Configuring filters...")
    app.processing.filter_enabled_sharpen1 = True
    app.processing.sharpen_amount = 1.2
    app.processing.filter_enabled_denoise_paillou = True
    app.processing.denoise_strength = 0.3
    print("   ✓ Sharpen filter enabled (amount: 1.2)")
    print("   ✓ Denoise filter enabled (strength: 0.3)")

    # Step 7: Ready to start acquisition
    print("\n7️⃣ Ready for acquisition!")
    app.acquisition_running = True
    app.camera_connected = True
    print("   ✓ Camera initialized and ready")

    print("\n📊 Final Configuration Summary:")
    print(f"   Camera: {app.camera_config.model}")
    print(f"   Resolution: {resolution[0]}x{resolution[1]} (mode {app.resolution_mode}, BIN{app.binning_mode})")
    print(f"   Exposition: {app.processing.exposition} µs")
    print(f"   Gain: {app.processing.gain}")
    print(f"   Filters: Sharpen + Denoise")
    print()


# =============================================================================
# EXAMPLE 5: Working with Multiple Cameras
# =============================================================================

def example_5_multiple_cameras():
    """
    Example 5: Comparing different camera configurations

    Shows how easy it is to work with multiple cameras.
    """
    print("="*70)
    print("EXAMPLE 5: Multiple Cameras Comparison")
    print("="*70)

    from core import get_camera_config

    cameras_to_compare = [
        "ZWO ASI120MC",   # Entry-level
        "ZWO ASI178MC",   # Mid-range
        "ZWO ASI294MC",   # High-end
        "ZWO ASI1600MC",  # Professional
    ]

    print("\n📷 Camera Comparison:\n")
    print(f"{'Model':<20} {'Resolution':<15} {'Sensor':<10} {'Bits':<5} {'Max Gain':<8}")
    print("-" * 70)

    for camera_model in cameras_to_compare:
        config = get_camera_config(camera_model)
        resolution = f"{config.resolution_x}x{config.resolution_y}"

        print(f"{config.model:<20} {resolution:<15} {config.sensor_factor:<10} "
              f"{config.sensor_bits:<5} {config.max_gain:<8}")

    print("\n💡 Adding a new camera is just adding one dictionary entry!")
    print("   No need to modify 1,500 lines of if-elif code!")
    print()


# =============================================================================
# EXAMPLE 6: State Persistence (Future Use)
# =============================================================================

def example_6_state_persistence():
    """
    Example 6: How dataclasses enable easy state persistence

    Shows how the new structure makes saving/loading state trivial.
    """
    print("="*70)
    print("EXAMPLE 6: State Persistence (Future)")
    print("="*70)

    from core import AppState, get_camera_config
    import json
    from dataclasses import asdict

    # Create and configure state
    app = AppState()
    app.camera_config = get_camera_config("ZWO ASI178MC")
    app.processing.exposition = 2500
    app.processing.gain = 200
    app.processing.flip_vertical = True
    app.resolution_mode = 2

    print("\n💾 Saving state to JSON...")

    # Convert to dictionary (dataclasses make this easy!)
    state_dict = {
        'camera_model': app.camera_config.model if app.camera_config else None,
        'resolution_mode': app.resolution_mode,
        'binning_mode': app.binning_mode,
        'processing': {
            'exposition': app.processing.exposition,
            'gain': app.processing.gain,
            'flip_vertical': app.processing.flip_vertical,
            'denoise_knn': app.processing.denoise_knn,
            'sharpen_amount': app.processing.sharpen_amount,
        }
    }

    json_str = json.dumps(state_dict, indent=2)
    print(f"\n{json_str}")

    print("\n✨ Benefits of dataclasses:")
    print("   • Easy to serialize (JSON, pickle, etc.)")
    print("   • Can save/load user presets")
    print("   • Can undo/redo state changes")
    print("   • Can share configurations between users")

    print("\n💡 With 300+ global variables, this was IMPOSSIBLE!")
    print("   Now it's just a few lines of code!")
    print()


# =============================================================================
# EXAMPLE 7: Type Safety and IDE Support
# =============================================================================

def example_7_type_safety():
    """
    Example 7: Type safety and IDE autocomplete

    Shows how type hints improve developer experience.
    """
    print("="*70)
    print("EXAMPLE 7: Type Safety & IDE Support")
    print("="*70)

    from core import AppState, ProcessingState

    print("\n✨ With type hints, your IDE can:")
    print("   • Autocomplete field names")
    print("   • Show documentation on hover")
    print("   • Catch type errors before running")
    print("   • Refactor safely")

    app = AppState()

    print("\n📝 Example - IDE knows all fields:")
    print("   app.processing.  <-- IDE shows all 60+ processing options!")
    print("   app.camera_config.  <-- IDE shows all camera config fields!")

    print("\n✓ Type checking catches errors:")
    print("   app.processing.exposition = '1000'  # ✗ Wrong type!")
    print("   # IDE/mypy: Expected int, got str")

    print("\n💡 OLD WAY (global variables):")
    print("   val_expositon = 1000  # Typo! Good luck finding it!")
    print("   # No autocomplete, no type checking, no help!")

    print("\n✨ NEW WAY (dataclasses):")
    print("   app.processing.exposition = 1000  # ✓ IDE autocompletes!")
    print("   app.processing.expostion = 1000   # ✗ IDE shows error!")
    print()


# =============================================================================
# MAIN: Run all examples
# =============================================================================

def main():
    """Run all usage examples."""
    print("\n" + "="*70)
    print(" JETSONSKY PHASE 1 - USAGE EXAMPLES")
    print("="*70)
    print("\n This file demonstrates the new modular architecture")
    print(" created in Phase 1 of the refactoring.\n")

    examples = [
        ("Camera Configuration", example_1_camera_config),
        ("Application State", example_2_application_state),
        ("Constants", example_3_constants),
        ("Practical Initialization", example_4_practical_initialization),
        ("Multiple Cameras", example_5_multiple_cameras),
        ("State Persistence", example_6_state_persistence),
        ("Type Safety", example_7_type_safety),
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
            if i < len(examples):
                input(f"\n⏎ Press Enter for Example {i+1}: {examples[i][0]}...")
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("="*70)
    print(" SUMMARY")
    print("="*70)
    print("\n✅ Phase 1 provides:")
    print("   • Clean camera configuration (99% code reduction!)")
    print("   • Organized state management (0 global variables!)")
    print("   • Centralized constants (easy maintenance)")
    print("   • Type safety and IDE support")
    print("   • Easy testing and persistence")

    print("\n🚀 Ready for Phase 2:")
    print("   • Extract filters into modular classes")
    print("   • Create FilterPipeline for processing")
    print("   • Reduce monolithic file by 2,000+ more lines")

    print("\n" + "="*70)
    print()


if __name__ == "__main__":
    main()
