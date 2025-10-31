#!/usr/bin/env python3
"""
Test script for Phase 1 refactoring modules.

This script verifies that the new core and utils modules
work correctly independently.
"""

import sys
import os

# Add JetsonSky directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("="*70)
    print("TESTING IMPORTS")
    print("="*70)

    try:
        from core import (
            CameraConfig, ProcessingState, AppState,
            MountState, CaptureState, QualityMetrics,
            get_camera_config, get_supported_cameras, is_camera_supported
        )
        print("✓ Core modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core modules: {e}")
        return False

    try:
        from utils.constants import (
            DEFAULT_EXPOSITION, DEFAULT_GAIN, COLOR_TURQUOISE,
            MAX_16BIT_VALUE, NB_THREADS_X
        )
        print("✓ Utils modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils modules: {e}")
        return False

    print()
    return True


def test_camera_registry():
    """Test camera model registry functionality."""
    print("="*70)
    print("TESTING CAMERA REGISTRY")
    print("="*70)

    from core import get_camera_config, get_supported_cameras, is_camera_supported

    # Test getting supported cameras
    cameras = get_supported_cameras()
    print(f"✓ Found {len(cameras)} supported camera models")
    print(f"  Sample cameras: {cameras[:5]}")

    # Test checking camera support
    assert is_camera_supported("ZWO ASI178MC"), "ASI178MC should be supported"
    print("✓ is_camera_supported() works correctly")

    assert not is_camera_supported("Fake Camera"), "Fake camera should not be supported"
    print("✓ Correctly rejects unsupported cameras")

    # Test getting camera config
    config = get_camera_config("ZWO ASI178MC")
    print(f"\n✓ Retrieved config for {config.model}")
    print(f"  Resolution: {config.resolution_x}x{config.resolution_y}")
    print(f"  Sensor factor: {config.sensor_factor}")
    print(f"  Bit depth: {config.sensor_bits}")
    print(f"  Bayer pattern: {config.bayer_pattern}")
    print(f"  Max gain: {config.max_gain}")
    print(f"  BIN1 resolutions: {len(config.supported_resolutions_bin1)}")
    print(f"  BIN2 resolutions: {len(config.supported_resolutions_bin2)}")

    # Verify resolution correctness
    assert config.resolution_x == 3096, "ASI178MC should be 3096 pixels wide"
    assert config.resolution_y == 2080, "ASI178MC should be 2080 pixels high"
    print("✓ Camera config values are correct")

    # Test error handling
    try:
        get_camera_config("Invalid Camera")
        print("✗ Should have raised ValueError for invalid camera")
        return False
    except ValueError as e:
        print(f"✓ Correctly raises ValueError for invalid camera")

    print()
    return True


def test_config_classes():
    """Test configuration dataclasses."""
    print("="*70)
    print("TESTING CONFIGURATION CLASSES")
    print("="*70)

    from core import ProcessingState, AppState, get_camera_config

    # Test ProcessingState defaults
    processing = ProcessingState()
    assert processing.exposition == 1000, "Default exposition should be 1000"
    assert processing.gain == 100, "Default gain should be 100"
    assert processing.flip_vertical == False, "Default flip_vertical should be False"
    print("✓ ProcessingState defaults are correct")

    # Test modifying ProcessingState
    processing.exposition = 2000
    processing.gain = 150
    processing.flip_vertical = True
    assert processing.exposition == 2000
    assert processing.gain == 150
    assert processing.flip_vertical == True
    print("✓ ProcessingState modifications work")

    # Test AppState
    app_state = AppState()
    assert app_state.acquisition_running == False
    assert app_state.camera_connected == False
    print("✓ AppState defaults are correct")

    # Test AppState with camera config
    camera_config = get_camera_config("ZWO ASI178MC")
    app_state.camera_config = camera_config
    resolution = app_state.get_current_resolution()
    print(f"✓ Current resolution (mode {app_state.resolution_mode}): {resolution[0]}x{resolution[1]}")
    # Resolution mode 1 should be second entry in list
    expected = camera_config.supported_resolutions_bin1[app_state.resolution_mode]
    assert resolution == expected, f"Resolution should be {expected} for mode {app_state.resolution_mode}"

    # Test resolution for different modes
    app_state.resolution_mode = 2
    resolution = app_state.get_current_resolution()
    print(f"✓ Resolution mode 2: {resolution[0]}x{resolution[1]}")

    # Test BIN2 mode
    app_state.binning_mode = 2
    app_state.resolution_mode = 1
    resolution = app_state.get_current_resolution()
    print(f"✓ BIN2 resolution: {resolution[0]}x{resolution[1]}")

    print()
    return True


def test_constants():
    """Test constants module."""
    print("="*70)
    print("TESTING CONSTANTS")
    print("="*70)

    from utils.constants import (
        DEFAULT_EXPOSITION,
        DEFAULT_GAIN,
        MAX_16BIT_VALUE,
        NB_THREADS_X,
        NB_THREADS_Y,
        COLOR_TURQUOISE,
        get_usb_bandwidth_for_platform,
        get_max_value_for_bits,
        PLATFORM_WINDOWS,
        PLATFORM_LINUX
    )

    # Test basic constants
    assert DEFAULT_EXPOSITION == 1000
    assert DEFAULT_GAIN == 100
    assert MAX_16BIT_VALUE == 65535
    assert NB_THREADS_X == 32
    assert NB_THREADS_Y == 32
    print("✓ Basic constants have correct values")

    # Test color constants
    assert COLOR_TURQUOISE == "#40E0D0"
    print("✓ Color constants are correct")

    # Test helper functions
    usb_win = get_usb_bandwidth_for_platform(PLATFORM_WINDOWS)
    usb_linux = get_usb_bandwidth_for_platform(PLATFORM_LINUX)
    assert usb_win == 95
    assert usb_linux == 70
    print(f"✓ USB bandwidth: Windows={usb_win}, Linux={usb_linux}")

    # Test bit depth helper
    max_12bit = get_max_value_for_bits(12)
    max_14bit = get_max_value_for_bits(14)
    max_16bit = get_max_value_for_bits(16)
    assert max_12bit == 4095
    assert max_14bit == 16383
    assert max_16bit == 65535
    print(f"✓ Max values: 12-bit={max_12bit}, 14-bit={max_14bit}, 16-bit={max_16bit}")

    print()
    return True


def test_camera_model_coverage():
    """Test that all cameras from original code are covered."""
    print("="*70)
    print("TESTING CAMERA MODEL COVERAGE")
    print("="*70)

    from core import get_supported_cameras

    cameras = get_supported_cameras()

    # Expected cameras from original code
    expected_cameras = [
        "ZWO ASI120MC", "ZWO ASI120MM",
        "ZWO ASI178MC", "ZWO ASI178MM", "ZWO ASI178MM Pro",
        "ZWO ASI183MC", "ZWO ASI183MM", "ZWO ASI183MC Pro", "ZWO ASI183MM Pro",
        "ZWO ASI224MC",
        "ZWO ASI290MC", "ZWO ASI290MM", "ZWO ASI290MM Mini",
        "ZWO ASI294MC", "ZWO ASI294MM", "ZWO ASI294MC Pro", "ZWO ASI294MM Pro",
        "ZWO ASI385MC",
        "ZWO ASI462MC",
        "ZWO ASI482MC",
        "ZWO ASI485MC", "ZWO ASI585MC", "ZWO ASI585MM",
        "ZWO ASI533MC", "ZWO ASI533MM", "ZWO ASI533MC Pro", "ZWO ASI533MM Pro",
        "ZWO ASI662MC",
        "ZWO ASI676MC",
        "ZWO ASI678MC", "ZWO ASI678MM",
        "ZWO ASI715MC",
        "ZWO ASI1600MC", "ZWO ASI1600MM",
    ]

    missing = []
    for camera in expected_cameras:
        if camera not in cameras:
            missing.append(camera)

    if missing:
        print(f"✗ Missing cameras: {missing}")
        return False
    else:
        print(f"✓ All {len(expected_cameras)} expected cameras are supported")

    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" PHASE 1 MODULE TESTING")
    print("="*70)
    print()

    tests = [
        ("Imports", test_imports),
        ("Camera Registry", test_camera_registry),
        ("Config Classes", test_config_classes),
        ("Constants", test_constants),
        ("Camera Coverage", test_camera_model_coverage),
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

    print("="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed}/{len(tests)} tests FAILED")
    else:
        print("         ALL TESTS PASSED! ✓")
    print("="*70)
    print()

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
