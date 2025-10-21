#!/usr/bin/env python3
"""
Test script for demo applications.

This script tests the demo components to ensure they work correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_camera_simulator():
    """Test the camera simulator."""
    print("="*70)
    print("TEST: Camera Simulator")
    print("="*70)

    try:
        import numpy as np
        print("✓ NumPy available")
    except ImportError:
        print("✗ NumPy not available - skipping camera simulator test")
        print("  Install NumPy to run this test: pip install numpy")
        return True  # Skip test but don't fail

    from demos.camera_simulator import create_simulated_camera

    # Create camera
    print("\n1. Creating simulated camera...")
    camera = create_simulated_camera("ZWO ASI178MC", (640, 480), 14)

    # Configure
    print("\n2. Configuring camera...")
    camera.set_exposition(1000)
    camera.set_gain(100)

    # Start capture
    print("\n3. Starting capture...")
    camera.start_capture()

    # Capture frames
    print("\n4. Capturing 5 frames...")
    for i in range(5):
        frame = camera.capture_frame()
        if frame is not None:
            print(f"   Frame {i+1}: Shape={frame.shape}, Mean={frame.mean():.1f}, Max={frame.max()}")
        else:
            print(f"   Frame {i+1}: Failed to capture")
            return False

    # Stop capture
    print("\n5. Stopping capture...")
    camera.stop_capture()

    # Get stats
    print("\n6. Getting stats...")
    stats = camera.get_stats()
    print(f"   Model: {stats['model']}")
    print(f"   Resolution: {stats['resolution']}")
    print(f"   Frames captured: {stats['frames_captured']}")

    print("\n✓ Camera simulator test PASSED")
    return True


def test_cli_demo_components():
    """Test CLI demo components without interaction."""
    print("\n" + "="*70)
    print("TEST: CLI Demo Components")
    print("="*70)

    from core import AppState, get_camera_config
    from utils.constants import DEFAULT_EXPOSITION, DEFAULT_GAIN

    # Check for NumPy
    try:
        import numpy as np
        has_numpy = True
        print("✓ NumPy available")
    except ImportError:
        has_numpy = False
        print("⚠ NumPy not available - testing without camera simulator")

    if has_numpy:
        from demos.camera_simulator import create_simulated_camera

    # Test 1: Create app state
    print("\n1. Creating application state...")
    app = AppState()
    print("   ✓ AppState created")

    # Test 2: Load camera
    print("\n2. Loading camera configuration...")
    app.camera_config = get_camera_config("ZWO ASI178MC")
    print(f"   ✓ Loaded: {app.camera_config.model}")

    # Test 3: Configure processing
    print("\n3. Configuring processing...")
    app.processing.exposition = DEFAULT_EXPOSITION
    app.processing.gain = DEFAULT_GAIN
    app.processing.flip_vertical = True
    print(f"   ✓ Exposition: {app.processing.exposition} µs")
    print(f"   ✓ Gain: {app.processing.gain}")
    print(f"   ✓ Flip vertical: {app.processing.flip_vertical}")

    # Test 4: Enable filters
    print("\n4. Enabling filters...")
    app.processing.filter_enabled_sharpen1 = True
    app.processing.filter_enabled_denoise_paillou = True
    print("   ✓ Sharpen enabled")
    print("   ✓ Denoise enabled")

    # Test 5: Get resolution
    print("\n5. Getting current resolution...")
    app.resolution_mode = 1
    app.binning_mode = 1
    resolution = app.get_current_resolution()
    print(f"   ✓ Resolution: {resolution[0]}x{resolution[1]}")

    # Test 6-7: Simulated camera (only if NumPy available)
    if has_numpy:
        print("\n6. Creating simulated camera...")
        camera = create_simulated_camera(
            app.camera_config.model,
            resolution,
            app.camera_config.sensor_bits
        )
        print("   ✓ Camera created")

        print("\n7. Testing simulated acquisition...")
        camera.set_exposition(app.processing.exposition)
        camera.set_gain(app.processing.gain)
        camera.start_capture()

        frame = camera.capture_frame()
        if frame is not None:
            print(f"   ✓ Frame captured: {frame.shape}")
        else:
            print("   ✗ Frame capture failed")
            return False

        camera.stop_capture()
    else:
        print("\n6-7. Skipping camera simulator tests (NumPy not available)")
        print("   ⚠ Install NumPy to test camera simulator: pip install numpy")

    # Test 8: Save configuration
    print("\n8. Testing configuration save/load...")
    import json

    config = {
        'camera_model': app.camera_config.model,
        'exposition': app.processing.exposition,
        'gain': app.processing.gain,
    }

    # Save
    with open('test_config.json', 'w') as f:
        json.dump(config, f)
    print("   ✓ Configuration saved")

    # Load
    with open('test_config.json', 'r') as f:
        loaded = json.load(f)

    assert loaded['camera_model'] == config['camera_model']
    assert loaded['exposition'] == config['exposition']
    print("   ✓ Configuration loaded and verified")

    # Cleanup
    os.remove('test_config.json')

    print("\n✓ CLI demo components test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" DEMO APPLICATIONS TEST SUITE")
    print("="*70)

    tests = [
        ("Camera Simulator", test_camera_simulator),
        ("CLI Demo Components", test_cli_demo_components),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
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
