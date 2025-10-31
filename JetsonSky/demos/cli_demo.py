#!/usr/bin/env python3
"""
JetsonSky CLI Demo Application

Interactive command-line demonstration of the Phase 1 refactoring.
Shows how to use the new core and utils modules with a simulated camera.

Usage:
    python3 cli_demo.py
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    AppState,
    get_camera_config,
    get_supported_cameras,
    is_camera_supported,
)
from utils.constants import (
    DEFAULT_EXPOSITION,
    DEFAULT_GAIN,
    get_usb_bandwidth_for_platform,
    PLATFORM_LINUX,
    COLOR_TURQUOISE,
    COLOR_BLUE,
)
from demos.camera_simulator import create_simulated_camera


class JetsonSkyDemo:
    """Interactive CLI demo application."""

    def __init__(self):
        """Initialize demo application."""
        self.app = AppState()
        self.camera = None
        self.running = False

    def print_header(self):
        """Print application header."""
        print("\n" + "="*70)
        print(" JETSONSKY PHASE 1 - INTERACTIVE DEMO")
        print("="*70)
        print("\n This demo showcases the new modular architecture.")
        print(" All state is managed with dataclasses, no global variables!\n")

    def print_menu(self):
        """Print main menu."""
        print("\n" + "-"*70)
        print("MENU:")
        print("-"*70)
        print("  1. List supported cameras")
        print("  2. Select and configure camera")
        print("  3. Configure processing settings")
        print("  4. Enable/disable filters")
        print("  5. Start simulated acquisition")
        print("  6. View current configuration")
        print("  7. Save configuration to JSON")
        print("  8. Load configuration from JSON")
        print("  9. Exit")
        print("-"*70)

    def list_cameras(self):
        """List all supported cameras."""
        print("\n" + "="*70)
        print("SUPPORTED CAMERAS")
        print("="*70)

        cameras = get_supported_cameras()
        print(f"\n📷 {len(cameras)} cameras supported:\n")

        for i, model in enumerate(cameras, 1):
            config = get_camera_config(model)
            res = f"{config.resolution_x}x{config.resolution_y}"
            print(f"  {i:2d}. {model:<25} {res:<12} {config.sensor_bits}-bit  {config.sensor_factor}")

        print()

    def select_camera(self):
        """Select and configure a camera."""
        print("\n" + "="*70)
        print("CAMERA SELECTION")
        print("="*70)

        # Get camera model from user
        cameras = get_supported_cameras()
        print(f"\n Available cameras: {len(cameras)}")

        for i, model in enumerate(cameras[:10], 1):
            print(f"  {i}. {model}")

        if len(cameras) > 10:
            print(f"  ... and {len(cameras) - 10} more")

        camera_model = input("\n Enter camera model (or number 1-10): ").strip()

        # Handle numeric input
        if camera_model.isdigit():
            idx = int(camera_model) - 1
            if 0 <= idx < min(10, len(cameras)):
                camera_model = cameras[idx]
            else:
                print(f"✗ Invalid camera number")
                return

        # Check if supported
        if not is_camera_supported(camera_model):
            print(f"✗ {camera_model} is not supported")
            return

        # Load configuration
        print(f"\n✓ Loading configuration for {camera_model}...")
        self.app.camera_config = get_camera_config(camera_model)

        print(f"\n✓ Camera configured: {self.app.camera_config.model}")
        print(f"  • Resolution: {self.app.camera_config.resolution_x}x{self.app.camera_config.resolution_y}")
        print(f"  • Sensor: {self.app.camera_config.sensor_factor} ({self.app.camera_config.sensor_bits}-bit)")
        print(f"  • Bayer: {self.app.camera_config.bayer_pattern}")

        # Set default settings
        self.app.processing.exposition = DEFAULT_EXPOSITION
        self.app.processing.gain = DEFAULT_GAIN
        self.app.processing.usb_bandwidth = get_usb_bandwidth_for_platform(PLATFORM_LINUX)

        print(f"\n✓ Default settings applied:")
        print(f"  • Exposition: {self.app.processing.exposition} µs")
        print(f"  • Gain: {self.app.processing.gain}")
        print(f"  • USB bandwidth: {self.app.processing.usb_bandwidth}")

        # Set resolution mode
        self.app.resolution_mode = 1
        self.app.binning_mode = 1
        resolution = self.app.get_current_resolution()
        print(f"  • Resolution: {resolution[0]}x{resolution[1]} (mode {self.app.resolution_mode}, BIN{self.app.binning_mode})")

        self.app.camera_connected = True

    def configure_processing(self):
        """Configure processing settings."""
        if self.app.camera_config is None:
            print("\n✗ Please select a camera first (option 2)")
            return

        print("\n" + "="*70)
        print("PROCESSING CONFIGURATION")
        print("="*70)

        # Exposition
        exposition = input(f"\n Exposition (µs) [{self.app.processing.exposition}]: ").strip()
        if exposition:
            self.app.processing.exposition = int(exposition)

        # Gain
        gain = input(f" Gain (0-{self.app.camera_config.max_gain}) [{self.app.processing.gain}]: ").strip()
        if gain:
            self.app.processing.gain = int(gain)

        # Flips
        flip_v = input(f" Flip vertical (y/n) [{'y' if self.app.processing.flip_vertical else 'n'}]: ").strip().lower()
        if flip_v:
            self.app.processing.flip_vertical = (flip_v == 'y')

        flip_h = input(f" Flip horizontal (y/n) [{'y' if self.app.processing.flip_horizontal else 'n'}]: ").strip().lower()
        if flip_h:
            self.app.processing.flip_horizontal = (flip_h == 'y')

        # Denoise
        denoise = input(f" Denoise strength (0.0-1.0) [{self.app.processing.denoise_knn}]: ").strip()
        if denoise:
            self.app.processing.denoise_knn = float(denoise)

        # Sharpen
        sharpen = input(f" Sharpen amount (0.0-3.0) [{self.app.processing.sharpen_amount}]: ").strip()
        if sharpen:
            self.app.processing.sharpen_amount = float(sharpen)

        print("\n✓ Processing settings updated")

    def configure_filters(self):
        """Enable/disable filters."""
        if self.app.camera_config is None:
            print("\n✗ Please select a camera first (option 2)")
            return

        print("\n" + "="*70)
        print("FILTER CONFIGURATION")
        print("="*70)

        filters = [
            ('Sharpen 1', 'filter_enabled_sharpen1'),
            ('Denoise Paillou', 'filter_enabled_denoise_paillou'),
            ('CLAHE Contrast', 'filter_enabled_clahe'),
            ('Saturation', 'filter_enabled_sat'),
            ('3-Frame Noise Removal', 'filter_enabled_3fnr'),
            ('Ghost Reducer', 'filter_enabled_gr'),
        ]

        print("\n Enable/Disable filters:")
        for name, attr in filters:
            current = getattr(self.app.processing, attr)
            response = input(f" {name} (y/n) [{'y' if current else 'n'}]: ").strip().lower()
            if response:
                setattr(self.app.processing, attr, (response == 'y'))

        print("\n✓ Filter configuration updated")

    def start_acquisition(self):
        """Start simulated camera acquisition."""
        if self.app.camera_config is None:
            print("\n✗ Please select a camera first (option 2)")
            return

        print("\n" + "="*70)
        print("SIMULATED ACQUISITION")
        print("="*70)

        # Get current resolution
        resolution = self.app.get_current_resolution()

        # Create simulated camera
        print(f"\n🎥 Creating simulated camera...")
        self.camera = create_simulated_camera(
            self.app.camera_config.model,
            resolution,
            self.app.camera_config.sensor_bits
        )

        # Configure camera
        self.camera.set_exposition(self.app.processing.exposition)
        self.camera.set_gain(self.app.processing.gain)

        # Start capture
        print(f"\n▶ Starting acquisition...")
        self.camera.start_capture()
        self.app.acquisition_running = True

        # Capture some frames
        num_frames = 10
        print(f"\n📸 Capturing {num_frames} frames...")

        for i in range(num_frames):
            frame = self.camera.capture_frame()
            if frame is not None:
                mean_val = frame.mean()
                max_val = frame.max()
                print(f"  Frame {i+1}/{num_frames}: Mean={mean_val:.1f}, Max={max_val}, Shape={frame.shape}")

        # Stop capture
        self.camera.stop_capture()
        self.app.acquisition_running = False

        print(f"\n✓ Captured {num_frames} frames successfully")
        print(f"\n💡 In real application, these frames would be:")
        print(f"   • Debayered (if color camera)")
        print(f"   • Processed through filter pipeline")
        print(f"   • Displayed in GUI")
        print(f"   • Saved to disk (if recording)")

    def view_configuration(self):
        """View current configuration."""
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)

        if self.app.camera_config:
            print(f"\n📷 Camera: {self.app.camera_config.model}")
            resolution = self.app.get_current_resolution()
            print(f"  • Resolution: {resolution[0]}x{resolution[1]} (mode {self.app.resolution_mode}, BIN{self.app.binning_mode})")
            print(f"  • Sensor: {self.app.camera_config.sensor_factor} ({self.app.camera_config.sensor_bits}-bit)")
            print(f"  • Connected: {'Yes' if self.app.camera_connected else 'No'}")
        else:
            print("\n📷 Camera: Not configured")

        print(f"\n🎛️  Processing:")
        print(f"  • Exposition: {self.app.processing.exposition} µs")
        print(f"  • Gain: {self.app.processing.gain}")
        print(f"  • Flip V/H: {self.app.processing.flip_vertical}/{self.app.processing.flip_horizontal}")
        print(f"  • Denoise: {self.app.processing.denoise_knn}")
        print(f"  • Sharpen: {self.app.processing.sharpen_amount}")

        # List enabled filters
        enabled_filters = []
        if self.app.processing.filter_enabled_sharpen1:
            enabled_filters.append("Sharpen")
        if self.app.processing.filter_enabled_denoise_paillou:
            enabled_filters.append("Denoise")
        if self.app.processing.filter_enabled_clahe:
            enabled_filters.append("CLAHE")
        if self.app.processing.filter_enabled_sat:
            enabled_filters.append("Saturation")
        if self.app.processing.filter_enabled_3fnr:
            enabled_filters.append("3FNR")

        print(f"\n🎚️  Enabled Filters: {', '.join(enabled_filters) if enabled_filters else 'None'}")

        print(f"\n📊 Status:")
        print(f"  • Acquisition: {'Running' if self.app.acquisition_running else 'Stopped'}")

        if self.camera:
            stats = self.camera.get_stats()
            print(f"  • Frames captured: {stats['frames_captured']}")

    def save_configuration(self):
        """Save configuration to JSON."""
        import json

        if self.app.camera_config is None:
            print("\n✗ Please select a camera first (option 2)")
            return

        print("\n" + "="*70)
        print("SAVE CONFIGURATION")
        print("="*70)

        filename = input("\n Save to file [config.json]: ").strip()
        if not filename:
            filename = "config.json"

        # Build configuration dictionary
        config = {
            'camera_model': self.app.camera_config.model,
            'resolution_mode': self.app.resolution_mode,
            'binning_mode': self.app.binning_mode,
            'processing': {
                'exposition': self.app.processing.exposition,
                'gain': self.app.processing.gain,
                'flip_vertical': self.app.processing.flip_vertical,
                'flip_horizontal': self.app.processing.flip_horizontal,
                'denoise_knn': self.app.processing.denoise_knn,
                'sharpen_amount': self.app.processing.sharpen_amount,
                'filter_enabled_sharpen1': self.app.processing.filter_enabled_sharpen1,
                'filter_enabled_denoise_paillou': self.app.processing.filter_enabled_denoise_paillou,
                'filter_enabled_clahe': self.app.processing.filter_enabled_clahe,
                'filter_enabled_sat': self.app.processing.filter_enabled_sat,
            }
        }

        # Save to file
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Configuration saved to {filename}")

    def load_configuration(self):
        """Load configuration from JSON."""
        import json

        print("\n" + "="*70)
        print("LOAD CONFIGURATION")
        print("="*70)

        filename = input("\n Load from file [config.json]: ").strip()
        if not filename:
            filename = "config.json"

        if not os.path.exists(filename):
            print(f"\n✗ File not found: {filename}")
            return

        # Load from file
        with open(filename, 'r') as f:
            config = json.load(f)

        # Apply configuration
        print(f"\n✓ Loading configuration from {filename}...")

        self.app.camera_config = get_camera_config(config['camera_model'])
        self.app.resolution_mode = config['resolution_mode']
        self.app.binning_mode = config['binning_mode']

        proc = config['processing']
        self.app.processing.exposition = proc['exposition']
        self.app.processing.gain = proc['gain']
        self.app.processing.flip_vertical = proc['flip_vertical']
        self.app.processing.flip_horizontal = proc['flip_horizontal']
        self.app.processing.denoise_knn = proc['denoise_knn']
        self.app.processing.sharpen_amount = proc['sharpen_amount']
        self.app.processing.filter_enabled_sharpen1 = proc['filter_enabled_sharpen1']
        self.app.processing.filter_enabled_denoise_paillou = proc['filter_enabled_denoise_paillou']
        self.app.processing.filter_enabled_clahe = proc['filter_enabled_clahe']
        self.app.processing.filter_enabled_sat = proc['filter_enabled_sat']

        self.app.camera_connected = True

        print(f"\n✓ Configuration loaded successfully")
        print(f"  • Camera: {self.app.camera_config.model}")
        print(f"  • Exposition: {self.app.processing.exposition} µs")
        print(f"  • Gain: {self.app.processing.gain}")

    def run(self):
        """Run the interactive demo."""
        self.print_header()

        while True:
            self.print_menu()

            choice = input("\n Select option (1-9): ").strip()

            if choice == '1':
                self.list_cameras()
            elif choice == '2':
                self.select_camera()
            elif choice == '3':
                self.configure_processing()
            elif choice == '4':
                self.configure_filters()
            elif choice == '5':
                self.start_acquisition()
            elif choice == '6':
                self.view_configuration()
            elif choice == '7':
                self.save_configuration()
            elif choice == '8':
                self.load_configuration()
            elif choice == '9':
                print("\n👋 Goodbye!\n")
                break
            else:
                print("\n✗ Invalid option")

            input("\n⏎ Press Enter to continue...")


def main():
    """Main entry point."""
    try:
        demo = JetsonSkyDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user\n")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
