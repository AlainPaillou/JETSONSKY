#!/usr/bin/env python3
"""
JetsonSky GUI Demo Application

Simple Tkinter GUI demonstrating the Phase 1 refactoring.
Shows how to integrate the new modular architecture with a GUI.

Usage:
    python3 gui_demo.py
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    AppState,
    get_camera_config,
    get_supported_cameras,
)
from utils.constants import (
    DEFAULT_EXPOSITION,
    DEFAULT_GAIN,
    COLOR_TURQUOISE,
    COLOR_BLUE,
)
from demos.camera_simulator import create_simulated_camera


class JetsonSkyGUI:
    """Simple GUI demo application."""

    def __init__(self, root):
        """
        Initialize GUI.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("JetsonSky Phase 1 Demo")
        self.root.geometry("800x600")

        # Application state (using Phase 1 modules!)
        self.app = AppState()
        self.camera = None
        self.acquisition_thread = None

        # Build UI
        self.build_ui()

    def build_ui(self):
        """Build user interface."""
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            text="JetsonSky Phase 1 Demo",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title.pack(pady=15)

        # Main container
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Camera selection and settings
        left_panel = tk.LabelFrame(main, text="Camera & Settings", padx=10, pady=10)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5)

        # Camera selection
        tk.Label(left_panel, text="Camera Model:").grid(row=0, column=0, sticky="w", pady=5)

        self.camera_var = tk.StringVar()
        cameras = get_supported_cameras()
        self.camera_combo = ttk.Combobox(left_panel, textvariable=self.camera_var, width=25)
        self.camera_combo['values'] = cameras
        self.camera_combo.current(cameras.index("ZWO ASI178MC"))
        self.camera_combo.grid(row=0, column=1, pady=5)

        tk.Button(
            left_panel,
            text="Load Camera",
            command=self.load_camera,
            bg="#3498db",
            fg="white"
        ).grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        # Exposition
        tk.Label(left_panel, text="Exposition (µs):").grid(row=2, column=0, sticky="w", pady=5)
        self.exposition_var = tk.IntVar(value=DEFAULT_EXPOSITION)
        tk.Scale(
            left_panel,
            from_=100,
            to=100000,
            orient=tk.HORIZONTAL,
            variable=self.exposition_var,
            command=self.update_exposition
        ).grid(row=2, column=1, sticky="ew")

        self.exposition_label = tk.Label(left_panel, text=f"{DEFAULT_EXPOSITION} µs")
        self.exposition_label.grid(row=3, column=1, sticky="w")

        # Gain
        tk.Label(left_panel, text="Gain:").grid(row=4, column=0, sticky="w", pady=5)
        self.gain_var = tk.IntVar(value=DEFAULT_GAIN)
        tk.Scale(
            left_panel,
            from_=0,
            to=600,
            orient=tk.HORIZONTAL,
            variable=self.gain_var,
            command=self.update_gain
        ).grid(row=4, column=1, sticky="ew")

        self.gain_label = tk.Label(left_panel, text=f"{DEFAULT_GAIN}")
        self.gain_label.grid(row=5, column=1, sticky="w")

        # Flips
        self.flip_v_var = tk.BooleanVar()
        tk.Checkbutton(
            left_panel,
            text="Flip Vertical",
            variable=self.flip_v_var,
            command=self.update_flips
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=5)

        self.flip_h_var = tk.BooleanVar()
        tk.Checkbutton(
            left_panel,
            text="Flip Horizontal",
            variable=self.flip_h_var,
            command=self.update_flips
        ).grid(row=7, column=0, columnspan=2, sticky="w")

        # Right panel - Filters and status
        right_panel = tk.LabelFrame(main, text="Filters & Status", padx=10, pady=10)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=5)

        # Filters
        tk.Label(right_panel, text="Enable Filters:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=10
        )

        self.filter_sharpen_var = tk.BooleanVar()
        tk.Checkbutton(
            right_panel,
            text="Sharpen",
            variable=self.filter_sharpen_var,
            command=self.update_filters
        ).grid(row=1, column=0, sticky="w")

        self.filter_denoise_var = tk.BooleanVar()
        tk.Checkbutton(
            right_panel,
            text="Denoise",
            variable=self.filter_denoise_var,
            command=self.update_filters
        ).grid(row=2, column=0, sticky="w")

        self.filter_clahe_var = tk.BooleanVar()
        tk.Checkbutton(
            right_panel,
            text="CLAHE Contrast",
            variable=self.filter_clahe_var,
            command=self.update_filters
        ).grid(row=3, column=0, sticky="w")

        self.filter_sat_var = tk.BooleanVar()
        tk.Checkbutton(
            right_panel,
            text="Saturation",
            variable=self.filter_sat_var,
            command=self.update_filters
        ).grid(row=4, column=0, sticky="w")

        # Status
        tk.Label(right_panel, text="Status:", font=("Arial", 10, "bold")).grid(
            row=5, column=0, sticky="w", pady=(20, 10)
        )

        self.status_text = tk.Text(right_panel, height=10, width=35, state=tk.DISABLED)
        self.status_text.grid(row=6, column=0, sticky="nsew")

        # Bottom panel - Control buttons
        bottom_panel = tk.Frame(main)
        bottom_panel.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        tk.Button(
            bottom_panel,
            text="▶ Start Acquisition",
            command=self.start_acquisition,
            bg="#27ae60",
            fg="white",
            height=2,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            bottom_panel,
            text="⏹ Stop Acquisition",
            command=self.stop_acquisition,
            bg="#e74c3c",
            fg="white",
            height=2,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            bottom_panel,
            text="📊 View Config",
            command=self.view_config,
            bg="#3498db",
            fg="white",
            height=2,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Initial status update
        self.update_status("Ready. Please load a camera.")

    def load_camera(self):
        """Load selected camera."""
        camera_model = self.camera_var.get()

        try:
            # Load camera configuration using Phase 1 modules
            self.app.camera_config = get_camera_config(camera_model)
            self.app.camera_connected = True

            # Apply default settings
            self.app.processing.exposition = self.exposition_var.get()
            self.app.processing.gain = self.gain_var.get()

            # Set resolution
            self.app.resolution_mode = 1
            self.app.binning_mode = 1

            resolution = self.app.get_current_resolution()

            self.update_status(
                f"Camera loaded: {camera_model}\n"
                f"Resolution: {resolution[0]}x{resolution[1]}\n"
                f"Sensor: {self.app.camera_config.sensor_factor}\n"
                f"Bit depth: {self.app.camera_config.sensor_bits}"
            )

            messagebox.showinfo("Success", f"Camera {camera_model} loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load camera: {e}")

    def update_exposition(self, value):
        """Update exposition setting."""
        self.app.processing.exposition = int(float(value))
        self.exposition_label.config(text=f"{self.app.processing.exposition} µs")

    def update_gain(self, value):
        """Update gain setting."""
        self.app.processing.gain = int(float(value))
        self.gain_label.config(text=f"{self.app.processing.gain}")

    def update_flips(self):
        """Update flip settings."""
        self.app.processing.flip_vertical = self.flip_v_var.get()
        self.app.processing.flip_horizontal = self.flip_h_var.get()

    def update_filters(self):
        """Update filter settings."""
        self.app.processing.filter_enabled_sharpen1 = self.filter_sharpen_var.get()
        self.app.processing.filter_enabled_denoise_paillou = self.filter_denoise_var.get()
        self.app.processing.filter_enabled_clahe = self.filter_clahe_var.get()
        self.app.processing.filter_enabled_sat = self.filter_sat_var.get()

    def start_acquisition(self):
        """Start simulated acquisition."""
        if self.app.camera_config is None:
            messagebox.showwarning("Warning", "Please load a camera first!")
            return

        if self.app.acquisition_running:
            messagebox.showwarning("Warning", "Acquisition already running!")
            return

        # Get current resolution
        resolution = self.app.get_current_resolution()

        # Create simulated camera
        self.camera = create_simulated_camera(
            self.app.camera_config.model,
            resolution,
            self.app.camera_config.sensor_bits
        )

        # Configure camera
        self.camera.set_exposition(self.app.processing.exposition)
        self.camera.set_gain(self.app.processing.gain)

        # Start capture
        self.camera.start_capture()
        self.app.acquisition_running = True

        self.update_status("Acquisition started...")

        # Start acquisition thread
        self.acquisition_thread = threading.Thread(target=self.acquisition_loop, daemon=True)
        self.acquisition_thread.start()

    def stop_acquisition(self):
        """Stop acquisition."""
        if not self.app.acquisition_running:
            messagebox.showwarning("Warning", "Acquisition not running!")
            return

        self.app.acquisition_running = False

        if self.camera:
            self.camera.stop_capture()

        self.update_status("Acquisition stopped.")

    def acquisition_loop(self):
        """Acquisition loop (runs in separate thread)."""
        frame_count = 0

        while self.app.acquisition_running:
            if self.camera:
                frame = self.camera.capture_frame()
                if frame is not None:
                    frame_count += 1
                    mean_val = frame.mean()
                    max_val = frame.max()

                    # Update status
                    status = (
                        f"Acquiring... Frame {frame_count}\n"
                        f"Mean: {mean_val:.1f}\n"
                        f"Max: {max_val}\n"
                        f"Shape: {frame.shape}"
                    )

                    self.root.after(0, self.update_status, status)

    def view_config(self):
        """View current configuration."""
        if self.app.camera_config is None:
            messagebox.showinfo("Configuration", "No camera loaded")
            return

        resolution = self.app.get_current_resolution()

        # Get enabled filters
        filters = []
        if self.app.processing.filter_enabled_sharpen1:
            filters.append("Sharpen")
        if self.app.processing.filter_enabled_denoise_paillou:
            filters.append("Denoise")
        if self.app.processing.filter_enabled_clahe:
            filters.append("CLAHE")
        if self.app.processing.filter_enabled_sat:
            filters.append("Saturation")

        config_text = f"""
Camera: {self.app.camera_config.model}
Resolution: {resolution[0]}x{resolution[1]} (mode {self.app.resolution_mode}, BIN{self.app.binning_mode})
Sensor: {self.app.camera_config.sensor_factor} ({self.app.camera_config.sensor_bits}-bit)

Processing:
  Exposition: {self.app.processing.exposition} µs
  Gain: {self.app.processing.gain}
  Flip V/H: {self.app.processing.flip_vertical}/{self.app.processing.flip_horizontal}

Enabled Filters: {', '.join(filters) if filters else 'None'}

Status:
  Acquisition: {'Running' if self.app.acquisition_running else 'Stopped'}
  Camera connected: {self.app.camera_connected}
"""

        messagebox.showinfo("Current Configuration", config_text)

    def update_status(self, message):
        """
        Update status text.

        Args:
            message: Status message to display
        """
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(1.0, message)
        self.status_text.config(state=tk.DISABLED)


def main():
    """Main entry point."""
    root = tk.Tk()
    app = JetsonSkyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
