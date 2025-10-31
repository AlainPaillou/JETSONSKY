"""
Configuration classes for JetsonSky application.

This module defines dataclasses for managing camera configuration,
image processing state, and application state. These replace the
300+ global variables from the monolithic implementation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime


@dataclass
class CameraConfig:
    """
    Camera configuration for a specific ZWO ASI camera model.

    This replaces the massive if-elif chain in init_camera() that was 1,500 lines.
    Instead of hardcoding camera configs, they're stored in a registry (camera_models.py).

    Attributes:
        model: Camera model name (e.g., "ZWO ASI178MC")
        resolution_x: Maximum sensor width in pixels
        resolution_y: Maximum sensor height in pixels
        display_x: Display window width in pixels
        display_y: Display window height in pixels
        sensor_factor: Sensor aspect ratio ("4_3", "16_9", or "1_1")
        sensor_bits: Bit depth of sensor (12, 14, or 16 bits)
        bayer_pattern: Bayer pattern for color cameras ("RGGB", "BGGR", "GRBG", "GBRG", or "MONO")
        supported_resolutions_bin1: List of (width, height) tuples for BIN1 mode
        supported_resolutions_bin2: List of (width, height) tuples for BIN2 mode
        max_gain: Maximum gain value supported by camera
        usb_bandwidth: Default USB bandwidth (40 for Linux, 95 for Windows)
    """
    model: str
    resolution_x: int
    resolution_y: int
    display_x: int
    display_y: int
    sensor_factor: str  # "4_3", "16_9", or "1_1"
    sensor_bits: int
    bayer_pattern: str
    supported_resolutions_bin1: List[Tuple[int, int]]
    supported_resolutions_bin2: List[Tuple[int, int]]
    max_gain: int
    usb_bandwidth: int


@dataclass
class ProcessingState:
    """
    Current image processing parameters.

    This encapsulates all the processing-related global variables
    from the monolithic file. Instead of 100+ scattered globals,
    all processing state is in one place.

    Attributes:
        exposition: Exposure time in microseconds
        gain: Camera gain value (0-600)
        usb_bandwidth: USB bandwidth limit

        # Flip and transforms
        flip_vertical: Enable vertical flip
        flip_horizontal: Enable horizontal flip
        image_negative: Enable negative image

        # Color balance
        red_balance: Red channel balance (0-127)
        blue_balance: Blue channel balance (0-127)
        red_multiplier: Red channel multiplier (0.0-2.0)
        green_multiplier: Green channel multiplier (0.0-2.0)
        blue_multiplier: Blue channel multiplier (0.0-2.0)

        # Denoise filters
        denoise_strength: General denoise strength (0.0-1.0)
        denoise_knn: KNN denoise strength (0.0-1.0)
        denoise_threshold: Denoise threshold value (0-255)

        # Sharpen filters
        sharpen_amount: Sharpen filter 1 amount (0.0-3.0)
        sharpen_sigma: Sharpen filter 1 sigma (0.0-5.0)
        sharpen2_amount: Sharpen filter 2 amount (0.0-3.0)
        sharpen2_sigma: Sharpen filter 2 sigma (0.0-5.0)

        # Contrast and histogram
        contrast_clahe: CLAHE contrast limit (0.0-4.0)
        grid_clahe: CLAHE grid size (4, 8, 16, 32)
        histo_min: Histogram stretch minimum (0-255)
        histo_max: Histogram stretch maximum (0-255)
        phi: Histogram phi parameter (0.0-2.0)
        theta: Histogram theta parameter (0-255)
        heq2: Histogram equalization amount (0.0-2.0)

        # Color and saturation
        saturation: Color saturation (0.0-3.0)

        # Advanced filters
        ghost_reducer: Ghost reduction strength (0-100)
        star_amplification: Star amplification amount (0.0-3.0)
        gradient_vignetting: Gradient/vignetting mode (0=off, 1=gradient, 2=vignetting)
        reduce_variation: Frame variation reduction (0-10)
        three_frame_noise_threshold: 3-frame noise removal threshold (0.0-1.0)

        # Stacking
        frame_stacking: Number of frames to stack (1-10)
        stacking_mode: Stacking mode ("mean" or "sum")

        # HDR
        hdr_enabled: Enable HDR processing
        hdr_method: HDR method ("Mertens", "Median", "Mean")

        # AI detection
        ai_craters_enabled: Enable moon crater detection
        ai_satellites_enabled: Enable satellite detection
        ai_trace_enabled: Enable satellite trace mode

        # Filter toggles (each can be enabled/disabled)
        filter_enabled_tip: Time Integration Processing
        filter_enabled_sat: Saturation enhancement
        filter_enabled_sat2pass: Two-pass saturation
        filter_enabled_sharpen1: Sharpen filter 1
        filter_enabled_sharpen2: Sharpen filter 2
        filter_enabled_nlm2: NLM2 denoise
        filter_enabled_denoise_paillou: Paillou denoise 1
        filter_enabled_denoise_paillou2: Paillou denoise 2
        filter_enabled_hst: Histogram stretch
        filter_enabled_clahe: CLAHE contrast
        filter_enabled_cll: Contrast low light
        filter_enabled_aanr: Adaptive absorber noise removal
        filter_enabled_aanrb: Adaptive absorber noise removal (back)
        filter_enabled_3fnr: 3-frame noise removal (front)
        filter_enabled_3fnrb: 3-frame noise removal (back)
        filter_enabled_gr: Ghost reducer
        filter_enabled_hdr: HDR processing
        filter_enabled_hotpix: Hot pixel removal
        filter_enabled_blur_ref: Blur reference
        filter_enabled_sub_ref: Subtract reference
    """
    # Camera settings
    exposition: int = 1000  # microseconds
    gain: int = 100
    usb_bandwidth: int = 70

    # Flip and transforms
    flip_vertical: bool = False
    flip_horizontal: bool = False
    image_negative: bool = False

    # Color balance
    red_balance: int = 63
    blue_balance: int = 74
    red_multiplier: float = 1.0
    green_multiplier: float = 1.0
    blue_multiplier: float = 1.0

    # Denoise filters
    denoise_strength: float = 0.4
    denoise_knn: float = 0.2
    denoise_threshold: int = 180

    # Sharpen filters
    sharpen_amount: float = 1.0
    sharpen_sigma: float = 1.0
    sharpen2_amount: float = 1.0
    sharpen2_sigma: float = 2.0

    # Contrast and histogram
    contrast_clahe: float = 1.0
    grid_clahe: int = 8
    histo_min: int = 0
    histo_max: int = 255
    phi: float = 1.0
    theta: int = 100
    heq2: float = 1.0

    # Color and saturation
    saturation: float = 1.0

    # Advanced filters
    ghost_reducer: int = 50
    star_amplification: float = 1.0
    gradient_vignetting: int = 1  # 0=off, 1=gradient, 2=vignetting
    reduce_variation: int = 1
    three_frame_noise_threshold: float = 0.5

    # Stacking
    frame_stacking: int = 1
    stacking_mode: str = "mean"  # "mean" or "sum"

    # HDR
    hdr_enabled: bool = False
    hdr_method: str = "Mertens"  # "Mertens", "Median", "Mean"

    # AI detection
    ai_craters_enabled: bool = False
    ai_satellites_enabled: bool = False
    ai_trace_enabled: bool = False

    # Filter toggles
    filter_enabled_tip: bool = False
    filter_enabled_sat: bool = False
    filter_enabled_sat2pass: bool = False
    filter_enabled_sharpen1: bool = False
    filter_enabled_sharpen2: bool = False
    filter_enabled_nlm2: bool = False
    filter_enabled_denoise_paillou: bool = False
    filter_enabled_denoise_paillou2: bool = False
    filter_enabled_hst: bool = False
    filter_enabled_clahe: bool = False
    filter_enabled_cll: bool = False
    filter_enabled_aanr: bool = False
    filter_enabled_aanrb: bool = False
    filter_enabled_3fnr: bool = False
    filter_enabled_3fnrb: bool = False
    filter_enabled_gr: bool = False
    filter_enabled_hdr: bool = False
    filter_enabled_hotpix: bool = False
    filter_enabled_blur_ref: bool = False
    filter_enabled_sub_ref: bool = False


@dataclass
class MountState:
    """
    Telescope mount state and calibration data.

    Attributes:
        mount_enabled: Enable mount tracking
        azimuth: Current azimuth in degrees
        altitude: Current altitude in degrees
        azimuth_mount: Mount-reported azimuth
        altitude_mount: Mount-reported altitude
        latitude: Observer latitude in degrees
        longitude: Observer longitude in degrees
        calibrated: Mount calibration status
    """
    mount_enabled: bool = False
    azimuth: float = 0.0
    altitude: float = 0.0
    azimuth_mount: float = 0.0
    altitude_mount: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    calibrated: bool = False


@dataclass
class CaptureState:
    """
    Image and video capture state.

    Attributes:
        num_captures: Number of images to capture
        num_video_frames: Number of video frames to capture
        video_recording: Currently recording video
        image_capturing: Currently capturing images
        capture_interval: Time between captures in seconds
        trigger_countdown: Countdown timer for triggered capture
        frame_number: Current frame number
        video_frame_position: Current position in video playback
    """
    num_captures: int = 1
    num_video_frames: int = 100
    video_recording: bool = False
    image_capturing: bool = False
    capture_interval: float = 0.0
    trigger_countdown: int = 0
    frame_number: int = 0
    video_frame_position: int = 0


@dataclass
class QualityMetrics:
    """
    Image quality tracking metrics.

    Attributes:
        method: Quality measurement method ("Sobel", "Laplacian")
        mean_quality: Mean quality score
        max_quality: Maximum quality score
        min_quality: Minimum quality score
        quality_history: List of recent quality scores
    """
    method: str = "Sobel"
    mean_quality: float = 0.0
    max_quality: float = 0.0
    min_quality: float = 10000.0
    quality_history: List[float] = field(default_factory=list)


@dataclass
class AppState:
    """
    Complete application state.

    This is the root state object that contains all configuration
    and runtime state for the JetsonSky application. Instead of
    300+ global variables scattered throughout the code, everything
    is organized in this single, well-structured object.

    Attributes:
        camera_config: Camera-specific configuration
        processing: Image processing parameters
        mount: Telescope mount state
        capture: Capture state
        quality: Quality metrics

        # Runtime state
        acquisition_running: Acquisition thread is running
        camera_connected: Camera is connected and initialized
        resolution_mode: Current resolution mode (0-8)
        binning_mode: Current binning mode (1 or 2)

        # Paths
        image_path: Directory for saving images
        video_path: Directory for saving videos

        # Display
        display_zoom: Display zoom level (0.5-2.0)
        display_offset_x: Display X offset for panning
        display_offset_y: Display Y offset for panning
        crosshair_enabled: Show crosshair on display

        # Performance
        frame_rate: Current frame rate (fps)
        last_frame_time: Timestamp of last frame
    """
    # Configuration
    camera_config: Optional[CameraConfig] = None
    processing: ProcessingState = field(default_factory=ProcessingState)
    mount: MountState = field(default_factory=MountState)
    capture: CaptureState = field(default_factory=CaptureState)
    quality: QualityMetrics = field(default_factory=QualityMetrics)

    # Runtime state
    acquisition_running: bool = False
    camera_connected: bool = False
    resolution_mode: int = 1
    binning_mode: int = 1

    # Paths
    image_path: str = "./Images"
    video_path: str = "./Videos"

    # Display
    display_zoom: float = 1.0
    display_offset_x: int = 0
    display_offset_y: int = 0
    crosshair_enabled: bool = False

    # Performance
    frame_rate: float = 0.0
    last_frame_time: float = 0.0

    def get_current_resolution(self) -> Tuple[int, int]:
        """
        Get the current camera resolution based on binning and resolution mode.

        Returns:
            Tuple of (width, height) in pixels
        """
        if self.camera_config is None:
            return (0, 0)

        if self.binning_mode == 1:
            resolutions = self.camera_config.supported_resolutions_bin1
        else:
            resolutions = self.camera_config.supported_resolutions_bin2

        if 0 <= self.resolution_mode < len(resolutions):
            return resolutions[self.resolution_mode]
        else:
            return resolutions[0] if resolutions else (0, 0)

    def get_display_size(self) -> Tuple[int, int]:
        """
        Get the display window size.

        Returns:
            Tuple of (width, height) for display window
        """
        if self.camera_config is None:
            return (1350, 1012)

        return (self.camera_config.display_x, self.camera_config.display_y)
