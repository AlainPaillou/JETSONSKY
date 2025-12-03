"""
Camera State - Camera configuration and acquisition state.
"""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class CameraState:
    """
    Holds camera configuration and acquisition state.

    This includes camera hardware settings, resolution, exposure,
    gain, and acquisition control flags.
    """
    # Camera object reference
    camera: Any = None

    # Camera status flags
    flag_camera_ok: bool = False
    flag_supported_camera: bool = False
    flag_colour_camera: bool = True
    flag_autorise_acquisition: bool = False
    flag_stop_acquisition: bool = False
    flag_premier_demarrage: bool = True

    # Resolution
    res_cam_x: int = 3096
    res_cam_y: int = 2080
    res_cam_x_base: int = 3096
    res_cam_y_base: int = 2080
    cam_displ_x: int = 1350
    cam_displ_y: int = 1012
    sensor_factor: str = "4/3"

    # Exposure and gain
    val_exposition: int = 1000
    exposition: int = 0
    timeoutexp: int = 500
    val_gain: int = 100
    val_maxgain: int = 600
    exp_min: int = 100
    exp_max: int = 10000
    exp_delta: int = 100
    exp_interval: int = 2000

    # BIN and format
    mode_BIN: int = 1
    flag_BIN2: bool = False
    format_capture: int = 0  # asi.ASI_IMG_RAW16
    sensor_bits_depth: int = 14
    flag_16b: bool = False
    TH_16B: int = 16
    threshold_16bits: int = 65535

    # Bayer pattern
    Camera_Bayer: str = "RAW"
    GPU_BAYER: int = 0  # 0:RAW 1:RGGB 2:BGGR 3:GRBG 4:GBRG
    type_debayer: int = 0

    # USB and other settings
    val_USB: int = 95
    ASIGAMMA: int = 50
    ASIAUTOMAXBRIGHTNESS: int = 50

    # Read speed
    flag_read_speed: str = "Slow"
    flag_acq_rapide: str = "MedF"

    # RGB balance
    val_red: int = 63
    val_blue: int = 74

    # Autoexposure
    flag_autoexposure_exposition: bool = False
    flag_autoexposure_gain: bool = False

    # Resolution tables (for different sensor ratios)
    RES_X_BIN1: List[int] = field(default_factory=lambda: [3096, 2560, 1920, 1600, 1280, 1024, 800, 640, 320])
    RES_Y_BIN1: List[int] = field(default_factory=lambda: [2080, 1920, 1440, 1200, 960, 768, 600, 480, 240])
    RES_X_BIN2: List[int] = field(default_factory=lambda: [1544, 1280, 960, 800, 640, 512, 400])
    RES_Y_BIN2: List[int] = field(default_factory=lambda: [1040, 960, 720, 600, 480, 384, 300])

    # Filter wheel
    flag_filter_wheel: bool = False

    # Error tracking
    nb_erreur: int = 0
