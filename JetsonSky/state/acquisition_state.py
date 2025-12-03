"""
Acquisition State - Video/image mode acquisition state.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AcquisitionState:
    """
    Holds video/image mode acquisition state.

    This includes file handles, frame counters, and capture settings
    for both camera and video file modes.
    """
    # Mode flags
    flag_image_mode: bool = False
    flag_image_video_loaded: bool = False
    flag_SER_file: bool = False

    # Video/file handle
    video: Any = None  # cv2.VideoCapture or Serfile
    Video_Test: str = ""
    Video_Bayer: str = "RAW"
    video_debayer: int = 0

    # Frame tracking
    video_frame_number: int = 0
    video_frame_position: int = 0
    frame_number: int = 0
    previous_frame_number: int = -1
    frame_position: int = 0
    SFN: int = 0  # Skip frame number

    # SER file depth
    SER_depth: int = 8

    # Capture flags and counters
    flag_cap_pic: bool = False
    flag_cap_video: bool = False
    flag_pause_video: bool = False
    flag_TRIGGER: int = 0
    trig_count_down: int = 0

    # Capture settings
    val_nb_captures: int = 1
    nb_cap_video: int = 0
    val_nb_capt_video: int = 100

    # High quality capture mode
    flag_HQ: int = 0

    # Frame limits
    frame_limit: int = 20
    frame_skip: int = 3

    # Reference image
    flag_capture_image_reference: bool = False
    flag_image_reference_OK: bool = False
    Image_Reference: Any = None

    # Subtract/blur reference
    flag_sub_img_ref: int = 0
    flag_Blur_img_ref: int = 0

    # Image buffers
    image_brute: Any = None
    image_brute_grey: Any = None

    # Video timing
    start_time_video: float = 0.0
    stop_time_video: float = 0.0

    # Bad frame removal
    flag_BFR: int = 0
    val_BFR: int = 50
    max_qual: int = 0
    min_qual: int = 10000

    # Image quality estimation
    flag_IQE: int = 0
    IQ_Method: str = "Sobel"
    laplacianksize: int = 7
    SobelSize: int = 5
