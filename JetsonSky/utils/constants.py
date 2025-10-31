"""
Constants for JetsonSky application.

This module contains all magic numbers and constants that were scattered
throughout the monolithic code. Instead of hardcoding values, they're
centralized here for easy maintenance and modification.
"""

# =============================================================================
# GUI COLORS
# =============================================================================

COLOR_TURQUOISE = "#40E0D0"
COLOR_BLUE = "#0000FF"
COLOR_RED = "#FF0000"
COLOR_GREEN = "#00FF00"
COLOR_YELLOW = "#FFFF00"
COLOR_GRAY = "#808080"
COLOR_WHITE = "#FFFFFF"
COLOR_BLACK = "#000000"


# =============================================================================
# DEFAULT CAMERA SETTINGS
# =============================================================================

# Default exposure in microseconds
DEFAULT_EXPOSITION = 1000

# Default gain value (0-600)
DEFAULT_GAIN = 100

# Maximum gain value for most cameras
DEFAULT_MAX_GAIN = 600

# Default USB bandwidth
# Linux (especially Jetson) needs lower bandwidth
DEFAULT_USB_BANDWIDTH_LINUX = 70
# Windows can handle higher bandwidth
DEFAULT_USB_BANDWIDTH_WINDOWS = 95

# Exposure timeout in milliseconds
TIMEOUT_EXPOSURE_MS = 500


# =============================================================================
# IMAGE PROCESSING DEFAULTS
# =============================================================================

# Denoise strength (0.0-1.0)
DEFAULT_DENOISE = 0.4

# KNN denoise strength
DEFAULT_DENOISE_KNN = 0.2

# Denoise threshold (0-255)
DEFAULT_DENOISE_THRESHOLD = 180

# Contrast CLAHE
DEFAULT_CONTRAST_CLAHE = 1.0
DEFAULT_GRID_CLAHE = 8

# Histogram stretch
DEFAULT_HISTO_MIN = 0
DEFAULT_HISTO_MAX = 255

# Histogram phi/theta
DEFAULT_PHI = 1.0
DEFAULT_THETA = 100

# Histogram equalization
DEFAULT_HEQ2 = 1.0

# Color balance (0-127)
DEFAULT_RED_BALANCE = 63
DEFAULT_BLUE_BALANCE = 74

# RGB multipliers (0.0-2.0)
DEFAULT_RED_MULTIPLIER = 1.0
DEFAULT_GREEN_MULTIPLIER = 1.0
DEFAULT_BLUE_MULTIPLIER = 1.0

# Saturation (0.0-3.0)
DEFAULT_SATURATION = 1.0

# Sharpen settings
DEFAULT_SHARPEN_AMOUNT = 1.0
DEFAULT_SHARPEN_SIGMA = 1.0
DEFAULT_SHARPEN2_AMOUNT = 1.0
DEFAULT_SHARPEN2_SIGMA = 2.0

# Ghost reducer (0-100)
DEFAULT_GHOST_REDUCER = 50

# Star amplification (0.0-3.0)
DEFAULT_STAR_AMPLIFICATION = 1.0

# Reduce variation (0-10)
DEFAULT_REDUCE_VARIATION = 1

# 3-frame noise removal threshold (0.0-1.0)
DEFAULT_3FNR_THRESHOLD = 0.5

# Frame stacking
DEFAULT_FRAME_STACKING = 1

# Blur filter reference (0-100)
DEFAULT_BFR = 50


# =============================================================================
# BIT DEPTH AND THRESHOLDS
# =============================================================================

# Sensor bit depth (12, 14, or 16)
DEFAULT_SENSOR_BITS = 14

# 16-bit threshold
DEFAULT_TH_16B = 16
MAX_16BIT_VALUE = 2 ** DEFAULT_TH_16B - 1  # 65535

# 8-bit max
MAX_8BIT_VALUE = 255


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

# Main window font size (5-7)
MAIN_WINDOW_FONT_SIZE = 6

# Display scale factor
DISPLAY_SCALE_FACTOR = 1.0

# Default display size
DEFAULT_DISPLAY_WIDTH = 1350
DEFAULT_DISPLAY_HEIGHT = 1012

# Display offset for positioning
DEFAULT_DISPLAY_OFFSET = 0


# =============================================================================
# CAPTURE SETTINGS
# =============================================================================

# Default number of captures
DEFAULT_NUM_CAPTURES = 1

# Default number of video frames
DEFAULT_NUM_VIDEO_FRAMES = 100

# Default capture interval in seconds
DEFAULT_CAPTURE_INTERVAL = 0.0


# =============================================================================
# FRAME PROCESSING
# =============================================================================

# Frame limit for buffering
DEFAULT_FRAME_LIMIT = 20

# Frame skip for processing
DEFAULT_FRAME_SKIP = 3

# Frame stacking divisor
DEFAULT_STACK_DIV = 1


# =============================================================================
# EXPOSURE LIMITS
# =============================================================================

# Minimum exposure in microseconds
EXP_MIN_US = 100

# Maximum exposure in microseconds
EXP_MAX_US = 10000

# Exposure delta in microseconds
EXP_DELTA_US = 100

# Exposure interval in milliseconds
EXP_INTERVAL_MS = 2000


# =============================================================================
# CUDA/CUPY SETTINGS
# =============================================================================

# Number of CUDA threads (X dimension)
NB_THREADS_X = 32

# Number of CUDA threads (Y dimension)
NB_THREADS_Y = 32


# =============================================================================
# IMAGE QUALITY METRICS
# =============================================================================

# Laplacian kernel size for quality measurement
LAPLACIAN_KERNEL_SIZE = 7

# Sobel kernel size for quality measurement
SOBEL_SIZE = 5

# Alpha for contrast correction
ALPHA_CONTRAST = 5

# Beta for contrast correction
BETA_CONTRAST = 1.0


# =============================================================================
# ASI CAMERA SPECIFIC
# =============================================================================

# ASI Gamma
DEFAULT_ASI_GAMMA = 50

# ASI Auto Max Brightness
DEFAULT_ASI_AUTO_MAX_BRIGHTNESS = 50


# =============================================================================
# BINNING MODES
# =============================================================================

BIN_MODE_1 = 1
BIN_MODE_2 = 2


# =============================================================================
# BAYER PATTERNS
# =============================================================================

BAYER_RAW = 0
BAYER_RGGB = 1
BAYER_BGGR = 2
BAYER_GRBG = 3
BAYER_GBRG = 4


# =============================================================================
# DEBAYER TYPES
# =============================================================================

DEBAYER_TYPE_NONE = 0
DEBAYER_TYPE_BILINEAR = 1
DEBAYER_TYPE_VNG = 2
DEBAYER_TYPE_EDGE_AWARE = 3


# =============================================================================
# SENSOR ASPECT RATIOS
# =============================================================================

SENSOR_RATIO_4_3 = "4_3"
SENSOR_RATIO_16_9 = "16_9"
SENSOR_RATIO_1_1 = "1_1"


# =============================================================================
# HDR METHODS
# =============================================================================

HDR_METHOD_MERTENS = "Mertens"
HDR_METHOD_MEDIAN = "Median"
HDR_METHOD_MEAN = "Mean"


# =============================================================================
# IMAGE QUALITY METHODS
# =============================================================================

IQ_METHOD_SOBEL = "Sobel"
IQ_METHOD_LAPLACIAN = "Laplacian"


# =============================================================================
# STACKING MODES
# =============================================================================

STACKING_MODE_MEAN = "mean"
STACKING_MODE_SUM = "sum"


# =============================================================================
# AI MODEL PATHS
# =============================================================================

# Moon crater detection model
MOON_CRATER_MODEL_PATH = "./AI_models/AI_craters_model6_8s_3c_180e.pt"

# Satellite detection model
SATELLITES_MODEL_PATH = "./AI_models/AI_Sat_model1_8n_3c_300e.pt"

# Custom satellite tracker configuration
SATELLITES_TRACKER_CONFIG_PATH = "./AI_models/sattelite_custom_tracker.yaml"


# =============================================================================
# DIRECTORY PATHS
# =============================================================================

# Default image output directory
DEFAULT_IMAGE_PATH = "./Images"

# Default video output directory
DEFAULT_VIDEO_PATH = "./Videos"

# Library directory for SDK files
DEFAULT_LIB_PATH = "./Lib"


# =============================================================================
# FLIP TYPES
# =============================================================================

FLIP_TYPE_NONE = "none"
FLIP_TYPE_VERTICAL = "vertical"
FLIP_TYPE_HORIZONTAL = "horizontal"
FLIP_TYPE_BOTH = "both"


# =============================================================================
# VIDEO CODECS
# =============================================================================

VIDEO_CODEC_RAW = "RAW"
VIDEO_CODEC_H264 = "H264"
VIDEO_CODEC_H265 = "H265"
VIDEO_CODEC_MJPEG = "MJPEG"


# =============================================================================
# FILE FORMATS
# =============================================================================

IMAGE_FORMAT_TIFF = "TIFF"
IMAGE_FORMAT_PNG = "PNG"
IMAGE_FORMAT_JPEG = "JPEG"
VIDEO_FORMAT_SER = "SER"
VIDEO_FORMAT_AVI = "AVI"
VIDEO_FORMAT_MP4 = "MP4"
VIDEO_FORMAT_MOV = "MOV"


# =============================================================================
# SER FILE FORMATS
# =============================================================================

SER_DEPTH_8BIT = 8
SER_DEPTH_16BIT = 16


# =============================================================================
# KEYBOARD LAYOUTS
# =============================================================================

KEYBOARD_LAYOUT_AZERTY = "AZERTY"
KEYBOARD_LAYOUT_QWERTY = "QWERTY"


# =============================================================================
# PLATFORM DETECTION
# =============================================================================

PLATFORM_WINDOWS = "Windows"
PLATFORM_LINUX = "Linux"


# =============================================================================
# ARCHITECTURE DETECTION
# =============================================================================

ARCH_AARCH64 = "aarch64"  # ARM64 (Jetson)
ARCH_X86_64 = "x86_64"    # Intel/AMD 64-bit


# =============================================================================
# GUI POSITIONS AND SIZES
# =============================================================================

# Dark position
DEFAULT_XDARK = 1750
DEFAULT_YDARK = 843

# Light indicator position
DEFAULT_XLI1 = 1475
DEFAULT_YLI1 = 1000


# =============================================================================
# MOUNT TRACKING
# =============================================================================

# Default observer position (can be configured)
DEFAULT_LATITUDE = 0.0
DEFAULT_LONGITUDE = 0.0


# =============================================================================
# OPENCV FONT
# =============================================================================

# Font for text overlay (imported from cv2 in actual usage)
# cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE = 1.0
DEFAULT_FONT_THICKNESS = 1


# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Maximum queue size for frame buffering
MAX_FRAME_QUEUE_SIZE = 10

# Timeout for thread joins (seconds)
THREAD_JOIN_TIMEOUT = 5.0


# =============================================================================
# ERROR HANDLING
# =============================================================================

# Maximum consecutive errors before stopping
MAX_CONSECUTIVE_ERRORS = 10

# Retry attempts for camera operations
MAX_CAMERA_RETRIES = 3


# =============================================================================
# DEBUG FLAGS
# =============================================================================

# Enable debug logging
DEBUG_ENABLED = False

# Enable performance profiling
PROFILING_ENABLED = False


# =============================================================================
# VERSION INFORMATION
# =============================================================================

# Application version (will be set by main application)
JETSONSKY_VERSION = "V53_07RC"


# =============================================================================
# SUPPORTED FEATURES FLAGS
# =============================================================================

# Features that can be enabled/disabled
FEATURE_GPU_ACCELERATION = True
FEATURE_AI_DETECTION = True
FEATURE_HDR_PROCESSING = True
FEATURE_MOUNT_CONTROL = True
FEATURE_FILTER_WHEEL = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_usb_bandwidth_for_platform(platform: str) -> int:
    """
    Get recommended USB bandwidth for platform.

    Args:
        platform: Platform name ("Windows" or "Linux")

    Returns:
        Recommended USB bandwidth value
    """
    if platform == PLATFORM_WINDOWS:
        return DEFAULT_USB_BANDWIDTH_WINDOWS
    else:
        return DEFAULT_USB_BANDWIDTH_LINUX


def get_max_value_for_bits(bits: int) -> int:
    """
    Get maximum pixel value for given bit depth.

    Args:
        bits: Bit depth (8, 12, 14, or 16)

    Returns:
        Maximum pixel value (2^bits - 1)
    """
    return 2 ** bits - 1
