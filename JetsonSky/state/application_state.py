"""
Application State - Root container for all application state.

This module provides the ApplicationState class, which serves as the single
source of truth for the JetsonSky application. It replaces the ~150 global
variables that were previously used throughout the codebase.

Architecture:
    ApplicationState (root container)
    ├── CameraState        # Camera config, acquisition flags, resolution, exposure
    ├── FilterState        # Temporal filter buffers (3FNR, AANR, stacking)
    ├── FilterParams       # Filter configuration flags/values
    ├── DisplayState       # Zoom, stabilization, overlays, RGB channel shifting
    ├── AcquisitionState   # Video/image mode, file handles, frame numbers
    ├── AIDetectionState   # YOLOv8 models, satellite/crater tracking
    ├── TimingState        # FPS, execution timing
    └── astronomy          # AstronomyCalculator instance

Usage:
    from state import ApplicationState

    app_state = ApplicationState()
    app_state.init_filter_state()  # Initialize filter state objects
    app_state.init_astronomy(lat_obs=48.0175, long_obs=-4.0340)  # Optional

    # Access sub-states
    app_state.camera.val_exposition = 1000
    app_state.filter_params.flag_AANR = True
    app_state.display.flag_full_res = 1
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, TYPE_CHECKING
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .camera_state import CameraState
from .display_state import DisplayState
from .acquisition_state import AcquisitionState
from .ai_detection_state import AIDetectionState
from .timing_state import TimingState

# Import filter state classes from filter_pipeline
# These are imported lazily to avoid circular imports
FilterState = None
FilterParams = None

if TYPE_CHECKING:
    # For type checking only - actual import is lazy
    from filter_pipeline import FilterState as FilterStateType, FilterParams as FilterParamsType


def _get_filter_classes() -> tuple:
    """
    Lazily import filter classes to avoid circular imports.

    Returns:
        Tuple of (FilterState, FilterParams) classes from filter_pipeline module.
    """
    global FilterState, FilterParams
    if FilterState is None:
        from filter_pipeline import FilterState as FS, FilterParams as FP
        FilterState = FS
        FilterParams = FP
    return FilterState, FilterParams


@dataclass
class ApplicationState:
    """
    Root container for all application state.

    This is the single source of truth for the application,
    replacing all global variables.

    Attributes:
        camera: Camera configuration and acquisition state
        display: Display, zoom, and overlay state
        acquisition: Video/image mode state
        ai: AI detection state
        timing: Performance timing state
        filter_state: Temporal filter state (from filter_pipeline)
        filter_params: Filter configuration (from filter_pipeline)
        astronomy: AstronomyCalculator instance
        kernels: CUDA kernel references
        cupy_context: CuPy CUDA context

    System state:
        flag_quitter: Application quit flag
        Dev_system: "Windows" or "Linux"
        keyboard_layout: "AZERTY" or "QWERTY"
    """
    # Sub-state objects
    camera: CameraState = field(default_factory=CameraState)
    display: DisplayState = field(default_factory=DisplayState)
    acquisition: AcquisitionState = field(default_factory=AcquisitionState)
    ai: AIDetectionState = field(default_factory=AIDetectionState)
    timing: TimingState = field(default_factory=TimingState)

    # Filter state - initialized separately after filter_pipeline import
    filter_state: Any = None
    filter_params: Any = None

    # Astronomy calculator instance
    astronomy: Any = None

    # CUDA context and kernels
    cupy_context: Any = None
    kernels: Any = None  # FilterKernels instance

    # System configuration
    flag_quitter: bool = False
    Dev_system: str = "Windows"
    keyboard_layout: str = "AZERTY"
    JetsonSky_version: str = "V53_07RC"

    # Thread references
    thread_1: Any = None
    thread_2: Any = None
    thread_3: Any = None

    # Mount state
    flag_mountpos: bool = False
    flag_mount_connect: bool = False
    flag_acquisition_mount: bool = False

    # Keyboard state
    flag_keyboard_management: bool = False
    key_pressed: str = ""

    # GUI window reference
    fenetre_principale: Any = None

    def init_filter_state(self) -> None:
        """
        Initialize filter state objects from filter_pipeline module.

        Creates new FilterState and FilterParams instances if they don't exist.
        This must be called after the filter_pipeline module is available.
        """
        FS, FP = _get_filter_classes()
        if self.filter_state is None:
            self.filter_state = FS()
        if self.filter_params is None:
            self.filter_params = FP()

    def init_astronomy(self, lat_obs: float = 48.0175, long_obs: float = -4.0340,
                       alt_obs: float = 0, zone: int = 2,
                       polaris_ad: float = 2.507, polaris_dec: float = 89.25) -> None:
        """
        Initialize astronomy calculator with observer location.

        Args:
            lat_obs: Observer latitude in degrees (default: Brest, France)
            long_obs: Observer longitude in degrees (negative = west)
            alt_obs: Observer altitude in meters
            zone: Timezone offset from UTC
            polaris_ad: Polaris right ascension in hours
            polaris_dec: Polaris declination in degrees
        """
        try:
            import astronomy_utils
            self.astronomy = astronomy_utils.AstronomyCalculator(
                lat_obs=lat_obs,
                long_obs=long_obs,
                alt_obs=alt_obs,
                zone=zone,
                polaris_ad=polaris_ad,
                polaris_dec=polaris_dec
            )
        except ImportError:
            pass  # astronomy_utils not available

    def reset_all_filters(self) -> None:
        """
        Reset all temporal filter states.

        This clears all frame buffers used by multi-frame filters:
        - Frame stacking (2-5 frame sum/mean)
        - 3FNR front and back noise reduction
        - AANR adaptive absorber noise reduction
        """
        if self.filter_state:
            self.filter_state.reset_stacking()
            self.filter_state.reset_3fnr_front()
            self.filter_state.reset_3fnr_back()
            self.filter_state.reset_aanr()

    def sync_display_from_globals(self, main_globals: Dict[str, Any]) -> None:
        """
        Sync display state from main globals to self.display.

        This is called once at initialization to populate display state
        with the initial values from globals. After this, callbacks are
        responsible for keeping display state updated directly.
        """
        if main_globals is None or self.display is None:
            return

        g = main_globals
        d = self.display

        # Full resolution mode
        d.flag_full_res = g.get('flag_full_res', 0)

        # Overlay flags
        d.flag_cross = g.get('flag_cross', False)
        d.flag_TIP = g.get('flag_TIP', False)
        d.text_TIP = g.get('text_TIP', '')
        d.flag_HST = g.get('flag_HST', 0)

        # Demo mode
        d.flag_DEMO = g.get('flag_DEMO', 0)
        d.flag_demo_side = g.get('flag_demo_side', 'Left')

        # Zoom position
        d.delta_zx = g.get('delta_zx', 0)
        d.delta_zy = g.get('delta_zy', 0)

        # Stabilization
        d.delta_tx = g.get('delta_tx', 0)
        d.delta_ty = g.get('delta_ty', 0)
        d.DSW = g.get('DSW', 0)
        d.flag_STAB = g.get('flag_STAB', False)
        d.flag_Template = g.get('flag_Template', False)
        d.flag_new_stab_window = g.get('flag_new_stab_window', False)

        # RGB channel shifting
        d.delta_RX = g.get('delta_RX', 0)
        d.delta_RY = g.get('delta_RY', 0)
        d.delta_BX = g.get('delta_BX', 0)
        d.delta_BY = g.get('delta_BY', 0)

        # Flip
        d.FlipV = g.get('FlipV', 0)
        d.FlipH = g.get('FlipH', 0)
        d.type_flip = g.get('type_flip', 'none')

        # Image negative
        d.ImageNeg = g.get('ImageNeg', 0)

        # Transfer function display
        d.flag_TRSF = g.get('flag_TRSF', 0)
        d.flag_TRGS = g.get('flag_TRGS', 0)
        d.flag_TRCLL = g.get('flag_TRCLL', 0)

        # Text info overlays
        d.text_info1 = g.get('text_info1', 'Test information')
        d.text_info10 = g.get('text_info10', '')

        # Display scaling
        d.delta_s = g.get('delta_s', 0)
        d.fact_s = g.get('fact_s', 1.0)

        # Quality display
        d.mean_quality = g.get('mean_quality', 0)
        d.max_quality = g.get('max_quality', 0)
        d.quality_pos = g.get('quality_pos', 1)

        # Hold picture mode
        d.flag_hold_picture = g.get('flag_hold_picture', False)

        # Black and white mode
        d.flag_noir_blanc = g.get('flag_noir_blanc', 0)
        d.flag_reverse_RB = g.get('flag_reverse_RB', 0)

        # False colors mode
        d.flag_false_colours = g.get('flag_false_colours', 0)

        # Font for text rendering (cv2.FONT_HERSHEY_SIMPLEX = 0)
        d.font = g.get('font', None)

        # HDR mode
        d.flag_HDR = g.get('flag_HDR', False)
        d.mode_HDR = g.get('mode_HDR', 'Mertens')

        # Satellite image mode
        d.flag_SAT_Img = g.get('flag_SAT_Img', False)
