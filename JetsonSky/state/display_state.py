"""
Display State - Display, zoom, stabilization, and overlay settings.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DisplayState:
    """
    Holds display, zoom, stabilization, and overlay state.

    This includes all settings related to how images are displayed
    and manipulated visually.
    """
    # Full resolution mode
    flag_full_res: int = 0

    # Overlay flags
    flag_cross: bool = False
    flag_TIP: bool = False
    text_TIP: str = ""
    flag_HST: int = 0

    # Demo mode
    flag_DEMO: int = 0
    flag_demo_side: str = "Left"

    # Zoom position
    delta_zx: int = 0
    delta_zy: int = 0

    # Stabilization
    delta_tx: int = 0
    delta_ty: int = 0
    DSW: int = 0
    flag_STAB: bool = False
    flag_Template: bool = False
    flag_new_stab_window: bool = False

    # RGB channel shifting (chromatic correction)
    delta_RX: int = 0
    delta_RY: int = 0
    delta_BX: int = 0
    delta_BY: int = 0

    # Flip
    FlipV: int = 0
    FlipH: int = 0
    type_flip: str = "none"

    # Image negative
    ImageNeg: int = 0

    # Transfer function display
    flag_TRSF: int = 0
    flag_TRGS: int = 0
    flag_TRCLL: int = 0

    # Text info overlays
    text_info1: str = "Test information"
    text_info10: str = ""

    # Display scaling
    delta_s: int = 0
    fact_s: float = 1.0

    # Quality display
    mean_quality: int = 0
    max_quality: int = 0
    quality_pos: int = 1

    # Hold picture mode
    flag_hold_picture: bool = False

    # Black and white mode
    flag_noir_blanc: int = 0
    flag_reverse_RB: int = 0

    # False colors mode
    flag_false_colours: int = 0

    # Font for text rendering
    font: Any = None  # cv2.FONT_HERSHEY_SIMPLEX

    # HDR mode
    flag_HDR: bool = False
    mode_HDR: str = "Mertens"

    # Satellite image mode
    flag_SAT_Img: bool = False
