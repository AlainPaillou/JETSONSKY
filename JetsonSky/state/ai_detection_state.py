"""
AI Detection State - YOLOv8 model and detection state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from collections import defaultdict
import numpy as np


@dataclass
class AIDetectionState:
    """
    Holds AI model and detection state for craters and satellites.

    This includes model references, detection flags, and tracking history.
    """
    # Detection enable flags
    flag_AI_Craters: bool = False
    flag_AI_Satellites: bool = False
    flag_AI_Trace: bool = False

    # Model loaded flags
    flag_crater_model_loaded: bool = False
    flag_satellites_model_loaded: bool = False

    # Model references
    model_craters_predict: Any = None
    model_craters_track: Any = None
    model_satellites_predict: Any = None
    model_satellites_track: Any = None

    # Tracking history (using default factory to avoid mutable default)
    track_crater_history: Dict = field(default_factory=lambda: defaultdict(list))
    track_satellite_history: Dict = field(default_factory=lambda: defaultdict(list))

    # Detection confidence threshold
    CONFIDENCE_THRESHOLD_LIMIT_CRATERS: float = 0.1

    # Star detection
    flag_DETECT_STARS: int = 0
    stars_x: Any = field(default_factory=lambda: np.zeros(1000000, dtype=int))
    stars_y: Any = field(default_factory=lambda: np.zeros(1000000, dtype=int))
    stars_s: Any = field(default_factory=lambda: np.zeros(1000000, dtype=int))
    nb_stars: int = 0

    # Satellite tracking state
    nb_sat: int = 0
    max_sat: int = 20
    compteur_sat: int = 0
    old_sat: int = 0
    flag_first_sat_pass: bool = True
    nb_trace_sat: int = -1
    flag_sat_detected: bool = False
    flag_sat_exist: bool = False

    # Satellite tracking/removal flags
    flag_TRKSAT: int = 0
    flag_REMSAT: int = 0
    flag_CONST: int = 0

    # Satellite arrays (will be numpy arrays)
    sat_x: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    sat_y: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    sat_s: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    sat_old_x: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    sat_old_y: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    sat_old_dx: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    sat_old_dy: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    sat_id: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    sat_old_id: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    correspondance: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))
    sat_speed: Any = field(default_factory=lambda: np.zeros(100000, dtype=int))

    # Frame targeting for satellite detection
    sat_frame_target: int = 5
    sat_frame_count: int = 0
    sat_frame_target_AI: int = 5
    sat_frame_count_AI: int = 0
