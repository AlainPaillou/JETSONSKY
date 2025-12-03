"""
JetsonSky Application State Module

This module contains dataclass definitions for the application state,
eliminating the need for global variables.

State Classes:
- CameraState: Camera configuration and acquisition state
- DisplayState: Display, zoom, stabilization settings
- AcquisitionState: Video/image mode state
- AIDetectionState: YOLOv8 model and detection state
- TimingState: Performance timing
- ApplicationState: Root container for all state objects

The existing FilterState and FilterParams from filter_pipeline.py are
reused for filter-specific state.
"""

from .camera_state import CameraState
from .display_state import DisplayState
from .acquisition_state import AcquisitionState
from .ai_detection_state import AIDetectionState
from .timing_state import TimingState
from .application_state import ApplicationState

__all__ = [
    'CameraState',
    'DisplayState',
    'AcquisitionState',
    'AIDetectionState',
    'TimingState',
    'ApplicationState',
]
