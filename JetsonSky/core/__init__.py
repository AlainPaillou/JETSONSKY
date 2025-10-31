"""
JetsonSky Core Module

This module contains the core business logic and configuration classes
for the JetsonSky astronomy imaging application.
"""

from .config import CameraConfig, ProcessingState, AppState, MountState, CaptureState, QualityMetrics
from .camera_models import get_camera_config, get_supported_cameras, is_camera_supported, CAMERA_MODELS

__all__ = [
    # Config classes
    'CameraConfig',
    'ProcessingState',
    'AppState',
    'MountState',
    'CaptureState',
    'QualityMetrics',
    # Camera model registry functions
    'get_camera_config',
    'get_supported_cameras',
    'is_camera_supported',
    'CAMERA_MODELS',
]
