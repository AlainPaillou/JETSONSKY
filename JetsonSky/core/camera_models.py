"""
Camera model registry for ZWO ASI cameras.

This module replaces the 1,500-line init_camera() function that consisted
of a massive if-elif chain. Instead, camera configurations are stored in
a dictionary registry for easy lookup.

Usage:
    config = get_camera_config("ZWO ASI178MC")
    print(f"Resolution: {config.resolution_x}x{config.resolution_y}")
"""

from typing import Dict, List
from .config import CameraConfig


# Camera Model Registry
# Each entry maps a camera model name to its configuration
CAMERA_MODELS: Dict[str, CameraConfig] = {

    # ZWO ASI1600MC/MM - 16MP camera
    "ZWO ASI1600MC": CameraConfig(
        model="ZWO ASI1600MC",
        resolution_x=4656,
        resolution_y=3520,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (4656, 3520), (3600, 2700), (3000, 2250),
            (2400, 1800), (2000, 1500), (1600, 1200),
            (1280, 960), (1024, 770), (800, 600)
        ],
        supported_resolutions_bin2=[
            (2328, 1760), (1800, 1350), (1504, 1130),
            (1200, 900), (1000, 750), (640, 480), (400, 300)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI1600MM": CameraConfig(
        model="ZWO ASI1600MM",
        resolution_x=4656,
        resolution_y=3520,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (4656, 3520), (3600, 2700), (3000, 2250),
            (2400, 1800), (2000, 1500), (1600, 1200),
            (1280, 960), (1024, 770), (800, 600)
        ],
        supported_resolutions_bin2=[
            (2328, 1760), (1800, 1350), (1504, 1130),
            (1200, 900), (1000, 750), (640, 480), (400, 300)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI294MC/MM - 11.7MP camera
    "ZWO ASI294MC": CameraConfig(
        model="ZWO ASI294MC",
        resolution_x=4144,
        resolution_y=2822,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (4144, 2822), (3240, 2430), (2880, 2160),
            (2400, 1800), (1800, 1350), (1536, 1152),
            (1200, 900), (960, 720), (600, 450)
        ],
        supported_resolutions_bin2=[
            (2072, 1410), (1624, 1216), (1440, 1080),
            (1200, 900), (900, 674), (768, 576), (600, 450)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI294MC Pro": CameraConfig(
        model="ZWO ASI294MC Pro",
        resolution_x=4144,
        resolution_y=2822,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (4144, 2822), (3240, 2430), (2880, 2160),
            (2400, 1800), (1800, 1350), (1536, 1152),
            (1200, 900), (960, 720), (600, 450)
        ],
        supported_resolutions_bin2=[
            (2072, 1410), (1624, 1216), (1440, 1080),
            (1200, 900), (900, 674), (768, 576), (600, 450)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI294MM": CameraConfig(
        model="ZWO ASI294MM",
        resolution_x=8288,
        resolution_y=5644,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (8288, 5644), (6480, 4860), (5760, 4320),
            (4800, 3600), (3600, 2700), (3072, 2304),
            (2400, 1800), (1920, 1440), (1200, 900)
        ],
        supported_resolutions_bin2=[
            (4144, 2820), (3248, 2432), (2880, 2160),
            (2400, 1800), (1800, 1348), (1536, 1152), (1200, 900)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI294MM Pro": CameraConfig(
        model="ZWO ASI294MM Pro",
        resolution_x=8288,
        resolution_y=5644,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (8288, 5644), (6480, 4860), (5760, 4320),
            (4800, 3600), (3600, 2700), (3072, 2304),
            (2400, 1800), (1920, 1440), (1200, 900)
        ],
        supported_resolutions_bin2=[
            (4144, 2820), (3248, 2432), (2880, 2160),
            (2400, 1800), (1800, 1348), (1536, 1152), (1200, 900)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI178MC/MM - 6.4MP camera
    "ZWO ASI178MC": CameraConfig(
        model="ZWO ASI178MC",
        resolution_x=3096,
        resolution_y=2080,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (3096, 2080), (2560, 1920), (1920, 1440),
            (1600, 1200), (1280, 960), (1024, 768),
            (800, 600), (640, 480), (320, 240)
        ],
        supported_resolutions_bin2=[
            (1544, 1040), (1280, 960), (960, 720),
            (800, 600), (640, 480), (512, 384), (400, 300)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI178MM": CameraConfig(
        model="ZWO ASI178MM",
        resolution_x=3096,
        resolution_y=2080,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (3096, 2080), (2560, 1920), (1920, 1440),
            (1600, 1200), (1280, 960), (1024, 768),
            (800, 600), (640, 480), (320, 240)
        ],
        supported_resolutions_bin2=[
            (1544, 1040), (1280, 960), (960, 720),
            (800, 600), (640, 480), (512, 384), (400, 300)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI178MM Pro": CameraConfig(
        model="ZWO ASI178MM Pro",
        resolution_x=3096,
        resolution_y=2080,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (3096, 2080), (2560, 1920), (1920, 1440),
            (1600, 1200), (1280, 960), (1024, 768),
            (800, 600), (640, 480), (320, 240)
        ],
        supported_resolutions_bin2=[
            (1544, 1040), (1280, 960), (960, 720),
            (800, 600), (640, 480), (512, 384), (400, 300)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI183MC/MM - 20.2MP camera
    "ZWO ASI183MC": CameraConfig(
        model="ZWO ASI183MC",
        resolution_x=5496,
        resolution_y=3672,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (4896, 3672), (4400, 3300), (4160, 3120),
            (3680, 2760), (3120, 2340), (2560, 1920),
            (1920, 1440), (1600, 1200), (1280, 960)
        ],
        supported_resolutions_bin2=[
            (2448, 1836), (2200, 1650), (2080, 1560),
            (1840, 1380), (1560, 1170), (1280, 960), (960, 720)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI183MM": CameraConfig(
        model="ZWO ASI183MM",
        resolution_x=5496,
        resolution_y=3672,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (4896, 3672), (4400, 3300), (4160, 3120),
            (3680, 2760), (3120, 2340), (2560, 1920),
            (1920, 1440), (1600, 1200), (1280, 960)
        ],
        supported_resolutions_bin2=[
            (2448, 1836), (2200, 1650), (2080, 1560),
            (1840, 1380), (1560, 1170), (1280, 960), (960, 720)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI183MC Pro": CameraConfig(
        model="ZWO ASI183MC Pro",
        resolution_x=5496,
        resolution_y=3672,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (4896, 3672), (4400, 3300), (4160, 3120),
            (3680, 2760), (3120, 2340), (2560, 1920),
            (1920, 1440), (1600, 1200), (1280, 960)
        ],
        supported_resolutions_bin2=[
            (2448, 1836), (2200, 1650), (2080, 1560),
            (1840, 1380), (1560, 1170), (1280, 960), (960, 720)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI183MM Pro": CameraConfig(
        model="ZWO ASI183MM Pro",
        resolution_x=5496,
        resolution_y=3672,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (4896, 3672), (4400, 3300), (4160, 3120),
            (3680, 2760), (3120, 2340), (2560, 1920),
            (1920, 1440), (1600, 1200), (1280, 960)
        ],
        supported_resolutions_bin2=[
            (2448, 1836), (2200, 1650), (2080, 1560),
            (1840, 1380), (1560, 1170), (1280, 960), (960, 720)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI485/585/678/715 - 8.3MP cameras (16:9 format)
    "ZWO ASI485MC": CameraConfig(
        model="ZWO ASI485MC",
        resolution_x=3840,
        resolution_y=2160,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (3840, 2160), (3240, 1822), (2880, 1620),
            (2400, 1350), (1800, 1012), (1536, 864),
            (1200, 674), (960, 540), (600, 336)
        ],
        supported_resolutions_bin2=[
            (1920, 1080), (1544, 870), (1280, 720),
            (960, 540), (800, 450), (640, 360), (512, 288)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI585MC": CameraConfig(
        model="ZWO ASI585MC",
        resolution_x=3840,
        resolution_y=2160,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (3840, 2160), (3240, 1822), (2880, 1620),
            (2400, 1350), (1800, 1012), (1536, 864),
            (1200, 674), (960, 540), (600, 336)
        ],
        supported_resolutions_bin2=[
            (1920, 1080), (1544, 870), (1280, 720),
            (960, 540), (800, 450), (640, 360), (512, 288)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI585MM": CameraConfig(
        model="ZWO ASI585MM",
        resolution_x=3840,
        resolution_y=2160,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (3840, 2160), (3240, 1822), (2880, 1620),
            (2400, 1350), (1800, 1012), (1536, 864),
            (1200, 674), (960, 540), (600, 336)
        ],
        supported_resolutions_bin2=[
            (1920, 1080), (1544, 870), (1280, 720),
            (960, 540), (800, 450), (640, 360), (512, 288)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI678MC": CameraConfig(
        model="ZWO ASI678MC",
        resolution_x=3840,
        resolution_y=2160,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (3840, 2160), (3240, 1822), (2880, 1620),
            (2400, 1350), (1800, 1012), (1536, 864),
            (1200, 674), (960, 540), (600, 336)
        ],
        supported_resolutions_bin2=[
            (1920, 1080), (1544, 870), (1280, 720),
            (960, 540), (800, 450), (640, 360), (512, 288)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI678MM": CameraConfig(
        model="ZWO ASI678MM",
        resolution_x=3840,
        resolution_y=2160,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (3840, 2160), (3240, 1822), (2880, 1620),
            (2400, 1350), (1800, 1012), (1536, 864),
            (1200, 674), (960, 540), (600, 336)
        ],
        supported_resolutions_bin2=[
            (1920, 1080), (1544, 870), (1280, 720),
            (960, 540), (800, 450), (640, 360), (512, 288)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI715MC": CameraConfig(
        model="ZWO ASI715MC",
        resolution_x=3840,
        resolution_y=2160,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (3840, 2160), (3240, 1822), (2880, 1620),
            (2400, 1350), (1800, 1012), (1536, 864),
            (1200, 674), (960, 540), (600, 336)
        ],
        supported_resolutions_bin2=[
            (1920, 1080), (1544, 870), (1280, 720),
            (960, 540), (800, 450), (640, 360), (512, 288)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI290/385/462 - 2.1MP cameras (16:9 format)
    "ZWO ASI290MM Mini": CameraConfig(
        model="ZWO ASI290MM Mini",
        resolution_x=1936,
        resolution_y=1096,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=12,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (1936, 1096), (1600, 900), (1280, 720),
            (1024, 580), (800, 460), (640, 360), (320, 180)
        ],
        supported_resolutions_bin2=[
            (968, 548), (800, 450), (640, 360),
            (512, 290), (400, 230), (320, 180), (160, 90)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI290MC": CameraConfig(
        model="ZWO ASI290MC",
        resolution_x=1936,
        resolution_y=1096,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=12,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (1936, 1096), (1600, 900), (1280, 720),
            (1024, 580), (800, 460), (640, 360), (320, 180)
        ],
        supported_resolutions_bin2=[
            (968, 548), (800, 450), (640, 360),
            (512, 290), (400, 230), (320, 180), (160, 90)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI290MM": CameraConfig(
        model="ZWO ASI290MM",
        resolution_x=1936,
        resolution_y=1096,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=12,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (1936, 1096), (1600, 900), (1280, 720),
            (1024, 580), (800, 460), (640, 360), (320, 180)
        ],
        supported_resolutions_bin2=[
            (968, 548), (800, 450), (640, 360),
            (512, 290), (400, 230), (320, 180), (160, 90)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI462MC": CameraConfig(
        model="ZWO ASI462MC",
        resolution_x=1936,
        resolution_y=1096,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=12,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (1936, 1096), (1600, 900), (1280, 720),
            (1024, 580), (800, 460), (640, 360), (320, 180)
        ],
        supported_resolutions_bin2=[
            (968, 548), (800, 450), (640, 360),
            (512, 290), (400, 230), (320, 180), (160, 90)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI385MC": CameraConfig(
        model="ZWO ASI385MC",
        resolution_x=1936,
        resolution_y=1096,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=12,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (1936, 1096), (1600, 900), (1280, 720),
            (1024, 580), (800, 460), (640, 360), (320, 180)
        ],
        supported_resolutions_bin2=[
            (968, 548), (800, 450), (640, 360),
            (512, 290), (400, 230), (320, 180), (160, 90)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI662MC/482MC - 2.1MP cameras (16:9 format)
    "ZWO ASI662MC": CameraConfig(
        model="ZWO ASI662MC",
        resolution_x=1920,
        resolution_y=1080,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=12,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (1920, 1080), (1600, 900), (1280, 720),
            (1024, 576), (800, 450), (640, 360), (320, 180)
        ],
        supported_resolutions_bin2=[
            (960, 540), (800, 450), (640, 360),
            (512, 288), (400, 225), (320, 180), (160, 90)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI482MC": CameraConfig(
        model="ZWO ASI482MC",
        resolution_x=1920,
        resolution_y=1080,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=12,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (1920, 1080), (1600, 900), (1280, 720),
            (1024, 576), (800, 450), (640, 360), (320, 180)
        ],
        supported_resolutions_bin2=[
            (960, 540), (800, 450), (640, 360),
            (512, 288), (400, 225), (320, 180), (160, 90)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI224MC - 1.2MP camera (4:3 format)
    "ZWO ASI224MC": CameraConfig(
        model="ZWO ASI224MC",
        resolution_x=1304,
        resolution_y=976,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=12,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (1304, 976), (1024, 768), (800, 600),
            (640, 480), (320, 240)
        ],
        supported_resolutions_bin2=[
            (652, 488), (512, 384), (400, 300),
            (320, 240), (160, 120)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI533MC/MM - 9.1MP camera (4:3 format)
    "ZWO ASI533MC": CameraConfig(
        model="ZWO ASI533MC",
        resolution_x=3008,
        resolution_y=3008,
        display_x=1350,
        display_y=1350,
        sensor_factor="1_1",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (3008, 3008), (2400, 2400), (2000, 2000),
            (1600, 1600), (1280, 1280), (1024, 1024),
            (800, 800), (640, 640), (400, 400)
        ],
        supported_resolutions_bin2=[
            (1504, 1504), (1200, 1200), (1000, 1000),
            (800, 800), (640, 640), (512, 512), (400, 400)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI533MM": CameraConfig(
        model="ZWO ASI533MM",
        resolution_x=3008,
        resolution_y=3008,
        display_x=1350,
        display_y=1350,
        sensor_factor="1_1",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (3008, 3008), (2400, 2400), (2000, 2000),
            (1600, 1600), (1280, 1280), (1024, 1024),
            (800, 800), (640, 640), (400, 400)
        ],
        supported_resolutions_bin2=[
            (1504, 1504), (1200, 1200), (1000, 1000),
            (800, 800), (640, 640), (512, 512), (400, 400)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI533MC Pro": CameraConfig(
        model="ZWO ASI533MC Pro",
        resolution_x=3008,
        resolution_y=3008,
        display_x=1350,
        display_y=1350,
        sensor_factor="1_1",
        sensor_bits=14,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (3008, 3008), (2400, 2400), (2000, 2000),
            (1600, 1600), (1280, 1280), (1024, 1024),
            (800, 800), (640, 640), (400, 400)
        ],
        supported_resolutions_bin2=[
            (1504, 1504), (1200, 1200), (1000, 1000),
            (800, 800), (640, 640), (512, 512), (400, 400)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI533MM Pro": CameraConfig(
        model="ZWO ASI533MM Pro",
        resolution_x=3008,
        resolution_y=3008,
        display_x=1350,
        display_y=1350,
        sensor_factor="1_1",
        sensor_bits=14,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (3008, 3008), (2400, 2400), (2000, 2000),
            (1600, 1600), (1280, 1280), (1024, 1024),
            (800, 800), (640, 640), (400, 400)
        ],
        supported_resolutions_bin2=[
            (1504, 1504), (1200, 1200), (1000, 1000),
            (800, 800), (640, 640), (512, 512), (400, 400)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI676MC - 2.0MP camera (16:9 format)
    "ZWO ASI676MC": CameraConfig(
        model="ZWO ASI676MC",
        resolution_x=1920,
        resolution_y=1080,
        display_x=1350,
        display_y=760,
        sensor_factor="16_9",
        sensor_bits=12,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (1920, 1080), (1600, 900), (1280, 720),
            (1024, 576), (800, 450), (640, 360), (320, 180)
        ],
        supported_resolutions_bin2=[
            (960, 540), (800, 450), (640, 360),
            (512, 288), (400, 225), (320, 180), (160, 90)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    # ZWO ASI120MC/MM - 1.2MP camera (4:3 format)
    "ZWO ASI120MC": CameraConfig(
        model="ZWO ASI120MC",
        resolution_x=1280,
        resolution_y=960,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=12,
        bayer_pattern="RGGB",
        supported_resolutions_bin1=[
            (1280, 960), (1024, 768), (800, 600),
            (640, 480), (320, 240)
        ],
        supported_resolutions_bin2=[
            (640, 480), (512, 384), (400, 300),
            (320, 240), (160, 120)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),

    "ZWO ASI120MM": CameraConfig(
        model="ZWO ASI120MM",
        resolution_x=1280,
        resolution_y=960,
        display_x=1350,
        display_y=1012,
        sensor_factor="4_3",
        sensor_bits=12,
        bayer_pattern="MONO",
        supported_resolutions_bin1=[
            (1280, 960), (1024, 768), (800, 600),
            (640, 480), (320, 240)
        ],
        supported_resolutions_bin2=[
            (640, 480), (512, 384), (400, 300),
            (320, 240), (160, 120)
        ],
        max_gain=600,
        usb_bandwidth=40
    ),
}


def get_camera_config(model_name: str) -> CameraConfig:
    """
    Get camera configuration for a specific camera model.

    This replaces the 1,500-line if-elif chain in the original init_camera()
    function. Instead of hardcoded conditionals, we do a simple dictionary lookup.

    Args:
        model_name: Camera model name (e.g., "ZWO ASI178MC")

    Returns:
        CameraConfig object for the specified camera

    Raises:
        ValueError: If camera model is not supported

    Example:
        >>> config = get_camera_config("ZWO ASI178MC")
        >>> print(f"{config.resolution_x}x{config.resolution_y}")
        3096x2080
    """
    if model_name not in CAMERA_MODELS:
        supported = ", ".join(sorted(CAMERA_MODELS.keys()))
        raise ValueError(
            f"Unsupported camera model: {model_name}\n"
            f"Supported models: {supported}"
        )

    return CAMERA_MODELS[model_name]


def get_supported_cameras() -> List[str]:
    """
    Get list of all supported camera models.

    Returns:
        List of camera model names, sorted alphabetically
    """
    return sorted(CAMERA_MODELS.keys())


def is_camera_supported(model_name: str) -> bool:
    """
    Check if a camera model is supported.

    Args:
        model_name: Camera model name to check

    Returns:
        True if camera is supported, False otherwise
    """
    return model_name in CAMERA_MODELS
