"""
Refresh Loop Class Module for JetsonSky

This module provides a class-based alternative to the exec() pattern used in refresh_loop.py.
The RefreshLoop class encapsulates the main display loop that handles:
- Camera acquisition and video processing
- Filter pipeline application
- AI-based object detection (craters, satellites)
- Display rendering and user interaction
- Video/image capture operations

Usage:
    from refresh_loop_class import RefreshLoop
    from state import ApplicationState

    app_state = ApplicationState()
    refresh_obj = RefreshLoop(globals(), app_state)
    refresh_obj.register()  # Injects refresh() function into main namespace

Architecture:
    The RefreshLoop class uses dependency injection to access the main script's
    global variables through a dictionary reference (self.g). This allows:
    1. IDE support (autocomplete, syntax checking, refactoring)
    2. Normal stack traces for debugging
    3. Gradual migration to state-based architecture
    4. State synchronization with app_state

Main flow in refresh():
    1. Camera mode (flag_camera_ok == True):
       - Capture frame from camera
       - HDR processing (if enabled)
       - Image stabilization (if enabled)
       - Apply filter pipeline
       - AI detection (craters, satellites)
       - Display and capture

    2. Video/Image mode (flag_camera_ok == False):
       - Load frame from video/image file
       - Apply debayering and preprocessing
       - Apply filter pipeline
       - Display and capture

Copyright Alain Paillou 2018-2025
"""

from __future__ import annotations
import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional, Any, Tuple
import numpy as np
import cv2

# Only import these for type checking to avoid circular imports
if TYPE_CHECKING:
    from state import ApplicationState
    import cupy as cp
    import PIL


class RefreshLoop:
    """
    Main display loop class handling frame acquisition, processing, and display.

    This class provides the refresh() method as a bound method that:
    1. Accesses main script variables through a globals dictionary
    2. Syncs state changes to app_state for eventual migration
    3. Can be registered into the main namespace for Tkinter's after() calls

    Attributes:
        g (Dict[str, Any]): Dictionary of globals from main script
        state (Optional[ApplicationState]): ApplicationState instance for state sync

    Example:
        >>> from refresh_loop_class import RefreshLoop
        >>> from state import ApplicationState
        >>>
        >>> app_state = ApplicationState()
        >>> refresh_obj = RefreshLoop(globals(), app_state)
        >>> refresh_obj.register()
        >>>
        >>> # Now Tkinter can use: fenetre_principale.after(4, refresh)
    """

    # Type annotations
    g: Dict[str, Any]
    state: Optional['ApplicationState']

    def __init__(self, main_globals: Dict[str, Any], app_state: Optional['ApplicationState'] = None) -> None:
        """
        Initialize RefreshLoop with globals reference and optional state.

        Args:
            main_globals: globals() dictionary from main script
            app_state: ApplicationState instance for state synchronization
        """
        self.g = main_globals
        self.state = app_state

    def register(self) -> None:
        """
        Register the refresh method into the main globals namespace.

        After calling this, Tkinter can reference refresh() directly.
        """
        self.g['refresh'] = self.refresh

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_cupy(self):
        """Get cupy module from globals."""
        return self.g.get('cp')

    def _get_pil(self):
        """Get PIL module from globals."""
        return self.g.get('PIL')

    def _update_fps(self, elapsed_time: float) -> None:
        """Update FPS tracking."""
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
            self.g['fpsQueue'].append(fps)
            if len(self.g['fpsQueue']) > 10:
                self.g['fpsQueue'].pop(0)
            self.g['curFPS'] = sum(self.g['fpsQueue']) / len(self.g['fpsQueue'])

    def _update_timestamp(self) -> None:
        """Update the current timestamp."""
        self.g['Date_hour_image'] = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')[:-3]

    def _camera_acquisition(self) -> Tuple[bool, Any, Any, Any]:
        """
        Wrapper for camera_acquisition function.
        Returns (success, res_rr1, res_gg1, res_bb1).
        """
        camera_acquisition = self.g.get('camera_acquisition')
        if camera_acquisition:
            return camera_acquisition()
        return False, None, None, None

    def _apply_filter_pipeline_color(self, res_rr1, res_gg1, res_bb1) -> None:
        """Apply color filter pipeline."""
        application_filtrage_color = self.g.get('application_filtrage_color')
        if application_filtrage_color:
            application_filtrage_color(res_rr1, res_gg1, res_bb1)

    def _apply_filter_pipeline_mono(self, res_bb1) -> None:
        """Apply mono filter pipeline."""
        application_filtrage_mono = self.g.get('application_filtrage_mono')
        if application_filtrage_mono:
            application_filtrage_mono(res_bb1)

    def _image_quality(self, img, method: str) -> float:
        """Calculate image quality."""
        Image_Quality = self.g.get('Image_Quality')
        if Image_Quality:
            return Image_Quality(img, method)
        return 0.0

    def _template_tracking(self, image, dim: int):
        """Apply template tracking for stabilization."""
        Template_tracking = self.g.get('Template_tracking')
        if Template_tracking:
            return Template_tracking(image, dim)
        return image

    def _opencv_color_debayer(self, image, type_debayer: int, flag_opencv_cuda: bool):
        """Apply OpenCV debayering."""
        opencv_color_debayer = self.g.get('opencv_color_debayer')
        if opencv_color_debayer:
            return opencv_color_debayer(image, type_debayer, flag_opencv_cuda)
        return image

    def _numpy_rgb_to_cupy_separate(self, image):
        """Convert numpy RGB image to separate CuPy R, G, B arrays."""
        func = self.g.get('numpy_RGBImage_2_cupy_separateRGB')
        if func:
            return func(image)
        return None, None, None

    def _cupy_separate_to_numpy_rgb(self, r, g, b):
        """Convert separate CuPy arrays to numpy RGB image."""
        func = self.g.get('cupy_separateRGB_2_numpy_RGBimage')
        if func:
            return func(r, g, b)
        return None

    def _start_mount(self) -> None:
        """Start mount communication."""
        start_mount = self.g.get('start_mount')
        if start_mount:
            start_mount()

    def _start_keyboard(self) -> None:
        """Start keyboard monitoring."""
        start_keyboard = self.g.get('start_keyboard')
        if start_keyboard:
            start_keyboard()

    # =========================================================================
    # Main Refresh Method
    # =========================================================================

    def refresh(self) -> None:
        """
        Main display loop - called repeatedly via Tkinter's after().

        This method handles:
        - Frame acquisition (camera or video)
        - HDR processing
        - Image stabilization
        - Filter pipeline application
        - AI detection (craters, satellites)
        - Display rendering
        - Capture operations
        """
        # Get commonly used globals as local references for speed
        g = self.g

        # Ensure image_traitee is defined
        if 'image_traitee' not in g or g.get('image_traitee') is None:
            g['image_traitee'] = None

        # Get CUDA context
        cupy_context = g.get('cupy_context')

        with cupy_context:
            if g.get('flag_camera_ok', False):
                self._refresh_camera_mode()
            else:
                self._refresh_video_mode()

    def _refresh_camera_mode(self) -> None:
        """Handle refresh in camera mode."""
        g = self.g
        cp = self._get_cupy()
        PIL = self._get_pil()

        # Start timing for FPS calculation
        total_start = cv2.getTickCount()

        # First start initialization
        if g.get('flag_premier_demarrage', True):
            g['flag_premier_demarrage'] = False
            self._start_mount()
            if g.get('Dev_system') == "Windows":
                self._start_keyboard()

        # Auto-exposure handling
        if g.get('flag_autoexposure_gain', False):
            camera = g.get('camera')
            asi = g.get('asi')
            val_gain = int(camera.get_control_value(asi.ASI_GAIN)[0])
            g['val_gain'] = val_gain
            g['echelle2'].set(val_gain)

        if g.get('flag_autoexposure_exposition', False):
            camera = g.get('camera')
            asi = g.get('asi')
            val_exposition = int(camera.get_control_value(asi.ASI_EXPOSURE)[0]) // 1000
            g['val_exposition'] = val_exposition
            g['echelle1'].set(val_exposition)

        # Main acquisition loop
        if not g.get('flag_stop_acquisition', False):
            # HDR mode
            if g.get('flag_HDR', False) and g.get('flag_filtrage_ON', False):
                ret_img = self._process_hdr_capture()
            else:
                ret_img, res_rr1, res_gg1, res_bb1 = self._camera_acquisition()
                if ret_img:
                    g['res_rr1'] = res_rr1
                    g['res_gg1'] = res_gg1
                    g['res_bb1'] = res_bb1

            self._update_timestamp()

            if ret_img:
                g['frame_number'] = g.get('frame_number', 0) + 1
                g['flag_GO'] = True

                # Best Frame Recording
                if g.get('flag_BFR', False) and not g.get('flag_image_mode', False):
                    self._process_bfr()

                # Best Frame Reference
                if (not g.get('flag_BFR', False) and g.get('flag_BFREF', False) and
                    g.get('flag_BFReference') == "BestFrame" and not g.get('flag_image_mode', False)):
                    self._process_bfref()

                # Apply filter pipeline
                if g.get('flag_filtrage_ON', False):
                    if g.get('flag_IsColor', True):
                        self._apply_filter_pipeline_color(
                            g.get('res_rr1'), g.get('res_gg1'), g.get('res_bb1'))
                    else:
                        self._apply_filter_pipeline_mono(g.get('res_bb1'))
                else:
                    g['image_traitee'] = g.get('res_bb1').get()
                    g['curTT'] = 0

                # Image Quality Estimation
                if g.get('flag_IQE', False):
                    self._process_iqe()

                # AI Detection
                if g.get('flag_AI_Craters', False) and g.get('flag_crater_model_loaded', False):
                    self._process_ai_craters()

                if g.get('flag_AI_Satellites', False) and g.get('flag_satellites_model_loaded', False):
                    self._process_ai_satellites()

                # Star Detection
                if g.get('flag_DETECT_STARS') == 1:
                    self._process_star_detection()

                # Satellite Tracking
                if g.get('flag_TRKSAT') == 1:
                    self._process_satellite_tracking()

                # Reference Image Capture
                if g.get('flag_capture_image_reference', False):
                    self._capture_reference_image()

                # Display processing
                self._process_display()

                # Calculate FPS
                total_stop = cv2.getTickCount()
                total_time = int((total_stop - total_start) / cv2.getTickFrequency() * 1000)
                if total_time > 0:
                    g['fpsQueue'].append(1000 / total_time)
                    if len(g['fpsQueue']) > 10:
                        g['fpsQueue'].pop(0)
                    g['curFPS'] = sum(g['fpsQueue']) / len(g['fpsQueue'])

                # Capture operations
                if g.get('flag_GO', True):
                    self._process_capture()

        # Schedule next refresh
        fenetre_principale = g.get('fenetre_principale')
        if fenetre_principale:
            if g.get('flag_image_mode', False):
                fenetre_principale.after(10, self.refresh)
            else:
                fenetre_principale.after(4, self.refresh)

    def _refresh_video_mode(self) -> None:
        """Handle refresh in video/image mode."""
        g = self.g
        cp = self._get_cupy()
        PIL = self._get_pil()

        # First start in video mode
        if g.get('flag_premier_demarrage', True):
            g['flag_premier_demarrage'] = False

        video = g.get('video')
        if video is None or not g.get('flag_image_video_loaded', False):
            # Schedule next refresh even if no video
            fenetre_principale = g.get('fenetre_principale')
            if fenetre_principale:
                fenetre_principale.after(100, self.refresh)
            return

        # Start timing for FPS calculation
        total_start = cv2.getTickCount()

        # Read frame from video
        ret_img = False

        if not g.get('flag_pause_video', False):
            if g.get('flag_SER_file', False):
                ret_img = self._read_ser_frame()
            else:
                ret_img = self._read_video_frame()

        if ret_img:
            g['frame_number'] = g.get('frame_number', 0) + 1

            # Apply filter pipeline
            if g.get('flag_filtrage_ON', False):
                if g.get('flag_IsColor', True):
                    self._apply_filter_pipeline_color(
                        g.get('res_rr1'), g.get('res_gg1'), g.get('res_bb1'))
                else:
                    self._apply_filter_pipeline_mono(g.get('res_bb1'))
            else:
                if g.get('flag_IsColor', True):
                    g['image_traitee'] = self._cupy_separate_to_numpy_rgb(
                        g.get('res_bb1'), g.get('res_gg1'), g.get('res_rr1'))
                else:
                    g['image_traitee'] = g.get('res_bb1').get()
                g['curTT'] = 0

            # Display processing
            self._process_display()

            # Calculate FPS
            total_stop = cv2.getTickCount()
            total_time = int((total_stop - total_start) / cv2.getTickFrequency() * 1000)
            if total_time > 0:
                g['fpsQueue'].append(1000 / total_time)
                if len(g['fpsQueue']) > 10:
                    g['fpsQueue'].pop(0)
                g['curFPS'] = sum(g['fpsQueue']) / len(g['fpsQueue'])

            # Capture operations
            self._process_capture()

        # Schedule next refresh
        fenetre_principale = g.get('fenetre_principale')
        if fenetre_principale:
            if g.get('flag_image_mode', False):
                fenetre_principale.after(10, self.refresh)
            else:
                fenetre_principale.after(4, self.refresh)

    # =========================================================================
    # Processing Helper Methods
    # =========================================================================

    def _process_hdr_capture(self) -> bool:
        """Process HDR capture mode. Returns True if successful."""
        g = self.g
        cp = self._get_cupy()

        try:
            camera = g.get('camera')
            asi = g.get('asi')
            timeoutexp = g.get('timeoutexp', 500)

            if not g.get('flag_16b', False):
                # 8-bit HDR mode
                image_brute1 = camera.capture_video_frame_RAW8_NUMPY(filename=None, timeout=timeoutexp)
                val_exposition = g['echelle1'].get()

                if g.get('flag_acq_rapide') == "Fast":
                    exposition = val_exposition // 2
                else:
                    exposition = val_exposition * 1000 // 2

                camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                time.sleep(0.1)
                camera.start_video_capture()
                image_brute2 = camera.capture_video_frame_RAW8_NUMPY(filename=None, timeout=timeoutexp)

                if g.get('flag_acq_rapide') == "Fast":
                    exposition = val_exposition // 4
                else:
                    exposition = val_exposition * 1000 // 4

                camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                time.sleep(0.1)
                camera.start_video_capture()
                image_brute3 = camera.capture_video_frame_RAW8_NUMPY(filename=None, timeout=timeoutexp)

                if g.get('flag_acq_rapide') == "Fast":
                    exposition = val_exposition // 6
                else:
                    exposition = val_exposition * 1000 // 6

                camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                time.sleep(0.1)
                camera.start_video_capture()
                image_brute4 = camera.capture_video_frame_RAW8_NUMPY(filename=None, timeout=timeoutexp)

                # Restore original exposure
                if g.get('flag_acq_rapide') == "Fast":
                    exposition = val_exposition
                else:
                    exposition = val_exposition * 1000
                camera.set_control_value(asi.ASI_EXPOSURE, exposition)

                # Merge HDR
                img_list = [image_brute1, image_brute2, image_brute3, image_brute4]
                merge_mertens = cv2.createMergeMertens()
                res_mertens = merge_mertens.process(img_list)
                image_brute = np.clip(res_mertens * 255, 0, 255).astype('uint8')
                image_brute = self._opencv_color_debayer(
                    image_brute, g.get('type_debayer', 0), g.get('flag_OpenCvCuda', False))

            else:
                # 16-bit HDR mode
                type_debayer = g.get('type_debayer', 0)
                if type_debayer != 0 and g.get('flag_IsColor', True):
                    mono_colour = "Colour"
                else:
                    mono_colour = "Mono"

                image_camera_base = camera.capture_video_frame_RAW16_CUPY(filename=None, timeout=timeoutexp)
                mode_HDR = g.get('mode_HDR', 'Mertens')
                TH_16B = g.get('TH_16B', 16)
                mode_BIN = g.get('mode_BIN', 1)
                flag_HB = g.get('flag_HB', False)

                if mode_HDR == "Mean":
                    # GPU-based HDR
                    res_r = cp.asarray(image_camera_base, dtype=cp.uint8)
                    height, width = image_camera_base.shape
                    nb_ThreadsX = g.get('nb_ThreadsX', 32)
                    nb_ThreadsY = g.get('nb_ThreadsY', 32)
                    nb_blocksX = (width // nb_ThreadsX) + 1
                    nb_blocksY = (height // nb_ThreadsY) + 1

                    HDR_compute_GPU = g.get('HDR_compute_GPU')
                    Hard_BIN = 1 if g.get('flag_HB', False) else 0

                    HDR_compute_GPU(
                        (nb_blocksX, nb_blocksY), (nb_ThreadsX, nb_ThreadsY),
                        (res_r, image_camera_base, np.intc(width), np.intc(height),
                         np.float32(TH_16B), np.intc(0), np.intc(mode_BIN), np.intc(Hard_BIN))
                    )
                    image_brute = res_r.get()
                    image_brute = cv2.cvtColor(image_brute, type_debayer)
                else:
                    HDR_compute = g.get('HDR_compute')
                    image_brute = HDR_compute(mono_colour, image_camera_base, mode_HDR,
                                             TH_16B, mode_BIN, flag_HB, type_debayer)

                if g.get('flag_noir_blanc') == 1 and g.get('flag_colour_camera', True):
                    if image_brute.ndim == 3:
                        image_brute = cv2.cvtColor(image_brute, cv2.COLOR_BGR2GRAY)

            self._update_timestamp()

            # Stabilization in HDR mode
            if g.get('flag_STAB', False):
                if image_brute.ndim == 3:
                    g['flag_IsColor'] = True
                    Dim = 3
                else:
                    g['flag_IsColor'] = False
                    Dim = 1
                image_brute = self._template_tracking(image_brute, Dim)

            # Convert to CuPy arrays
            if image_brute.ndim == 3:
                g['flag_IsColor'] = True
                res_bb1, res_gg1, res_rr1 = self._numpy_rgb_to_cupy_separate(image_brute)
                g['res_rr1'] = res_rr1
                g['res_gg1'] = res_gg1
                g['res_bb1'] = res_bb1
            else:
                g['flag_IsColor'] = False
                g['res_bb1'] = cp.asarray(image_brute)

            g['flag_image_disponible'] = True
            return True

        except Exception as error:
            g['nb_erreur'] = g.get('nb_erreur', 0) + 1
            print("An error occurred:", error)
            print("Capture error:", g['nb_erreur'])
            time.sleep(0.01)
            return False

    def _process_bfr(self) -> None:
        """Process Best Frame Recording."""
        g = self.g

        res_cam_x = g.get('res_cam_x', 0)
        res_cam_y = g.get('res_cam_y', 0)
        delta_tx = g.get('delta_tx', 0)
        delta_ty = g.get('delta_ty', 0)
        IQ_Method = g.get('IQ_Method', 'Sobel')
        val_BFR = g.get('val_BFR', 50)

        rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
        re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
        cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
        ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx

        if g.get('flag_IsColor', True):
            img_numpy_crop = self._cupy_separate_to_numpy_rgb(
                g.get('res_rr1'), g.get('res_gg1'), g.get('res_bb1'))
            crop_im_grey = cv2.cvtColor(img_numpy_crop, cv2.COLOR_BGR2GRAY)
            img_qual = self._image_quality(crop_im_grey, IQ_Method)
        else:
            img_numpy_crop = g.get('res_bb1').get()
            crop_Im = img_numpy_crop[rs:re, cs:ce]
            img_qual = self._image_quality(crop_Im, IQ_Method)

        max_qual = g.get('max_qual', 0)
        min_qual = g.get('min_qual', 10000)

        if img_qual > max_qual:
            g['max_qual'] = img_qual
            max_qual = img_qual
        if img_qual < min_qual:
            g['min_qual'] = img_qual
            min_qual = img_qual

        quality_threshold = min_qual + (max_qual - min_qual) * (val_BFR / 100)

        if img_qual < quality_threshold:
            g['flag_GO'] = False
            g['SFN'] = g.get('SFN', 0) + 1

        frame_number = g.get('frame_number', 1)
        SFN = g.get('SFN', 0)
        ratio = int((SFN / frame_number) * 1000) / 10
        texte = f"SFN : {SFN}   B/T : {ratio}  Thres : {int(quality_threshold*10)/10}  Qual : {int(img_qual*10)/10}             "
        g['labelInfo10'].config(text=texte)

    def _process_bfref(self) -> None:
        """Process Best Frame Reference."""
        g = self.g

        res_cam_x = g.get('res_cam_x', 0)
        res_cam_y = g.get('res_cam_y', 0)
        delta_tx = g.get('delta_tx', 0)
        delta_ty = g.get('delta_ty', 0)
        IQ_Method = g.get('IQ_Method', 'Sobel')

        rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
        re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
        cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
        ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx

        if g.get('flag_IsColor', True):
            img_numpy_crop = self._cupy_separate_to_numpy_rgb(
                g.get('res_rr1'), g.get('res_gg1'), g.get('res_bb1'))
            crop_Im = img_numpy_crop[rs:re, cs:ce]
            crop_im_grey = cv2.cvtColor(crop_Im, cv2.COLOR_BGR2GRAY)
            img_qual = self._image_quality(crop_im_grey, IQ_Method)

            if img_qual > g.get('max_qual', 0):
                g['max_qual'] = img_qual
                g['BFREF_image'] = self._cupy_separate_to_numpy_rgb(
                    g.get('res_bb1'), g.get('res_gg1'), g.get('res_rr1'))
                g['flag_BFREF_image'] = True
        else:
            img_numpy_crop = g.get('res_bb1').get()
            crop_Im = img_numpy_crop[rs:re, cs:ce]
            img_qual = self._image_quality(crop_Im, IQ_Method)

            if img_qual > g.get('max_qual', 0):
                g['max_qual'] = img_qual
                g['BFREF_image'] = g.get('res_bb1').get()
                g['flag_BFREF_image'] = True

    def _process_iqe(self) -> None:
        """Process Image Quality Estimation."""
        g = self.g

        res_cam_x = g.get('res_cam_x', 0)
        res_cam_y = g.get('res_cam_y', 0)
        IQ_Method = g.get('IQ_Method', 'Sobel')

        rs = res_cam_y // 2 - res_cam_y // 8
        re = res_cam_y // 2 + res_cam_y // 8
        cs = res_cam_x // 2 - res_cam_x // 8
        ce = res_cam_x // 2 + res_cam_x // 8

        image_traitee = g.get('image_traitee')

        if g.get('flag_IsColor', True):
            crop_Im = image_traitee[rs:re, cs:ce]
            crop_im_grey = cv2.cvtColor(crop_Im, cv2.COLOR_BGR2GRAY)
            quality_val = self._image_quality(crop_im_grey, IQ_Method)
        else:
            crop_Im = image_traitee[rs:re, cs:ce]
            quality_val = self._image_quality(crop_Im, IQ_Method)

        quality_pos = g.get('quality_pos', 1)
        g['quality'][quality_pos] = quality_val

        if quality_val > g.get('max_quality', 0):
            g['max_quality'] = quality_val

        quality_pos += 1
        if quality_pos > 255:
            quality_pos = 1
        g['quality_pos'] = quality_pos

    def _process_ai_craters(self) -> None:
        """Process AI crater detection."""
        g = self.g
        from collections import defaultdict

        image_traitee = g.get('image_traitee')

        if g.get('flag_IsColor', True):
            image_model = image_traitee
        else:
            image_model = cv2.merge((image_traitee, image_traitee, image_traitee))

        if g.get('flag_AI_Trace', False):
            model = g.get('model_craters_track')
            result_craters = model.track(image_model, device=0, half=True, conf=0.05,
                                         persist=True, verbose=False)
            result_craters2 = model(image_model, conf=0.05)[0]
        else:
            model = g.get('model_craters_predict')
            result_craters = model.predict(image_model, device=0, max_det=100,
                                          half=True, verbose=False)
            result_craters2 = model(image_model, conf=0.05)[0]

        # Extract detection results
        boxes_crater = result_craters2.boxes.xywh.cpu()
        bboxes_crater = np.array(result_craters2.boxes.xyxy.cpu(), dtype="int")
        classes_crater = np.array(result_craters2.boxes.cls.cpu(), dtype="int")
        confidence_crater = result_craters2.boxes.conf.cpu()

        if g.get('flag_AI_Trace', False):
            track_crater_ids = (result_craters2.boxes.id.int().cpu().tolist()
                               if result_craters2.boxes.id is not None else None)
        else:
            track_crater_ids = None

        # Draw detections on image
        if track_crater_ids:
            self._draw_crater_detections(boxes_crater, classes_crater, track_crater_ids)

    def _process_ai_satellites(self) -> None:
        """Process AI satellite detection."""
        g = self.g

        image_traitee = g.get('image_traitee')

        if g.get('flag_IsColor', True):
            image_model = image_traitee
        else:
            image_model = cv2.merge((image_traitee, image_traitee, image_traitee))

        if g.get('flag_AI_Trace', False):
            model = g.get('model_satellites_track')
            result_sats = model.track(image_model, device=0, half=True, conf=0.05,
                                      persist=True, verbose=False)
            result_sats2 = model(image_model, conf=0.05)[0]
        else:
            model = g.get('model_satellites_predict')
            result_sats = model.predict(image_model, device=0, max_det=100,
                                        half=True, verbose=False)
            result_sats2 = model(image_model, conf=0.05)[0]

        # Draw detections
        if g.get('flag_AI_Trace', False) and result_sats2.boxes.id is not None:
            track_sat_ids = result_sats2.boxes.id.int().cpu().tolist()
            boxes_sat = result_sats2.boxes.xywh.cpu()
            classes_sat = np.array(result_sats2.boxes.cls.cpu(), dtype="int")
            self._draw_satellite_detections(boxes_sat, classes_sat, track_sat_ids)

    def _draw_crater_detections(self, boxes, classes, track_ids) -> None:
        """Draw crater detection boxes and traces on image."""
        g = self.g
        model = g.get('model_craters_track')

        for cls, box, track_id in zip(classes, boxes, track_ids):
            x, y, w1, h1 = box
            object_name = model.names[cls]

            if object_name == "Small crater":
                BOX_COLOUR = (0, 255, 255)
            elif object_name == "Crater":
                BOX_COLOUR = (0, 255, 0)
            else:
                BOX_COLOUR = (255, 0, 0)

            # Draw on image_traitee
            image_traitee = g.get('image_traitee')
            cv2.circle(image_traitee, (int(x), int(y)), int(w1 // 2), BOX_COLOUR, 2)

            # Update tracking history
            track_crater_history = g.get('track_crater_history', {})
            track_crater_history[track_id].append((int(x), int(y)))

            # Draw trace
            if len(track_crater_history[track_id]) > 1:
                points = track_crater_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(image_traitee, points[i-1], points[i], BOX_COLOUR, 2)

    def _draw_satellite_detections(self, boxes, classes, track_ids) -> None:
        """Draw satellite detection boxes and traces on image."""
        g = self.g
        model = g.get('model_satellites_track')

        for cls, box, track_id in zip(classes, boxes, track_ids):
            x, y, w1, h1 = box
            BOX_COLOUR = (0, 0, 255)

            image_traitee = g.get('image_traitee')
            cv2.rectangle(image_traitee, (int(x - w1/2), int(y - h1/2)),
                         (int(x + w1/2), int(y + h1/2)), BOX_COLOUR, 2)

            track_satellite_history = g.get('track_satellite_history', {})
            track_satellite_history[track_id].append((int(x), int(y)))

            if len(track_satellite_history[track_id]) > 1:
                points = track_satellite_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(image_traitee, points[i-1], points[i], BOX_COLOUR, 2)

    def _process_star_detection(self) -> None:
        """Process star detection."""
        g = self.g
        # Star detection uses the Star_detection function
        Star_detection = g.get('Star_detection')
        if Star_detection:
            Star_detection()

    def _process_satellite_tracking(self) -> None:
        """Process satellite tracking."""
        g = self.g
        Satellite_tracking = g.get('Satellite_tracking')
        if Satellite_tracking:
            Satellite_tracking()

    def _capture_reference_image(self) -> None:
        """Capture reference image for subtraction."""
        g = self.g

        if g.get('flag_IsColor', True):
            g['Image_Reference'] = self._cupy_separate_to_numpy_rgb(
                g.get('res_bb1'), g.get('res_gg1'), g.get('res_rr1'))
        else:
            g['Image_Reference'] = g.get('res_bb1').get()

        g['flag_image_reference_OK'] = True
        g['flag_capture_image_reference'] = False

    def _process_display(self) -> None:
        """Process display output."""
        g = self.g
        PIL = self._get_pil()

        image_traitee = g.get('image_traitee')
        if image_traitee is None:
            return

        cadre_image = g.get('cadre_image')
        if cadre_image is None:
            return

        res_cam_x = g.get('res_cam_x', 0)
        res_cam_y = g.get('res_cam_y', 0)
        cam_displ_x = g.get('cam_displ_x', 1350)
        cam_displ_y = g.get('cam_displ_y', 1012)
        fact_s = g.get('fact_s', 1.0)
        flag_full_res = g.get('flag_full_res', 0)

        # Resize or crop for display
        if res_cam_x > int(cam_displ_x * fact_s) and flag_full_res == 0:
            # Resize to fit display
            image_resized = cv2.resize(image_traitee,
                                       (int(cam_displ_x * fact_s), int(cam_displ_y * fact_s)))
            cadre_image.im = PIL.Image.fromarray(image_resized)
        elif flag_full_res == 1:
            # Full resolution with zoom
            delta_zx = g.get('delta_zx', 0)
            delta_zy = g.get('delta_zy', 0)

            rs = max(0, (res_cam_y - int(cam_displ_y * fact_s)) // 2 + delta_zy)
            re = min(res_cam_y, (res_cam_y + int(cam_displ_y * fact_s)) // 2 + delta_zy)
            cs = max(0, (res_cam_x - int(cam_displ_x * fact_s)) // 2 + delta_zx)
            ce = min(res_cam_x, (res_cam_x + int(cam_displ_x * fact_s)) // 2 + delta_zx)

            image_crop = image_traitee[rs:re, cs:ce]
            cadre_image.im = PIL.Image.fromarray(image_crop)
        else:
            cadre_image.im = PIL.Image.fromarray(image_traitee)

        # Draw overlays
        self._draw_overlays(cadre_image)

        # Update Tkinter canvas
        cadre_image.photo = PIL.ImageTk.PhotoImage(cadre_image.im)
        cadre_image.create_image(cam_displ_x * fact_s / 2, cam_displ_y * fact_s / 2,
                                 image=cadre_image.photo)

        # Update treatment time and FPS display
        curTT = g.get('curTT', 0)
        curFPS = g.get('curFPS', 0)
        labelInfo2 = g.get('labelInfo2')
        if labelInfo2 is not None:
            if g.get('flag_image_mode', False):
                labelInfo2.config(text=f"{curTT} ms      ")
            else:
                labelInfo2.config(text=f"{curTT} ms   FPS : {int(curFPS*10)/10}    ")

    def _draw_overlays(self, cadre_image) -> None:
        """Draw overlay graphics on the image."""
        g = self.g
        PIL = self._get_pil()

        cam_displ_x = g.get('cam_displ_x', 1350)
        cam_displ_y = g.get('cam_displ_y', 1012)
        fact_s = g.get('fact_s', 1.0)

        # Crosshair
        if g.get('flag_cross', False):
            draw = PIL.ImageDraw.Draw(cadre_image.im)
            SX, SY = cadre_image.im.size
            draw.line(((SX/2 - 100, SY/2), (SX/2 + 100, SY/2)), fill="red", width=1)
            draw.line(((SX/2, SY/2 - 100), (SX/2, SY/2 + 100)), fill="red", width=1)

        # Histogram
        if g.get('flag_HST') == 1 and g.get('flag_IsColor', True):
            r, green, b = cadre_image.im.split()
            hst_r = r.histogram()
            hst_g = green.histogram()
            hst_b = b.histogram()
            histo = PIL.ImageDraw.Draw(cadre_image.im)
            for x in range(1, 256):
                histo.line(((x*3, cam_displ_y), (x*3, cam_displ_y - hst_r[x]/100)), fill="red")
                histo.line(((x*3+1, cam_displ_y), (x*3+1, cam_displ_y - hst_g[x]/100)), fill="green")
                histo.line(((x*3+2, cam_displ_y), (x*3+2, cam_displ_y - hst_b[x]/100)), fill="blue")
            histo.line(((256*3, cam_displ_y), (256*3, cam_displ_y - 256*3)), fill="red", width=3)
            histo.line(((1, cam_displ_y - 256*3), (256*3, cam_displ_y - 256*3)), fill="red", width=3)

        # Image Quality display
        if g.get('flag_IQE', False):
            quality = g.get('quality', {})
            max_quality = g.get('max_quality', 1)
            transform = PIL.ImageDraw.Draw(cadre_image.im)
            for x in range(2, 256):
                y2 = int((quality.get(x, 0) / max_quality) * 400)
                y1 = int((quality.get(x-1, 0) / max_quality) * 400)
                transform.line((((x-1)*3, cam_displ_y - y1), (x*3, cam_displ_y - y2)),
                              fill="red", width=2)
            transform.line(((256*3, cam_displ_y), (256*3, cam_displ_y - 256*3)), fill="blue", width=3)
            transform.line(((1, cam_displ_y - 256*3), (256*3, cam_displ_y - 256*3)), fill="blue", width=3)

        # Transfer function display
        if g.get('flag_TRSF') == 1 and g.get('flag_IsColor', True):
            trsf_r = g.get('trsf_r', [0] * 256)
            trsf_g = g.get('trsf_g', [0] * 256)
            trsf_b = g.get('trsf_b', [0] * 256)
            transform = PIL.ImageDraw.Draw(cadre_image.im)
            for x in range(2, 256):
                transform.line((((x-1)*3, cam_displ_y - trsf_r[x-1]*3),
                               (x*3, cam_displ_y - trsf_r[x]*3)), fill="red", width=2)
                transform.line((((x-1)*3, cam_displ_y - trsf_g[x-1]*3),
                               (x*3, cam_displ_y - trsf_g[x]*3)), fill="green", width=2)
                transform.line((((x-1)*3, cam_displ_y - trsf_b[x-1]*3),
                               (x*3, cam_displ_y - trsf_b[x]*3)), fill="blue", width=2)
            transform.line(((256*3, cam_displ_y), (256*3, cam_displ_y - 256*3)), fill="red", width=3)
            transform.line(((1, cam_displ_y - 256*3), (256*3, cam_displ_y - 256*3)), fill="red", width=3)

        # BFR display
        if g.get('flag_BFR', False) and not g.get('flag_image_mode', False):
            max_qual = g.get('max_qual', 1)
            min_qual = g.get('min_qual', 0)
            img_qual = g.get('img_qual', 0)
            quality_threshold = g.get('quality_threshold', 0)

            transform = PIL.ImageDraw.Draw(cadre_image.im)
            mul_par = (cam_displ_x * fact_s - 600) / max(max_qual, 1)

            transform.line(((0, cam_displ_y * fact_s - 50),
                           (int(min_qual * mul_par), cam_displ_y * fact_s - 50)), fill="red", width=4)
            transform.line(((0, cam_displ_y * fact_s - 110),
                           (int(max_qual * mul_par), cam_displ_y * fact_s - 110)), fill="blue", width=4)
            transform.line(((0, cam_displ_y * fact_s - 80),
                           (int(img_qual * mul_par), cam_displ_y * fact_s - 80)), fill="yellow", width=4)
            transform.line(((int(quality_threshold * mul_par), cam_displ_y * fact_s - 55),
                           (int(quality_threshold * mul_par), cam_displ_y * fact_s - 105)),
                          fill="green", width=6)

        # Skip frame indicator
        if not g.get('flag_GO', True):
            transform = PIL.ImageDraw.Draw(cadre_image.im)
            transform.line(((0, 0), (cam_displ_x * fact_s, cam_displ_y * fact_s)), fill="red", width=2)
            transform.line(((0, cam_displ_y * fact_s), (cam_displ_x * fact_s, 0)), fill="red", width=2)

    def _process_capture(self) -> None:
        """Process image/video capture."""
        g = self.g

        # Image capture
        if g.get('flag_cap_pic', False):
            self._capture_image()

        # Video capture
        if g.get('flag_cap_video', False):
            self._capture_video_frame()

    def _capture_image(self) -> None:
        """Capture single image."""
        g = self.g
        image_traitee = g.get('image_traitee')
        if image_traitee is None:
            return

        Date_hour_image = g.get('Date_hour_image', '')
        filename = f"capture_{Date_hour_image.replace('/', '_').replace(':', '_').replace(' ', '_')}.png"

        # Save image
        if g.get('flag_IsColor', True):
            cv2.imwrite(filename, cv2.cvtColor(image_traitee, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(filename, image_traitee)

        g['flag_cap_pic'] = False

    def _capture_video_frame(self) -> None:
        """Capture video frame."""
        g = self.g
        # Video capture is handled by the video_capture thread
        pass

    def _read_ser_frame(self) -> bool:
        """Read frame from SER file. Returns True if successful."""
        g = self.g
        cp = self._get_cupy()

        video = g.get('video')
        if video is None:
            return False

        try:
            frame_position = g.get('video_frame_position', 0)
            video.setCurrentPosition(frame_position)
            frame = video.getFrame()

            if frame is not None:
                # Process SER frame
                SER_depth = g.get('SER_depth', 8)
                type_debayer = g.get('type_debayer', 0)

                if SER_depth > 8:
                    # 16-bit SER file
                    frame = (frame / 256).astype(np.uint8)

                # Debayer if needed
                if type_debayer != 0:
                    frame = cv2.cvtColor(frame, type_debayer)

                # Convert to CuPy
                if frame.ndim == 3:
                    g['flag_IsColor'] = True
                    res_bb1, res_gg1, res_rr1 = self._numpy_rgb_to_cupy_separate(frame)
                    g['res_rr1'] = res_rr1
                    g['res_gg1'] = res_gg1
                    g['res_bb1'] = res_bb1
                else:
                    g['flag_IsColor'] = False
                    g['res_bb1'] = cp.asarray(frame)

                g['video_frame_position'] = frame_position + 1
                return True
        except Exception as e:
            print(f"SER read error: {e}")

        return False

    def _read_video_frame(self) -> bool:
        """Read frame from video file. Returns True if successful."""
        g = self.g
        cp = self._get_cupy()

        video = g.get('video')
        if video is None:
            return False

        ret, frame = video.read()
        if not ret:
            return False

        type_debayer = g.get('type_debayer', 0)

        # Debayer if needed
        if type_debayer != 0 and frame.ndim == 2:
            frame = cv2.cvtColor(frame, type_debayer)

        # Convert to CuPy
        if frame.ndim == 3:
            g['flag_IsColor'] = True
            res_bb1, res_gg1, res_rr1 = self._numpy_rgb_to_cupy_separate(frame)
            g['res_rr1'] = res_rr1
            g['res_gg1'] = res_gg1
            g['res_bb1'] = res_bb1
        else:
            g['flag_IsColor'] = False
            g['res_bb1'] = cp.asarray(frame)

        g['video_frame_position'] = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        return True


# =========================================================================
# Module-level function for backward compatibility
# =========================================================================

def init_refresh_loop(main_globals: dict, app_state=None) -> RefreshLoop:
    """
    Initialize RefreshLoop and register it into the main namespace.

    This function creates a RefreshLoop instance and registers the refresh()
    method into the main globals namespace.

    Args:
        main_globals: globals() dictionary from main script
        app_state: ApplicationState instance (optional)

    Returns:
        RefreshLoop instance

    Usage:
        from refresh_loop_class import init_refresh_loop

        refresh_obj = init_refresh_loop(globals(), app_state)
    """
    refresh_loop = RefreshLoop(main_globals, app_state)
    refresh_loop.register()
    return refresh_loop
