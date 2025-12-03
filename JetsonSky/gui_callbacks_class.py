"""
GUI Callbacks Class Module for JetsonSky

This module provides a class-based alternative to the exec() pattern used in gui_callbacks.py.
The GUICallbacks class encapsulates all GUI callbacks as methods that access globals through
a dictionary reference and sync state to app_state.

Usage:
    from gui_callbacks_class import GUICallbacks
    from state import ApplicationState

    app_state = ApplicationState()
    callbacks = GUICallbacks(globals(), app_state)
    callbacks.register_all()  # Injects callback functions into main namespace

Architecture:
    This class bridges the legacy globals-based approach with the new state-based architecture:

    1. Callbacks read/write to globals dict (self.g) for backward compatibility
    2. After modifying globals, callbacks sync the values to app_state
    3. Eventually, globals can be phased out and callbacks will work directly with state

State Synchronization:
    Each callback that modifies state follows this pattern:
    - Update the global variable via self.g['variable_name'] = value
    - Sync to app_state: self.state.filter_params.field = value (if state exists)

Copyright Alain Paillou 2018-2025
"""

from __future__ import annotations
import time
import logging
import traceback
from typing import TYPE_CHECKING, Dict, Optional, Any, Callable
from functools import wraps
import numpy as np
import cv2
from collections import defaultdict

if TYPE_CHECKING:
    from state import ApplicationState

# Configure module logger
logger = logging.getLogger(__name__)


def safe_callback(func: Callable) -> Callable:
    """
    Decorator to wrap callbacks with error handling.
    Catches exceptions and logs them without crashing the GUI.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            logger.error(f"KeyError in {func.__name__}: Missing key {e}")
            logger.debug(traceback.format_exc())
        except AttributeError as e:
            logger.error(f"AttributeError in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())
        return None
    return wrapper


class GUICallbacks:
    """
    Class containing all GUI callback methods.

    This class provides callbacks as methods that:
    1. Access main script variables through a globals dictionary
    2. Sync state changes to app_state for eventual migration away from globals
    3. Can be bound directly to Tkinter widgets via command= parameter

    Attributes:
        g (Dict[str, Any]): Dictionary of globals from main script
        state (Optional[ApplicationState]): ApplicationState instance for state synchronization

    Example:
        >>> from gui_callbacks_class import GUICallbacks
        >>> from state import ApplicationState
        >>>
        >>> app_state = ApplicationState()
        >>> callbacks = GUICallbacks(globals(), app_state)
        >>> callbacks.register_all()
        >>>
        >>> # Now widgets can use callbacks
        >>> button = Button(root, command=callbacks.commande_filtrage_ON)
    """

    # Type annotations for instance attributes
    g: Dict[str, Any]
    state: Optional['ApplicationState']

    def __init__(self, main_globals: Dict[str, Any], app_state: Optional['ApplicationState'] = None) -> None:
        """
        Initialize the GUICallbacks with globals reference and optional state.

        Args:
            main_globals: globals() dictionary from main script containing all
                         IntVars, widget references, and callback functions
            app_state: ApplicationState instance for state synchronization.
                      If None, state sync is skipped (legacy mode).
        """
        self.g = main_globals
        self.state = app_state
        
        # Acquisition mode configurations: (mode_name, exp_min, exp_max, exp_delta, exp_interval, is_microseconds)
        self._acq_modes = {
            'Fast': ('Fast', 100, 10000, 100, 2000, True),      # microseconds
            'MedF': ('MedF', 1, 400, 1, 50, False),              # milliseconds
            'MedS': ('MedS', 1, 1000, 1, 200, False),            # milliseconds
            'Slow': ('Slow', 500, 20000, 100, 5000, False),      # milliseconds
        }

    # =========================================================================
    # Safe Accessor Methods
    # =========================================================================

    def _get(self, key: str, default: Any = None) -> Any:
        """Safely get a value from globals with a default."""
        return self.g.get(key, default)

    def _get_var(self, key: str) -> Optional[int]:
        """Safely get IntVar value from globals, returns None if not found or error."""
        try:
            var = self.g.get(key)
            if var is not None and hasattr(var, 'get'):
                return var.get()
        except Exception as e:
            logger.warning(f"Error getting IntVar '{key}': {e}")
        return None

    def _set(self, key: str, value: Any) -> None:
        """Safely set a value in globals."""
        self.g[key] = value

    def _get_widget(self, key: str) -> Optional[Any]:
        """Safely get a widget from globals."""
        widget = self.g.get(key)
        if widget is None:
            logger.warning(f"Widget '{key}' not found in globals")
        return widget

    def _call_func(self, func_key: str, *args, **kwargs) -> Any:
        """Safely call a function stored in globals."""
        func = self.g.get(func_key)
        if func is not None and callable(func):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error calling '{func_key}': {e}")
        elif func is None:
            logger.warning(f"Function '{func_key}' not found in globals")
        return None

    # =========================================================================
    # Helper Methods for Reducing Repetition
    # =========================================================================

    def _toggle_flag(self, choix_key: str, flag_key: str, state_attr: str = None) -> None:
        """
        Generic toggle handler for checkbox callbacks.
        
        Args:
            choix_key: The key for the IntVar checkbox in globals (e.g., 'choix_GR')
            flag_key: The key for the flag in globals (e.g., 'flag_GR')
            state_attr: Optional attribute name on state.filter_params to sync
        """
        choix_val = self._get_var(choix_key)
        if choix_val is not None:
            self.g[flag_key] = choix_val != 0
        else:
            # Fallback: toggle current value
            self.g[flag_key] = not self.g.get(flag_key, False)
            logger.debug(f"Toggle fallback for '{choix_key}' -> '{flag_key}'")
        
        # Sync to app_state
        if state_attr and self.state is not None:
            try:
                if self.state.filter_params is not None:
                    setattr(self.state.filter_params, state_attr, self.g[flag_key])
            except AttributeError as e:
                logger.warning(f"Cannot sync state_attr '{state_attr}': {e}")

    def _set_acquisition_mode(self, mode: str) -> None:
        """
        Generic acquisition mode setter.
        
        Args:
            mode: One of 'Fast', 'MedF', 'MedS', 'Slow'
        """
        if not self._get('flag_camera_ok', False):
            logger.debug(f"Skipping acquisition mode '{mode}': camera not available")
            return
        
        if mode not in self._acq_modes:
            logger.error(f"Unknown acquisition mode: {mode}")
            return
            
        mode_name, exp_min, exp_max, exp_delta, exp_interval, is_microseconds = self._acq_modes[mode]
        
        try:
            self._set('flag_acq_rapide', mode_name)
            self._set('flag_stop_acquisition', True)
            time.sleep(1)
            
            # Set exposure parameters
            self._set('exp_min', exp_min)
            self._set('exp_max', exp_max)
            self._set('exp_delta', exp_delta)
            self._set('exp_interval', exp_interval)
            self._set('val_exposition', exp_min)
            
            # Recreate scale widget
            Scale = self._get('Scale')
            cadre = self._get('cadre')
            
            if Scale is None or cadre is None:
                logger.error("Cannot create scale widget: Scale or cadre not found")
                self._set('flag_stop_acquisition', False)
                return
                
            echelle1 = Scale(
                cadre, from_=exp_min, to=exp_max,
                command=self.valeur_exposition, orient=self._get('HORIZONTAL'),
                length=330, width=7, resolution=exp_delta,
                label="", showvalue=1, tickinterval=exp_interval,
                sliderlength=20
            )
            echelle1.set(exp_min)
            echelle1.place(anchor="w", x=self._get('xS1', 0) + self._get('delta_s', 0), y=self._get('yS1', 0))
            self._set('echelle1', echelle1)
            
            # Calculate exposition in microseconds
            if is_microseconds:
                exposition = exp_min  # Already in microseconds
                timeoutexp = (exposition / 1000) * 2 + 500
            else:
                exposition = exp_min * 1000  # Convert ms to us
                timeoutexp = exp_min * 2 + 500
            
            self._set('exposition', exposition)
            self._set('timeoutexp', timeoutexp)
            
            # Configure camera
            camera = self._get('camera')
            asi = self._get('asi')
            
            if camera is None or asi is None:
                logger.error("Cannot configure camera: camera or asi module not found")
                self._set('flag_stop_acquisition', False)
                return
                
            camera.set_control_value(asi.ASI_EXPOSURE, exposition)
            camera.default_timeout = timeoutexp
            
            # Set speed mode
            if self._get('flag_read_speed') == "Slow":
                camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, 0)
            else:
                camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, 1)
            
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error setting acquisition mode '{mode}': {e}")
        finally:
            self._set('flag_stop_acquisition', False)

    def _set_slider_value(self, slider_key: str, val_key: str, multiplier: float = 1.0) -> None:
        """
        Generic slider value handler.
        
        Args:
            slider_key: Key for the slider widget in globals
            val_key: Key for the value variable in globals
            multiplier: Optional multiplier for the value
        """
        slider = self._get(slider_key)
        if slider is not None:
            try:
                self._set(val_key, slider.get() * multiplier)
            except Exception as e:
                logger.warning(f"Error getting slider value from '{slider_key}': {e}")
        else:
            logger.debug(f"Slider '{slider_key}' not found")

    def register_all(self) -> None:
        """
        Register all callback methods into the main globals namespace.

        This allows widgets to reference callbacks by name (e.g., command=mode_Lineaire).
        After calling this method, the following callbacks are available in globals:

        Mode callbacks:
            mode_Lineaire, mode_Gauss, mode_Stars, HDR_Mertens, HDR_Median, HDR_Mean,
            mode_gradient, mode_vignetting

        Acquisition mode:
            mode_acq_rapide, mode_acq_mediumF, mode_acq_mediumS, mode_acq_lente

        Camera control:
            valeur_exposition, valeur_gain, choix_BIN1, choix_BIN2,
            choix_resolution_camera, choix_USB, choix_ASI_GAMMA, etc.

        Filter toggles:
            commande_filtrage_ON, commande_AANR, commande_3FNR, commande_NLM2, etc.

        And many more - see method body for complete list.
        """
        # Mode callbacks
        self.g['mode_Lineaire'] = self.mode_Lineaire
        self.g['mode_Gauss'] = self.mode_Gauss
        self.g['mode_Stars'] = self.mode_Stars
        self.g['HDR_Mertens'] = self.HDR_Mertens
        self.g['HDR_Median'] = self.HDR_Median
        self.g['HDR_Mean'] = self.HDR_Mean
        self.g['mode_gradient'] = self.mode_gradient
        self.g['mode_vignetting'] = self.mode_vignetting

        # Acquisition mode callbacks
        self.g['mode_acq_rapide'] = self.mode_acq_rapide
        self.g['mode_acq_mediumF'] = self.mode_acq_mediumF
        self.g['mode_acq_mediumS'] = self.mode_acq_mediumS
        self.g['mode_acq_lente'] = self.mode_acq_lente

        # Camera control callbacks
        self.g['valeur_exposition'] = self.valeur_exposition
        self.g['valeur_gain'] = self.valeur_gain
        self.g['choix_BIN1'] = self.choix_BIN1
        self.g['choix_BIN2'] = self.choix_BIN2
        self.g['choix_resolution_camera'] = self.choix_resolution_camera
        self.g['choix_USB'] = self.choix_USB
        self.g['choix_ASI_GAMMA'] = self.choix_ASI_GAMMA
        self.g['choix_read_speed_fast'] = self.choix_read_speed_fast
        self.g['choix_read_speed_slow'] = self.choix_read_speed_slow
        self.g['choix_w_red'] = self.choix_w_red
        self.g['choix_w_blue'] = self.choix_w_blue

        # Filter toggle callbacks
        self.g['commande_img_Neg'] = self.commande_img_Neg
        self.g['commande_TIP'] = self.commande_TIP
        self.g['commande_SAT'] = self.commande_SAT
        self.g['commande_SAT2PASS'] = self.commande_SAT2PASS
        self.g['commande_mount'] = self.commande_mount
        self.g['commande_cross'] = self.commande_cross
        self.g['commande_mode_full_res'] = self.commande_mode_full_res
        self.g['commande_sharpen_soft1'] = self.commande_sharpen_soft1
        self.g['commande_sharpen_soft2'] = self.commande_sharpen_soft2
        self.g['commande_NLM2'] = self.commande_NLM2
        self.g['commande_denoise_Paillou'] = self.commande_denoise_Paillou
        self.g['commande_denoise_Paillou2'] = self.commande_denoise_Paillou2
        self.g['commande_HST'] = self.commande_HST
        self.g['commande_TRSF'] = self.commande_TRSF
        self.g['commande_TRGS'] = self.commande_TRGS
        self.g['commande_TRCLL'] = self.commande_TRCLL
        self.g['commande_DEMO'] = self.commande_DEMO
        self.g['commande_STAB'] = self.commande_STAB
        self.g['commande_DETECT_STARS'] = self.commande_DETECT_STARS
        self.g['commande_AANR'] = self.commande_AANR
        self.g['commande_AANRB'] = self.commande_AANRB
        self.g['commande_3FNR'] = self.commande_3FNR
        self.g['commande_3FNR2'] = self.commande_3FNR2
        self.g['commande_3FNRB'] = self.commande_3FNRB
        self.g['commande_3FNR2B'] = self.commande_3FNR2B
        self.g['commande_GR'] = self.commande_GR
        self.g['commande_ghost_reducer'] = self.commande_ghost_reducer
        self.g['commande_KNN'] = self.commande_KNN
        self.g['commande_reduce_variation'] = self.commande_reduce_variation
        self.g['commande_reduce_variation_post_treatment'] = self.commande_reduce_variation_post_treatment
        self.g['commande_histo_equalize'] = self.commande_histo_equalize
        self.g['commande_histogram_equalize2'] = self.commande_histo_equalize  # Alias for widget
        self.g['commande_histo_stretch'] = self.commande_histo_stretch
        self.g['commande_histogram_stretch'] = self.commande_histo_stretch  # Alias for widget
        self.g['commande_histo_phitheta'] = self.commande_histo_phitheta
        self.g['commande_histogram_phitheta'] = self.commande_histo_phitheta  # Alias for widget
        self.g['commande_contrast_CLAHE'] = self.commande_contrast_CLAHE
        self.g['commande_CLL'] = self.commande_CLL
        self.g['commande_filtrage_ON'] = self.commande_filtrage_ON
        self.g['commande_AmpSoft'] = self.commande_AmpSoft
        self.g['commande_HDR'] = self.commande_HDR
        self.g['commande_HQ_capt'] = self.commande_HQ_capt
        self.g['commande_hard_bin'] = self.commande_hard_bin
        self.g['commande_hot_pixels'] = self.commande_hot_pixels
        self.g['commande_HOTPIX'] = self.commande_hot_pixels  # Alias for widget

        # Slider callbacks - registered with widget-expected names and internal names
        self.g['choix_valeur_denoise'] = self.choix_valeur_denoise
        self.g['choix_val_grid_CLAHE'] = self.choix_val_grid_CLAHE
        self.g['choix_grid_CLAHE'] = self.choix_val_grid_CLAHE  # Alias for widget
        self.g['choix_val_contrast_CLAHE'] = self.choix_val_contrast_CLAHE
        self.g['choix_valeur_CLAHE'] = self.choix_val_contrast_CLAHE  # Alias for widget
        self.g['choix_ghost_reducer'] = self.choix_ghost_reducer
        self.g['choix_val_ghost_reducer'] = self.choix_ghost_reducer  # Alias for widget
        self.g['choix_val_3FNR_Thres'] = self.choix_val_3FNR_Thres
        self.g['choix_val_histo_min'] = self.choix_val_histo_min
        self.g['choix_histo_min'] = self.choix_val_histo_min  # Alias for widget
        self.g['choix_val_phi'] = self.choix_val_phi
        self.g['choix_phi'] = self.choix_val_phi  # Alias for widget
        self.g['choix_val_theta'] = self.choix_val_theta
        self.g['choix_theta'] = self.choix_val_theta  # Alias for widget
        self.g['choix_val_histo_max'] = self.choix_val_histo_max
        self.g['choix_histo_max'] = self.choix_val_histo_max  # Alias for widget
        self.g['choix_val_heq2'] = self.choix_val_heq2
        self.g['choix_heq2'] = self.choix_val_heq2  # Alias for widget
        self.g['choix_val_denoise_KNN'] = self.choix_val_denoise_KNN
        self.g['choix_val_KNN'] = self.choix_val_denoise_KNN  # Alias for widget
        self.g['choix_val_ampl'] = self.choix_val_ampl
        self.g['choix_amplif'] = self.choix_val_ampl  # Alias for widget
        self.g['choix_val_SGR'] = self.choix_val_SGR
        self.g['choix_SGR'] = self.choix_val_SGR  # Alias for widget
        self.g['choix_val_reduce_variation'] = self.choix_val_reduce_variation
        self.g['choix_val_AGR'] = self.choix_val_AGR
        self.g['choix_AGR'] = self.choix_val_AGR  # Alias for widget
        self.g['choix_val_sharpen'] = self.choix_val_sharpen
        self.g['choix_val_sharpen2'] = self.choix_val_sharpen2
        self.g['choix_val_sigma_sharpen'] = self.choix_val_sigma_sharpen
        self.g['choix_val_sigma_sharpen2'] = self.choix_val_sigma_sharpen2
        self.g['choix_val_SAT'] = self.choix_val_SAT
        self.g['choix_reds'] = self.choix_reds
        self.g['choix_w_reds'] = self.choix_reds  # Alias for widget
        self.g['choix_greens'] = self.choix_greens
        self.g['choix_w_greens'] = self.choix_greens  # Alias for widget
        self.g['choix_blues'] = self.choix_blues
        self.g['choix_w_blues'] = self.choix_blues  # Alias for widget
        self.g['choix_val_Mu'] = self.choix_val_Mu
        self.g['choix_Mu'] = self.choix_val_Mu  # Alias for widget
        self.g['choix_val_Ro'] = self.choix_val_Ro
        self.g['choix_Ro'] = self.choix_val_Ro  # Alias for widget
        self.g['choix_val_nb_sat'] = self.choix_val_nb_sat
        self.g['choix_val_nb_cra'] = self.choix_val_nb_cra

        # Display/misc callbacks
        self.g['commande_noir_blanc'] = self.commande_noir_blanc
        self.g['commande_reverse_RB'] = self.commande_reverse_RB
        self.g['commande_SAT_Img'] = self.commande_SAT_Img

        # Flip callbacks
        self.g['commande_flipV'] = self.commande_flipV
        self.g['commande_flipH'] = self.commande_flipH

        # Stacking callbacks
        self.g['choix_mean_stacking'] = self.choix_mean_stacking
        self.g['choix_sum_stacking'] = self.choix_sum_stacking
        self.g['choix_dyn_high'] = self.choix_dyn_high
        self.g['choix_dyn_low'] = self.choix_dyn_low
        self.g['choix_FS'] = self.choix_FS
        self.g['reset_FS'] = self.reset_FS
        self.g['command_BFReference'] = self.command_BFReference
        self.g['command_PFReference'] = self.command_PFReference

        # Sensor ratio callbacks
        self.g['choix_sensor_ratio_4_3'] = self.choix_sensor_ratio_4_3
        self.g['choix_sensor_ratio_16_9'] = self.choix_sensor_ratio_16_9
        self.g['choix_sensor_ratio_1_1'] = self.choix_sensor_ratio_1_1

        # Bayer callbacks
        self.g['choix_bayer_RAW'] = self.choix_bayer_RAW
        self.g['choix_bayer_RGGB'] = self.choix_bayer_RGGB
        self.g['choix_bayer_BGGR'] = self.choix_bayer_BGGR
        self.g['choix_bayer_GBRG'] = self.choix_bayer_GBRG
        self.g['choix_bayer_GRBG'] = self.choix_bayer_GRBG

        # Filter wheel callbacks
        self.g['choix_position_EFW0'] = self.choix_position_EFW0
        self.g['choix_position_EFW1'] = self.choix_position_EFW1
        self.g['choix_position_EFW2'] = self.choix_position_EFW2
        self.g['choix_position_EFW3'] = self.choix_position_EFW3
        self.g['choix_position_EFW4'] = self.choix_position_EFW4
        self.g['commande_FW'] = self.commande_FW

        # Tracking callbacks
        self.g['commande_TRKSAT'] = self.commande_TRKSAT
        self.g['commande_CONST'] = self.commande_CONST
        self.g['commande_REMSAT'] = self.commande_REMSAT
        self.g['commande_TRIGGER'] = self.commande_TRIGGER
        self.g['commande_BFR'] = self.commande_BFR
        self.g['commande_false_colours'] = self.commande_false_colours
        self.g['commande_AI_Craters'] = self.commande_AI_Craters
        self.g['commande_AI_Satellites'] = self.commande_AI_Satellites
        self.g['commande_AI_Trace'] = self.commande_AI_Trace

        # Misc callbacks
        self.g['choix_demo_left'] = self.choix_demo_left
        self.g['choix_demo_right'] = self.choix_demo_right
        self.g['choix_SAT_Vid'] = self.choix_SAT_Vid
        self.g['choix_SAT_Img'] = self.choix_SAT_Img
        self.g['commande_autoexposure'] = self.commande_autoexposure
        self.g['commande_autogain'] = self.commande_autogain
        self.g['commande_16bLL'] = self.commande_16bLL
        self.g['commande_IMQE'] = self.commande_IMQE
        self.g['choix_nb_captures'] = self.choix_nb_captures
        self.g['choix_deltat'] = self.choix_deltat
        self.g['choix_nb_video'] = self.choix_nb_video
        self.g['choix_position_frame'] = self.choix_position_frame
        self.g['raz_framecount'] = self.raz_framecount
        self.g['raz_tracking'] = self.raz_tracking
        self.g['stop_tracking'] = self.stop_tracking
        self.g['Capture_Ref_Img'] = self.Capture_Ref_Img
        self.g['commande_sub_img_ref'] = self.commande_sub_img_ref
        self.g['commande_Blur_img_ref'] = self.commande_Blur_img_ref
        self.g['commande_GBL'] = self.commande_GBL
        self.g['reset_general_FS'] = self.reset_general_FS

        # Additional missing callbacks (aliases and new)
        self.g['choix_KNN'] = self.commande_KNN  # Alias for checkbox
        self.g['choix_TH_16B'] = self.choix_TH_16B
        self.g['choix_Var_CLL'] = self.choix_Var_CLL
        self.g['choix_val_BFR'] = self.choix_val_BFR

    # =========================================================================
    # Mode Selection Callbacks
    # =========================================================================

    def mode_Lineaire(self):
        """Set linear amplification mode."""
        self.g['flag_lin_gauss'] = 1
        for x in range(0, 255):
            self.g['Corr_GS'][x] = 1

    def mode_Gauss(self):
        """Set Gaussian amplification mode."""
        self.g['flag_lin_gauss'] = 2
        val_Mu = self.g['val_Mu']
        val_Ro = self.g['val_Ro']
        for x in range(0, 255):
            self.g['Corr_GS'][x] = np.exp(-0.5 * ((x * 0.0392157 - 5 - val_Mu) / val_Ro) ** 2)

    def mode_Stars(self):
        """Set stars amplification mode."""
        self.g['flag_lin_gauss'] = 3
        val_Mu = self.g['val_Mu']
        val_Ro = self.g['val_Ro']
        for x in range(0, 255):
            self.g['Corr_GS'][x] = np.exp(-0.5 * ((x * 0.0392157 - 5 - val_Mu) / val_Ro) ** 2)

    def HDR_Mertens(self):
        """Set HDR mode to Mertens."""
        self.g['mode_HDR'] = "Mertens"

    def HDR_Median(self):
        """Set HDR mode to Median."""
        self.g['mode_HDR'] = "Median"

    def HDR_Mean(self):
        """Set HDR mode to Mean."""
        self.g['mode_HDR'] = "Mean"

    def mode_gradient(self):
        """Set gradient/vignetting mode to gradient."""
        self.g['grad_vignet'] = 1

    def mode_vignetting(self):
        """Set gradient/vignetting mode to vignetting."""
        self.g['grad_vignet'] = 2

    # =========================================================================
    # Acquisition Mode Callbacks
    # =========================================================================

    def mode_acq_rapide(self):
        """Set fast acquisition mode (100-10000 microseconds)."""
        self._set_acquisition_mode('Fast')

    def mode_acq_mediumF(self):
        """Set medium-fast acquisition mode (1-400 ms)."""
        self._set_acquisition_mode('MedF')

    def mode_acq_mediumS(self):
        """Set medium-slow acquisition mode (1-1000 ms)."""
        self._set_acquisition_mode('MedS')

    def mode_acq_lente(self):
        """Set slow acquisition mode (500-20000 ms)."""
        self._set_acquisition_mode('Slow')

    # =========================================================================
    # Camera Control Callbacks
    # =========================================================================

    @safe_callback
    def valeur_exposition(self, event=None):
        """Handle exposure value change from slider."""
        if not self._get('flag_camera_ok', False):
            return
        if self._get('flag_autoexposure_exposition', False):
            return
            
        self._set('flag_stop_acquisition', True)
        
        echelle1 = self._get_widget('echelle1')
        if echelle1 is None:
            self._set('flag_stop_acquisition', False)
            return
            
        val_exposition = echelle1.get()
        self._set('val_exposition', val_exposition)

        if self._get('flag_acq_rapide') == "Fast":
            exposition = val_exposition
        else:
            exposition = val_exposition * 1000
        self._set('exposition', exposition)

        camera = self._get('camera')
        asi = self._get('asi')
        if camera is None or asi is None:
            self._set('flag_stop_acquisition', False)
            return
            
        camera.set_control_value(asi.ASI_EXPOSURE, exposition)

        if self._get('flag_acq_rapide') == "Fast":
            timeoutexp = (exposition / 1000) * 2 + 500
        else:
            timeoutexp = val_exposition * 2 + 500
        self._set('timeoutexp', timeoutexp)
        camera.default_timeout = timeoutexp

        time.sleep(0.05)
        self._set('flag_stop_acquisition', False)

    @safe_callback
    def valeur_gain(self, event=None):
        """Handle gain value change from slider."""
        if not self._get('flag_camera_ok', False):
            return
        if self._get('flag_autoexposure_gain', False):
            return
            
        echelle2 = self._get_widget('echelle2')
        if echelle2 is None:
            return
            
        val_gain = echelle2.get()
        self._set('val_gain', val_gain)
        
        camera = self._get('camera')
        asi = self._get('asi')
        if camera is not None and asi is not None:
            camera.set_control_value(asi.ASI_GAIN, val_gain)

    @safe_callback
    def choix_BIN1(self, event=None):
        """Set BIN mode to 1."""
        self._set_bin_mode(1, max_resolution=9)

    @safe_callback
    def choix_BIN2(self, event=None):
        """Set BIN mode to 2."""
        self._set_bin_mode(2, max_resolution=7)

    def _set_bin_mode(self, bin_mode: int, max_resolution: int) -> None:
        """
        Set camera BIN mode with error handling.
        
        Args:
            bin_mode: 1 or 2
            max_resolution: Maximum resolution setting for the scale
        """
        if not self._get('flag_camera_ok', False):
            self._set('mode_BIN', bin_mode)
            return
            
        if self._get('flag_cap_video', False):
            return
            
        try:
            self._set('flag_image_disponible', False)
            self._call_func('reset_general_FS')
            self._call_func('reset_FS')
            self._set('flag_TIP', 0)
            
            choix_TIP = self._get('choix_TIP')
            if choix_TIP is not None:
                choix_TIP.set(0)
                
            self._set('flag_nouvelle_resolution', True)
            self._set('flag_stop_acquisition', True)
            self._call_func('stop_tracking')
            time.sleep(0.5)

            Scale = self._get('Scale')
            cadre = self._get('cadre')
            
            if Scale is None or cadre is None:
                logger.error("Cannot create scale widget for BIN mode")
                self._set('flag_stop_acquisition', False)
                return
                
            echelle3 = Scale(
                cadre, from_=1, to=max_resolution, 
                command=self.choix_resolution_camera,
                orient=self._get('HORIZONTAL'), length=130, width=7,
                resolution=1, label="", showvalue=1, tickinterval=1,
                sliderlength=20
            )
            self._set('val_resolution', 1)
            echelle3.set(1)
            echelle3.place(
                anchor="w", 
                x=self._get('xS3', 0) + self._get('delta_s', 0), 
                y=self._get('yS3', 0)
            )
            self._set('echelle3', echelle3)

            time.sleep(0.1)
            self._set('mode_BIN', bin_mode)
            self.choix_resolution_camera()
            
        except Exception as e:
            logger.error(f"Error setting BIN mode {bin_mode}: {e}")
        finally:
            self._set('flag_stop_acquisition', False)

    @safe_callback
    def choix_resolution_camera(self, event=None):
        """Handle camera resolution change."""
        if self.g.get('flag_camera_ok', False):
            if not self.g.get('flag_cap_video', False):
                self.g['flag_autorise_acquisition'] = False
                self.g['reset_FS']()
                self.g['reset_general_FS']()
                time.sleep(0.5)
                self.g['flag_TIP'] = 0
                self.g['choix_TIP'].set(0)
                self.g['flag_stop_acquisition'] = True
                time.sleep(0.1)
                self.g['stop_tracking']()
                time.sleep(0.1)

                val_resolution = self.g['echelle3'].get()
                self.g['val_resolution'] = val_resolution
                mode_BIN = self.g['mode_BIN']

                if mode_BIN == 1:
                    res_cam_x = self.g['RES_X_BIN1'][val_resolution - 1]
                    res_cam_y = self.g['RES_Y_BIN1'][val_resolution - 1]
                else:
                    res_cam_x = self.g['RES_X_BIN2'][val_resolution - 1]
                    res_cam_y = self.g['RES_Y_BIN2'][val_resolution - 1]

                self.g['res_cam_x'] = res_cam_x
                self.g['res_cam_y'] = res_cam_y
                self.g['inSize'] = (int(res_cam_x), int(res_cam_y))

                camera = self.g['camera']
                camera.stop_video_capture()
                time.sleep(0.1)
                camera.set_roi(None, None, res_cam_x, res_cam_y, mode_BIN, self.g['format_capture'])
                time.sleep(0.1)
                self.g['flag_nouvelle_resolution'] = True
                camera.start_video_capture()
                print("resolution camera = ", res_cam_x, " ", res_cam_y)

                if not self.g.get('flag_HDR', False):
                    self.g['flag_autorise_acquisition'] = True
                self.g['flag_stop_acquisition'] = False

    def choix_USB(self, event=None):
        """Handle USB bandwidth change."""
        if self.g.get('flag_camera_ok', False):
            val_USB = self.g['echelle50'].get()
            self.g['val_USB'] = val_USB
            camera = self.g['camera']
            asi = self.g['asi']
            camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, val_USB)

    def choix_ASI_GAMMA(self, event=None):
        """Handle ASI gamma change."""
        if self.g.get('flag_camera_ok', False):
            ASIGAMMA = self.g['echelle204'].get()
            self.g['ASIGAMMA'] = ASIGAMMA
            camera = self.g['camera']
            asi = self.g['asi']
            camera.set_control_value(asi.ASI_GAMMA, ASIGAMMA)

    def choix_read_speed_fast(self, event=None):
        """Set camera read speed to fast."""
        if self.g.get('flag_camera_ok', False):
            camera = self.g['camera']
            asi = self.g['asi']
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, 1)
            self.g['flag_read_speed'] = "Fast"

    def choix_read_speed_slow(self, event=None):
        """Set camera read speed to slow."""
        if self.g.get('flag_camera_ok', False):
            camera = self.g['camera']
            asi = self.g['asi']
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, 0)
            self.g['flag_read_speed'] = "Slow"

    def choix_w_red(self, event=None):
        """Handle white balance red change."""
        if self.g.get('flag_camera_ok', False):
            val_red = self.g['echelle14'].get()
            self.g['val_red'] = val_red
            camera = self.g['camera']
            asi = self.g['asi']
            camera.set_control_value(asi.ASI_WB_R, val_red)

    def choix_w_blue(self, event=None):
        """Handle white balance blue change."""
        if self.g.get('flag_camera_ok', False):
            val_blue = self.g['echelle15'].get()
            self.g['val_blue'] = val_blue
            camera = self.g['camera']
            asi = self.g['asi']
            camera.set_control_value(asi.ASI_WB_B, val_blue)

    # =========================================================================
    # Filter Toggle Callbacks (checkboxes)
    # =========================================================================

    def commande_img_Neg(self):
        """Toggle image negative."""
        self._toggle_flag('choix_img_Neg', 'ImageNeg')

    def commande_TIP(self):
        """Toggle TIP mode."""
        self._toggle_flag('choix_TIP', 'flag_TIP')

    def commande_SAT(self):
        """Toggle saturation enhancement."""
        self._toggle_flag('choix_SAT', 'flag_SAT', 'flag_SAT')

    def commande_SAT2PASS(self):
        """Toggle 2-pass saturation enhancement."""
        self._toggle_flag('choix_SAT2PASS', 'flag_SAT2PASS', 'flag_SAT2PASS')

    def commande_mount(self):
        """Toggle mount position display."""
        self._toggle_flag('choix_mount', 'flag_mountpos')
        # Sync to app_state (non-filter_params)
        if self.state is not None:
            self.state.flag_mountpos = self.g['flag_mountpos']

    def commande_cross(self):
        """Toggle crosshair display."""
        self._toggle_flag('choix_cross', 'flag_cross')
        # Sync to app_state.display
        if self.state is not None and self.state.display is not None:
            self.state.display.flag_cross = self.g['flag_cross']

    def commande_mode_full_res(self):
        """Toggle full resolution mode."""
        self.g['delta_zx'] = 0
        self.g['delta_zy'] = 0
        self._toggle_flag('choix_mode_full_res', 'flag_full_res')
        # Sync to app_state
        if self.state is not None and self.state.display is not None:
            self.state.display.flag_full_res = self.g['flag_full_res']
            self.state.display.delta_zx = self.g['delta_zx']
            self.state.display.delta_zy = self.g['delta_zy']

    def commande_sharpen_soft1(self):
        """Toggle sharpen 1."""
        self._toggle_flag('choix_sharpen_soft1', 'flag_sharpen_soft1', 'flag_sharpen_soft1')

    def commande_sharpen_soft2(self):
        """Toggle sharpen 2."""
        self._toggle_flag('choix_sharpen_soft2', 'flag_sharpen_soft2', 'flag_sharpen_soft2')

    def commande_NLM2(self):
        """Toggle NLM2 denoising."""
        self._toggle_flag('choix_NLM2', 'flag_NLM2', 'flag_NLM2')

    def commande_denoise_Paillou(self):
        """Toggle Paillou denoising 1."""
        if self.g['choix_denoise_Paillou'].get() == 0:
            self.g['flag_denoise_Paillou'] = 0
        else:
            self.g['flag_denoise_Paillou'] = 1
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.flag_denoise_Paillou = self.g['flag_denoise_Paillou']

    def commande_denoise_Paillou2(self):
        """Toggle Paillou denoising 2."""
        if self.g['choix_denoise_Paillou2'].get() == 0:
            self.g['flag_denoise_Paillou2'] = 0
        else:
            self.g['flag_denoise_Paillou2'] = 1
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.flag_denoise_Paillou2 = self.g['flag_denoise_Paillou2']

    def commande_HST(self):
        """Toggle histogram display."""
        self._toggle_flag('choix_HST', 'flag_HST')

    def commande_TRSF(self):
        """Toggle TRSF display."""
        self._toggle_flag('choix_TRSF', 'flag_TRSF')

    def commande_TRGS(self):
        """Toggle TRGS display."""
        self._toggle_flag('choix_TRGS', 'flag_TRGS')

    def commande_TRCLL(self):
        """Toggle TRCLL display."""
        self._toggle_flag('choix_TRCLL', 'flag_TRCLL')

    def commande_DEMO(self):
        """Toggle demo mode."""
        self._toggle_flag('choix_DEMO', 'flag_DEMO')

    def commande_STAB(self):
        """Toggle stabilization."""
        self.g['delta_tx'] = 0
        self.g['delta_ty'] = 0
        self.g['DSW'] = 0
        if self.g['choix_STAB'].get() == 0:
            self.g['flag_STAB'] = False
        else:
            self.g['flag_STAB'] = True
        self.g['flag_Template'] = False

    def commande_DETECT_STARS(self):
        """Toggle star detection."""
        if self.g['choix_DETECT_STARS'].get() == 0:
            self.g['flag_DETECT_STARS'] = 0
        else:
            self.g['flag_DETECT_STARS'] = 1
            self.g['flag_nouvelle_resolution'] = True

    def commande_AANR(self):
        """Toggle AANR filter (front, high dynamic)."""
        if self.g['choix_AANR'].get() == 0:
            self.g['flag_AANR'] = False
        else:
            self.g['flag_AANR'] = True
            self.g['compteur_AANR'] = 0
            self.g['Im1fsdnOK'] = False
            self.g['Im2fsdnOK'] = False
        # Sync to app_state
        if self.state is not None:
            if self.state.filter_params is not None:
                self.state.filter_params.flag_AANR = self.g['flag_AANR']
            if self.state.filter_state is not None:
                self.state.filter_state.compteur_AANR = self.g['compteur_AANR']
                self.state.filter_state.Im1fsdnOK = self.g['Im1fsdnOK']
                self.state.filter_state.Im2fsdnOK = self.g['Im2fsdnOK']

    def commande_AANRB(self):
        """Toggle AANRB filter (back, high dynamic)."""
        if self.g['choix_AANRB'].get() == 0:
            self.g['flag_AANRB'] = False
        else:
            self.g['flag_AANRB'] = True
            self.g['compteur_AANRB'] = 0
            self.g['Im1fsdnOKB'] = False
            self.g['Im2fsdnOKB'] = False
        # Sync to app_state
        if self.state is not None:
            if self.state.filter_params is not None:
                self.state.filter_params.flag_AANRB = self.g['flag_AANRB']
            if self.state.filter_state is not None:
                self.state.filter_state.compteur_AANRB = self.g['compteur_AANRB']
                self.state.filter_state.Im1fsdnOKB = self.g['Im1fsdnOKB']
                self.state.filter_state.Im2fsdnOKB = self.g['Im2fsdnOKB']

    def commande_3FNR(self):
        """Toggle 3FNR filter (front)."""
        if self.g['choix_3FNR'].get() == 0:
            self.g['flag_3FNR'] = False
        else:
            self.g['flag_3FNR'] = True
            self.g['compteur_3FNR'] = 0
            self.g['img1_3FNROK'] = False
            self.g['img2_3FNROK'] = False
            self.g['img3_3FNROK'] = False
            self.g['FNR_First_Start'] = True
        # Sync to app_state
        if self.state is not None:
            if self.state.filter_params is not None:
                self.state.filter_params.flag_3FNR = self.g['flag_3FNR']
            if self.state.filter_state is not None:
                self.state.filter_state.compteur_3FNR = self.g['compteur_3FNR']
                self.state.filter_state.img1_3FNROK = self.g['img1_3FNROK']
                self.state.filter_state.img2_3FNROK = self.g['img2_3FNROK']
                self.state.filter_state.img3_3FNROK = self.g['img3_3FNROK']
                self.state.filter_state.FNR_First_Start = self.g['FNR_First_Start']

    def commande_3FNR2(self):
        """Toggle 3FNR2 filter (back)."""
        if self.g['choix_3FNR2'].get() == 0:
            self.g['flag_3FNR2'] = False
        else:
            self.g['flag_3FNR2'] = True
            self.g['compteur_3FNR2'] = 0
            self.g['img1_3FNR2OK'] = False
            self.g['img2_3FNR2OK'] = False
            self.g['img3_3FNR2OK'] = False
            self.g['FNR2_First_Start'] = True
        # Sync to app_state
        if self.state is not None:
            if self.state.filter_params is not None:
                self.state.filter_params.flag_3FNR2 = self.g['flag_3FNR2']
            if self.state.filter_state is not None:
                self.state.filter_state.compteur_3FNR2 = self.g['compteur_3FNR2']
                self.state.filter_state.img1_3FNR2OK = self.g['img1_3FNR2OK']
                self.state.filter_state.img2_3FNR2OK = self.g['img2_3FNR2OK']
                self.state.filter_state.img3_3FNR2OK = self.g['img3_3FNR2OK']
                self.state.filter_state.FNR2_First_Start = self.g['FNR2_First_Start']

    def commande_3FNRB(self):
        """Toggle 3FNRB filter (front, low dynamic)."""
        if self.g['choix_3FNRB'].get() == 0:
            self.g['flag_3FNRB'] = False
        else:
            self.g['flag_3FNRB'] = True
            self.g['compteur_3FNRB'] = 0
            self.g['img1_3FNRBOK'] = False
            self.g['img2_3FNRBOK'] = False
            self.g['img3_3FNRBOK'] = False
            self.g['FNRB_First_Start'] = True
        # Sync to app_state
        if self.state is not None:
            if self.state.filter_params is not None:
                self.state.filter_params.flag_3FNRB = self.g['flag_3FNRB']
            if self.state.filter_state is not None:
                self.state.filter_state.compteur_3FNRB = self.g['compteur_3FNRB']
                self.state.filter_state.img1_3FNRBOK = self.g['img1_3FNRBOK']
                self.state.filter_state.img2_3FNRBOK = self.g['img2_3FNRBOK']
                self.state.filter_state.img3_3FNRBOK = self.g['img3_3FNRBOK']
                self.state.filter_state.FNRB_First_Start = self.g['FNRB_First_Start']

    def commande_3FNR2B(self):
        """Toggle 3FNR2B filter (back, low dynamic)."""
        if self.g['choix_3FNR2B'].get() == 0:
            self.g['flag_3FNR2B'] = False
        else:
            self.g['flag_3FNR2B'] = True
            self.g['compteur_3FNR2B'] = 0
            self.g['img1_3FNR2BOK'] = False
            self.g['img2_3FNR2BOK'] = False
            self.g['img3_3FNR2BOK'] = False
            self.g['FNR2B_First_Start'] = True
        # Sync to app_state
        if self.state is not None:
            if self.state.filter_params is not None:
                self.state.filter_params.flag_3FNR2B = self.g['flag_3FNR2B']
            if self.state.filter_state is not None:
                self.state.filter_state.compteur_3FNR2B = self.g['compteur_3FNR2B']
                self.state.filter_state.img1_3FNR2BOK = self.g['img1_3FNR2BOK']
                self.state.filter_state.img2_3FNR2BOK = self.g['img2_3FNR2BOK']
                self.state.filter_state.img3_3FNR2BOK = self.g['img3_3FNR2BOK']
                self.state.filter_state.FNR2B_First_Start = self.g['FNR2B_First_Start']

    def commande_GR(self):
        """Toggle gradient removal."""
        self._toggle_flag('choix_GR', 'flag_GR', 'flag_GR')

    def commande_ghost_reducer(self):
        """Toggle ghost reducer."""
        self._toggle_flag('choix_ghost_reducer', 'flag_ghost_reducer', 'flag_ghost_reducer')

    def commande_KNN(self):
        """Toggle KNN denoising."""
        self._toggle_flag('choix_KNN', 'flag_KNN', 'flag_KNN')

    def commande_reduce_variation(self):
        """Toggle reduce variation filter."""
        self._toggle_flag('choix_reduce_variation', 'flag_reduce_variation', 'flag_reduce_variation')

    def commande_histo_equalize(self):
        """Toggle histogram equalization."""
        self._toggle_flag('choix_histo_equalize', 'flag_histogram_equalize', 'flag_histogram_equalize')

    def commande_histo_stretch(self):
        """Toggle histogram stretch."""
        self._toggle_flag('choix_histogram_stretch', 'flag_histogram_stretch', 'flag_histogram_stretch')

    def commande_histo_phitheta(self):
        """Toggle histogram phi-theta correction."""
        self._toggle_flag('choix_histogram_phitheta', 'flag_histogram_phitheta', 'flag_histogram_phitheta')

    def commande_contrast_CLAHE(self):
        """Toggle CLAHE contrast enhancement."""
        self._toggle_flag('choix_contrast_CLAHE', 'flag_contrast_CLAHE', 'flag_contrast_CLAHE')

    def commande_CLL(self):
        """Toggle CLL (Contrast Low Light)."""
        self._toggle_flag('choix_CLL', 'flag_CLL', 'flag_CLL')

    def commande_filtrage_ON(self):
        """Toggle filter ON/OFF."""
        self._toggle_flag('choix_filtrage_ON', 'flag_filtrage_ON', 'flag_filtrage_ON')

    def commande_AmpSoft(self):
        """Toggle software amplification."""
        self._toggle_flag('choix_AmpSoft', 'flag_AmpSoft', 'flag_AmpSoft')

    def commande_HDR(self):
        """Toggle HDR mode."""
        self._toggle_flag('choix_HDR', 'flag_HDR')
        # HDR has special side effect
        if not self.g.get('flag_HDR', False):
            self.g['flag_autorise_acquisition'] = True

    def commande_hot_pixels(self):
        """Toggle hot pixel removal."""
        self._toggle_flag('choix_hot_pixels', 'flag_hot_pixels')

    def commande_HQ_capt(self):
        """Toggle high quality capture mode."""
        choix = self.g.get('choix_HQ_capt')
        self.g['flag_HQ'] = 1 if (choix and choix.get() != 0) else 0

    def commande_hard_bin(self):
        """Toggle hardware binning."""
        self._toggle_flag('choix_hard_bin', 'flag_HB')

    def commande_reduce_variation_post_treatment(self):
        """Toggle reduce variation post-treatment mode."""
        if self.g['choix_reduce_variation_post_treatment'].get() == 0:
            self.g['flag_reduce_variation_post_treatment'] = False
            self.g['flag_BFREFPT'] = False
            self.g['flag_BFREF_image_PT'] = False
            self.g['max_qual_PT'] = 0
        else:
            self.g['flag_reduce_variation_post_treatment'] = True
            self.g['max_qual_PT'] = 0
            self.g['flag_BFREFPT'] = True
            self.g['flag_BFREF_image_PT'] = False

    def choix_TH_16B(self, event=None):
        """Set 16-bit threshold from slider."""
        TH_16B = self.g['echelle804'].get()
        self.g['TH_16B'] = TH_16B
        self.g['threshold_16bits'] = 2 ** TH_16B - 1

    def choix_Var_CLL(self, event=None):
        """Update CLL correction parameters from sliders."""
        val_MuCLL = self.g['echelle200'].get()
        val_RoCLL = self.g['echelle201'].get()
        val_AmpCLL = self.g['echelle202'].get()
        self.g['val_MuCLL'] = val_MuCLL
        self.g['val_RoCLL'] = val_RoCLL
        self.g['val_AmpCLL'] = val_AmpCLL
        for x in range(0, 256):
            Corr = np.exp(-0.5 * ((x * 0.0392157 - val_MuCLL) / val_RoCLL) ** 2)
            self.g['Corr_CLL'][x] = int(x * (1 / (1 + val_AmpCLL * Corr)))
            if x > 0:
                if self.g['Corr_CLL'][x] <= self.g['Corr_CLL'][x - 1]:
                    self.g['Corr_CLL'][x] = self.g['Corr_CLL'][x - 1]

    def choix_val_BFR(self, event=None):
        """Set BFR value from slider."""
        self.g['val_BFR'] = self.g['echelle300'].get()

    # =========================================================================
    # Slider Callbacks
    # =========================================================================

    def choix_valeur_denoise(self, event=None):
        """Handle denoise value change."""
        val_denoise = float(event) if event is not None else self.g.get('val_denoise', 0.5)
        if val_denoise == 0:
            val_denoise += 1
        self.g['val_denoise'] = val_denoise
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_denoise = val_denoise

    def choix_val_grid_CLAHE(self, event=None):
        """Handle CLAHE grid value change."""
        val_grid_CLAHE = int(float(event)) if event is not None else self.g.get('val_grid_CLAHE', 8)
        if val_grid_CLAHE == 0:
            val_grid_CLAHE = 1
        self.g['val_grid_CLAHE'] = val_grid_CLAHE
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_grid_CLAHE = val_grid_CLAHE

    def choix_val_contrast_CLAHE(self, event=None):
        """Handle CLAHE contrast value change."""
        val_contrast_CLAHE = float(event) if event is not None else self.g.get('val_contrast_CLAHE', 1.0)
        self.g['val_contrast_CLAHE'] = val_contrast_CLAHE
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_contrast_CLAHE = val_contrast_CLAHE

    def choix_ghost_reducer(self, event=None):
        """Handle ghost reducer value change."""
        val_ghost_reducer = float(event) if event is not None else self.g.get('val_ghost_reducer', 40)
        self.g['val_ghost_reducer'] = val_ghost_reducer
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_ghost_reducer = val_ghost_reducer

    def choix_val_3FNR_Thres(self, event=None):
        """Handle 3FNR threshold value change."""
        val_3FNR_Threshold = float(event) if event is not None else self.g.get('val_3FNR_Threshold', 0.5)
        self.g['val_3FNR_Threshold'] = val_3FNR_Threshold
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_3FNR_Threshold = val_3FNR_Threshold

    def choix_val_histo_min(self, event=None):
        """Handle histogram min value change."""
        val_histo_min = float(event) if event is not None else self.g.get('val_histo_min', 0)
        self.g['val_histo_min'] = val_histo_min
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_histo_min = val_histo_min

    def choix_val_phi(self, event=None):
        """Handle phi value change."""
        val_phi = float(event) if event is not None else self.g.get('val_phi', 1.0)
        self.g['val_phi'] = val_phi
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_phi = val_phi

    def choix_val_theta(self, event=None):
        """Handle theta value change."""
        val_theta = float(event) if event is not None else self.g.get('val_theta', 100)
        self.g['val_theta'] = val_theta
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_theta = val_theta

    def choix_val_histo_max(self, event=None):
        """Handle histogram max value change."""
        val_histo_max = float(event) if event is not None else self.g.get('val_histo_max', 255)
        self.g['val_histo_max'] = val_histo_max
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_histo_max = val_histo_max

    def choix_val_heq2(self, event=None):
        """Handle heq2 value change."""
        val_heq2 = float(event) if event is not None else self.g.get('val_heq2', 1.0)
        self.g['val_heq2'] = val_heq2
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_heq2 = val_heq2

    def choix_val_denoise_KNN(self, event=None):
        """Handle KNN denoise value change."""
        val_denoise_KNN = float(event) if event is not None else self.g.get('val_denoise_KNN', 0.5)
        self.g['val_denoise_KNN'] = val_denoise_KNN
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_denoise_KNN = val_denoise_KNN

    def choix_val_ampl(self, event=None):
        """Handle amplification value change."""
        val_ampl = float(event) if event is not None else self.g.get('val_ampl', 1.0)
        self.g['val_ampl'] = val_ampl
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_ampl = val_ampl

    def choix_val_SGR(self, event=None):
        """Handle SGR value change."""
        val_SGR = float(event) if event is not None else self.g.get('val_SGR', 50)
        self.g['val_SGR'] = val_SGR
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_SGR = val_SGR

    def choix_val_reduce_variation(self, event=None):
        """Handle reduce variation value change."""
        val_reduce_variation = float(event) if event is not None else self.g.get('val_reduce_variation', 1.0)
        self.g['val_reduce_variation'] = val_reduce_variation
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_reduce_variation = val_reduce_variation

    def choix_val_AGR(self, event=None):
        """Handle AGR value change."""
        val_AGR = float(event) if event is not None else self.g.get('val_AGR', 50)
        self.g['val_AGR'] = val_AGR
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_AGR = val_AGR

    def choix_val_sharpen(self, event=None):
        """Handle sharpen 1 value change."""
        val_sharpen = float(event) if event is not None else self.g.get('val_sharpen', 1.0)
        self.g['val_sharpen'] = val_sharpen
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_sharpen = val_sharpen

    def choix_val_sharpen2(self, event=None):
        """Handle sharpen 2 value change."""
        val_sharpen2 = float(event) if event is not None else self.g.get('val_sharpen2', 1.0)
        self.g['val_sharpen2'] = val_sharpen2
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_sharpen2 = val_sharpen2

    def choix_val_sigma_sharpen(self, event=None):
        """Handle sharpen 1 sigma value change."""
        val_sigma_sharpen = float(event) if event is not None else self.g.get('val_sigma_sharpen', 3)
        self.g['val_sigma_sharpen'] = val_sigma_sharpen
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_sigma_sharpen = val_sigma_sharpen

    def choix_val_sigma_sharpen2(self, event=None):
        """Handle sharpen 2 sigma value change."""
        val_sigma_sharpen2 = float(event) if event is not None else self.g.get('val_sigma_sharpen2', 3)
        self.g['val_sigma_sharpen2'] = val_sigma_sharpen2
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_sigma_sharpen2 = val_sigma_sharpen2

    def choix_val_SAT(self, event=None):
        """Handle saturation value change."""
        val_SAT = float(event) if event is not None else self.g.get('val_SAT', 1.0)
        self.g['val_SAT'] = val_SAT
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_SAT = val_SAT

    def choix_reds(self, event=None):
        """Handle red channel adjustment."""
        val_reds = float(event) if event is not None else self.g['echelle100'].get()
        self.g['val_reds'] = val_reds
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_reds = val_reds

    def choix_greens(self, event=None):
        """Handle green channel adjustment."""
        val_greens = float(event) if event is not None else self.g['echelle101'].get()
        self.g['val_greens'] = val_greens
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_greens = val_greens

    def choix_blues(self, event=None):
        """Handle blue channel adjustment."""
        val_blues = float(event) if event is not None else self.g['echelle102'].get()
        self.g['val_blues'] = val_blues
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_blues = val_blues

    def choix_val_Mu(self, event=None):
        """Handle Mu value change (for Gaussian mode)."""
        val_Mu = float(event) if event is not None else self.g['echelle82'].get()
        self.g['val_Mu'] = val_Mu
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_Mu = val_Mu

    def choix_val_Ro(self, event=None):
        """Handle Ro value change (for Gaussian mode)."""
        val_Ro = float(event) if event is not None else self.g['echelle84'].get()
        self.g['val_Ro'] = val_Ro
        # Sync to app_state
        if self.state is not None and self.state.filter_params is not None:
            self.state.filter_params.val_Ro = val_Ro

    def choix_val_nb_sat(self, event=None):
        """Handle number of satellites value change."""
        val_nb_sat = int(float(event)) if event is not None else int(self.g['echelle500'].get())
        self.g['val_nb_sat'] = val_nb_sat

    def choix_val_nb_cra(self, event=None):
        """Handle number of craters value change."""
        val_nb_cra = int(float(event)) if event is not None else int(self.g['echelle501'].get())
        self.g['val_nb_cra'] = val_nb_cra

    # =========================================================================
    # Display/Misc Callbacks
    # =========================================================================

    def commande_noir_blanc(self):
        """Toggle black and white mode."""
        if self.g['choix_noir_blanc'].get() == 0:
            self.g['flag_noir_blanc'] = False
        else:
            self.g['flag_noir_blanc'] = True
        # Sync to app_state
        if self.state is not None and self.state.display is not None:
            self.state.display.flag_noir_blanc = self.g['flag_noir_blanc']

    def commande_reverse_RB(self):
        """Toggle red-blue channel swap."""
        if self.g['choix_reverse_RB'].get() == 0:
            self.g['flag_reverse_RB'] = False
        else:
            self.g['flag_reverse_RB'] = True
        # Sync to app_state
        if self.state is not None and self.state.display is not None:
            self.state.display.flag_reverse_RB = self.g['flag_reverse_RB']

    def commande_SAT_Img(self):
        """Toggle satellite image mode."""
        if self.g['choix_SAT_Img'].get() == 0:
            self.g['flag_SAT_Img'] = False
        else:
            self.g['flag_SAT_Img'] = True
        # Sync to app_state
        if self.state is not None and self.state.display is not None:
            self.state.display.flag_SAT_Img = self.g['flag_SAT_Img']

    # =========================================================================
    # Flip Callbacks
    # =========================================================================

    def _apply_bayer_flip(self, bayer_type, flip_v, flip_h, is_camera=True):
        """Helper to apply Bayer pattern changes based on flip state."""
        asi = self.g.get('asi')
        camera = self.g.get('camera')

        if bayer_type == "RAW":
            self.g['type_debayer'] = 0
            self.g['GPU_BAYER'] = 0
            if is_camera and camera:
                flip_val = 3 if (flip_v and flip_h) else (2 if flip_v else (1 if flip_h else 0))
                camera.set_control_value(asi.ASI_FLIP, flip_val)
            else:
                if flip_v and flip_h:
                    self.g['type_flip'] = "both"
                elif flip_v:
                    self.g['type_flip'] = "vertical"
                elif flip_h:
                    self.g['type_flip'] = "horizontal"
                else:
                    self.g['type_flip'] = "none"
        elif bayer_type == "RGGB":
            self.g['type_debayer'] = cv2.COLOR_BayerBG2RGB
            bayer_map = {(1, 1): 2, (0, 0): 1, (1, 0): 4, (0, 1): 3}
            self.g['GPU_BAYER'] = bayer_map.get((flip_v, flip_h), 1)
            if is_camera and camera:
                flip_val = 3 if (flip_v and flip_h) else (2 if flip_v else (1 if flip_h else 0))
                camera.set_control_value(asi.ASI_FLIP, flip_val)
            else:
                if flip_v and flip_h:
                    self.g['type_flip'] = "both"
                elif flip_v:
                    self.g['type_flip'] = "vertical"
                elif flip_h:
                    self.g['type_flip'] = "horizontal"
                else:
                    self.g['type_flip'] = "none"
        elif bayer_type == "BGGR":
            self.g['type_debayer'] = cv2.COLOR_BayerRG2RGB
            bayer_map = {(1, 1): 1, (0, 0): 2, (1, 0): 3, (0, 1): 4}
            self.g['GPU_BAYER'] = bayer_map.get((flip_v, flip_h), 2)
            if is_camera and camera:
                flip_val = 3 if (flip_v and flip_h) else (2 if flip_v else (1 if flip_h else 0))
                camera.set_control_value(asi.ASI_FLIP, flip_val)
            else:
                if flip_v and flip_h:
                    self.g['type_flip'] = "both"
                elif flip_v:
                    self.g['type_flip'] = "vertical"
                elif flip_h:
                    self.g['type_flip'] = "horizontal"
                else:
                    self.g['type_flip'] = "none"
        elif bayer_type == "GRBG":
            self.g['type_debayer'] = cv2.COLOR_BayerGR2RGB
            bayer_map = {(1, 1): 3, (0, 0): 4, (1, 0): 2, (0, 1): 1}
            self.g['GPU_BAYER'] = bayer_map.get((flip_v, flip_h), 4)
            if is_camera and camera:
                flip_val = 3 if (flip_v and flip_h) else (2 if flip_v else (1 if flip_h else 0))
                camera.set_control_value(asi.ASI_FLIP, flip_val)
            else:
                if flip_v and flip_h:
                    self.g['type_flip'] = "both"
                elif flip_v:
                    self.g['type_flip'] = "vertical"
                elif flip_h:
                    self.g['type_flip'] = "horizontal"
                else:
                    self.g['type_flip'] = "none"
        elif bayer_type == "GBRG":
            self.g['type_debayer'] = cv2.COLOR_BayerGB2RGB
            bayer_map = {(1, 1): 4, (0, 0): 3, (1, 0): 1, (0, 1): 2}
            self.g['GPU_BAYER'] = bayer_map.get((flip_v, flip_h), 3)
            if is_camera and camera:
                flip_val = 3 if (flip_v and flip_h) else (2 if flip_v else (1 if flip_h else 0))
                camera.set_control_value(asi.ASI_FLIP, flip_val)
            else:
                if flip_v and flip_h:
                    self.g['type_flip'] = "both"
                elif flip_v:
                    self.g['type_flip'] = "vertical"
                elif flip_h:
                    self.g['type_flip'] = "horizontal"
                else:
                    self.g['type_flip'] = "none"

    def commande_flipV(self):
        """Toggle vertical flip."""
        self.g['reset_FS']()
        self.g['FlipV'] = 1 if self.g['choix_flipV'].get() != 0 else 0
        flip_v = self.g['FlipV']
        flip_h = self.g['FlipH']

        if self.g.get('flag_camera_ok', False):
            bayer = self.g.get('Camera_Bayer', 'RAW')
            self._apply_bayer_flip(bayer, flip_v, flip_h, is_camera=True)
        else:
            bayer = self.g.get('Video_Bayer', 'RAW')
            self._apply_bayer_flip(bayer, flip_v, flip_h, is_camera=False)

        # Sync to display state
        if self.state is not None and self.state.display is not None:
            self.state.display.FlipV = self.g['FlipV']
            self.state.display.FlipH = self.g['FlipH']
            self.state.display.type_flip = self.g.get('type_flip', 'none')

    def commande_flipH(self):
        """Toggle horizontal flip."""
        self.g['reset_FS']()
        self.g['FlipH'] = 1 if self.g['choix_flipH'].get() != 0 else 0
        flip_v = self.g['FlipV']
        flip_h = self.g['FlipH']

        if self.g.get('flag_camera_ok', False):
            bayer = self.g.get('Camera_Bayer', 'RAW')
            self._apply_bayer_flip(bayer, flip_v, flip_h, is_camera=True)
        else:
            bayer = self.g.get('Video_Bayer', 'RAW')
            self._apply_bayer_flip(bayer, flip_v, flip_h, is_camera=False)

        # Sync to display state
        if self.state is not None and self.state.display is not None:
            self.state.display.FlipV = self.g['FlipV']
            self.state.display.FlipH = self.g['FlipH']
            self.state.display.type_flip = self.g.get('type_flip', 'none')

    # =========================================================================
    # Stacking Callbacks
    # =========================================================================

    def choix_mean_stacking(self, event=None):
        """Set mean stacking mode."""
        self.g['flag_stacking'] = "Mean"
        self.g['stack_div'] = self.g.get('val_FS', 1)

    def choix_sum_stacking(self, event=None):
        """Set sum stacking mode."""
        self.g['flag_stacking'] = "Sum"
        self.g['stack_div'] = 1

    def choix_dyn_high(self, event=None):
        """Set AANR dynamic range to high."""
        self.g['flag_dyn_AANR'] = 1

    def choix_dyn_low(self, event=None):
        """Set AANR dynamic range to low."""
        self.g['flag_dyn_AANR'] = 0

    def choix_FS(self, event=None):
        """Handle frame stacking count change."""
        val_FS = self.g['echelle20'].get()
        self.g['val_FS'] = val_FS
        self.g['compteur_FS'] = 0
        self.g['Im1OK'] = False
        self.g['Im2OK'] = False
        self.g['Im3OK'] = False
        self.g['Im4OK'] = False
        self.g['Im5OK'] = False
        if self.g.get('flag_stacking') == "Mean":
            self.g['stack_div'] = val_FS
        else:
            self.g['stack_div'] = 1

    def reset_FS(self, event=None):
        """Reset frame stacking state."""
        self.g['val_FS'] = 1
        self.g['compteur_FS'] = 0
        self.g['Im1OK'] = False
        self.g['Im2OK'] = False
        self.g['Im3OK'] = False
        self.g['Im4OK'] = False
        self.g['Im5OK'] = False
        if 'echelle20' in self.g:
            self.g['echelle20'].set(1)

    def command_BFReference(self, event=None):
        """Set best frame reference mode."""
        self.g['flag_BFReference'] = "BestFrame"

    def command_PFReference(self, event=None):
        """Set previous frame reference mode."""
        self.g['flag_BFReference'] = "PreviousFrame"

    # =========================================================================
    # Sensor Ratio Callbacks
    # =========================================================================

    def choix_sensor_ratio_4_3(self, event=None):
        """Set sensor aspect ratio to 4:3."""
        if self.g.get('flag_camera_ok', False):
            self.g['sensor_factor'] = "4_3"
            fact_s = self.g.get('fact_s', 1.0)
            self.g['cam_displ_x'] = int(1350 * fact_s)
            self.g['cam_displ_y'] = int(1012 * fact_s)
            self.g['RES_X_BIN1'] = self.g['RES_X_BIN1_4_3']
            self.g['RES_Y_BIN1'] = self.g['RES_Y_BIN1_4_3']
            self.g['RES_X_BIN2'] = self.g['RES_X_BIN2_4_3']
            self.g['RES_Y_BIN2'] = self.g['RES_Y_BIN2_4_3']
            self.choix_resolution_camera()

    def choix_sensor_ratio_16_9(self, event=None):
        """Set sensor aspect ratio to 16:9."""
        if self.g.get('flag_camera_ok', False):
            self.g['sensor_factor'] = "16_9"
            fact_s = self.g.get('fact_s', 1.0)
            self.g['cam_displ_x'] = int(1350 * fact_s)
            self.g['cam_displ_y'] = int(760 * fact_s)
            self.g['RES_X_BIN1'] = self.g['RES_X_BIN1_16_9']
            self.g['RES_Y_BIN1'] = self.g['RES_Y_BIN1_16_9']
            self.g['RES_X_BIN2'] = self.g['RES_X_BIN2_16_9']
            self.g['RES_Y_BIN2'] = self.g['RES_Y_BIN2_16_9']
            self.choix_resolution_camera()

    def choix_sensor_ratio_1_1(self, event=None):
        """Set sensor aspect ratio to 1:1."""
        if self.g.get('flag_camera_ok', False):
            self.g['sensor_factor'] = "1_1"
            fact_s = self.g.get('fact_s', 1.0)
            self.g['cam_displ_x'] = int(1012 * fact_s)
            self.g['cam_displ_y'] = int(1012 * fact_s)
            self.g['RES_X_BIN1'] = self.g['RES_X_BIN1_1_1']
            self.g['RES_Y_BIN1'] = self.g['RES_Y_BIN1_1_1']
            self.g['RES_X_BIN2'] = self.g['RES_X_BIN2_1_1']
            self.g['RES_Y_BIN2'] = self.g['RES_Y_BIN2_1_1']
            self.choix_resolution_camera()

    # =========================================================================
    # Bayer Callbacks
    # =========================================================================

    def choix_bayer_RAW(self, event=None):
        """Set Bayer pattern to RAW (no debayering)."""
        self.g['type_debayer'] = 0
        self.g['Camera_Bayer'] = "RAW"
        self.g['Video_Bayer'] = "RAW"
        self.g['GPU_BAYER'] = 0
        self.g['choix_flipV'].set(0)
        self.g['choix_flipH'].set(0)
        self.commande_flipV()

    def choix_bayer_RGGB(self, event=None):
        """Set Bayer pattern to RGGB."""
        self.g['type_debayer'] = cv2.COLOR_BayerBG2RGB
        self.g['Camera_Bayer'] = "RGGB"
        self.g['Video_Bayer'] = "RGGB"
        self.g['GPU_BAYER'] = 1
        self.g['choix_flipV'].set(0)
        self.g['choix_flipH'].set(0)
        self.commande_flipV()

    def choix_bayer_BGGR(self, event=None):
        """Set Bayer pattern to BGGR."""
        self.g['type_debayer'] = cv2.COLOR_BayerRG2RGB
        self.g['Camera_Bayer'] = "BGGR"
        self.g['Video_Bayer'] = "BGGR"
        self.g['GPU_BAYER'] = 2
        self.g['choix_flipV'].set(0)
        self.g['choix_flipH'].set(0)
        self.commande_flipV()

    def choix_bayer_GBRG(self, event=None):
        """Set Bayer pattern to GBRG."""
        self.g['type_debayer'] = cv2.COLOR_BayerGB2RGB
        self.g['Camera_Bayer'] = "GBRG"
        self.g['Video_Bayer'] = "GBRG"
        self.g['GPU_BAYER'] = 3
        self.g['choix_flipV'].set(0)
        self.g['choix_flipH'].set(0)
        self.commande_flipV()

    def choix_bayer_GRBG(self, event=None):
        """Set Bayer pattern to GRBG."""
        self.g['type_debayer'] = cv2.COLOR_BayerGR2RGB
        self.g['Camera_Bayer'] = "GRBG"
        self.g['Video_Bayer'] = "GRBG"
        self.g['GPU_BAYER'] = 4
        self.g['choix_flipV'].set(0)
        self.g['choix_flipH'].set(0)
        self.commande_flipV()

    # =========================================================================
    # Filter Wheel Callbacks
    # =========================================================================

    def choix_position_EFW0(self, event=None):
        """Set filter wheel to position 0."""
        if self.g.get('flag_camera_ok', False) and self.g.get('flag_filter_wheel', False):
            self.g['fw_position'] = 0
            self.g['filter_wheel'].set_position(0)

    def choix_position_EFW1(self, event=None):
        """Set filter wheel to position 1."""
        if self.g.get('flag_camera_ok', False) and self.g.get('flag_filter_wheel', False):
            self.g['fw_position'] = 1
            self.g['filter_wheel'].set_position(1)

    def choix_position_EFW2(self, event=None):
        """Set filter wheel to position 2."""
        if self.g.get('flag_camera_ok', False) and self.g.get('flag_filter_wheel', False):
            self.g['fw_position'] = 2
            self.g['filter_wheel'].set_position(2)

    def choix_position_EFW3(self, event=None):
        """Set filter wheel to position 3."""
        if self.g.get('flag_camera_ok', False) and self.g.get('flag_filter_wheel', False):
            self.g['fw_position'] = 3
            self.g['filter_wheel'].set_position(3)

    def choix_position_EFW4(self, event=None):
        """Set filter wheel to position 4."""
        if self.g.get('flag_camera_ok', False) and self.g.get('flag_filter_wheel', False):
            self.g['fw_position'] = 4
            self.g['filter_wheel'].set_position(4)

    def commande_FW(self):
        """Filter wheel command placeholder."""
        time.sleep(0.01)

    # =========================================================================
    # Tracking Callbacks
    # =========================================================================

    def commande_TRKSAT(self):
        """Toggle satellite tracking."""
        if self.g['choix_TRKSAT'].get() == 0:
            self.g['flag_TRKSAT'] = 0
        else:
            self.g['flag_TRKSAT'] = 1
            self.g['flag_nouvelle_resolution'] = True
        self.g['flag_first_sat_pass'] = True
        self.g['sat_frame_count'] = 0
        self.g['flag_img_sat_buf1'] = False
        self.g['flag_img_sat_buf2'] = False
        self.g['flag_img_sat_buf3'] = False
        self.g['flag_img_sat_buf4'] = False
        self.g['flag_img_sat_buf5'] = False

    def commande_CONST(self):
        """Toggle constellation display."""
        if self.g['choix_CONST'].get() == 0:
            self.g['flag_CONST'] = 0
        else:
            self.g['flag_CONST'] = 1
            self.g['flag_nouvelle_resolution'] = True

    def commande_REMSAT(self):
        """Toggle satellite removal."""
        if self.g['choix_REMSAT'].get() == 0:
            self.g['flag_REMSAT'] = 0
        else:
            self.g['flag_REMSAT'] = 1
            self.g['flag_nouvelle_resolution'] = True

    def commande_TRIGGER(self):
        """Toggle trigger mode."""
        if self.g['choix_TRIGGER'].get() == 0:
            self.g['flag_TRIGGER'] = 0
        else:
            self.g['flag_TRIGGER'] = 1

    def commande_BFR(self):
        """Toggle best frame recording."""
        if self.g['choix_BFR'].get() == 0:
            self.g['flag_BFR'] = False
        else:
            self.g['flag_BFR'] = True
        self.g['max_qual'] = 0
        self.g['min_qual'] = 10000
        self.g['val_BFR'] = 50
        self.g['echelle300'].set(50)
        self.g['SFN'] = 0
        self.g['frame_number'] = 0
        self.g['labelInfo10'].config(text="                                             ")

    def commande_false_colours(self):
        """Toggle false colors display."""
        if self.g['choix_false_colours'].get() == 0:
            self.g['flag_false_colours'] = False
        else:
            self.g['flag_false_colours'] = True

    def commande_AI_Craters(self):
        """Toggle AI crater detection."""
        if self.g['choix_AI_Craters'].get() == 0:
            if self.g.get('flag_crater_model_loaded', False):
                try:
                    self.g['model_craters_track'].predictor.trackers[0].reset()
                except:
                    pass
            self.g['track_crater_history'] = defaultdict(lambda: [])
            self.g['flag_AI_Craters'] = False
        else:
            if self.g.get('flag_crater_model_loaded', False):
                try:
                    self.g['model_craters_track'].predictor.trackers[0].reset()
                except:
                    pass
            self.g['track_crater_history'] = defaultdict(lambda: [])
            self.g['flag_AI_Craters'] = True

    def commande_AI_Satellites(self):
        """Toggle AI satellite detection."""
        if self.g['choix_AI_Satellites'].get() == 0:
            if self.g.get('flag_satellites_model_loaded', False):
                try:
                    self.g['model_satellites_track'].predictor.trackers[0].reset()
                except:
                    pass
            self.g['track_satellite_history'] = defaultdict(lambda: [])
            self.g['flag_AI_Satellites'] = False
        else:
            if self.g.get('flag_satellites_model_loaded', False):
                try:
                    self.g['model_satellites_track'].predictor.trackers[0].reset()
                except:
                    pass
            self.g['track_satellite_history'] = defaultdict(lambda: [])
            self.g['flag_AI_Satellites'] = True
        self.g['sat_frame_count_AI'] = 0
        self.g['flag_img_sat_buf1_AI'] = False
        self.g['flag_img_sat_buf2_AI'] = False
        self.g['flag_img_sat_buf3_AI'] = False
        self.g['flag_img_sat_buf4_AI'] = False
        self.g['flag_img_sat_buf5_AI'] = False
        self.g['flag_first_sat_pass_AI'] = True

    def commande_AI_Trace(self):
        """Toggle AI tracking trace display."""
        if self.g['choix_AI_Trace'].get() == 0:
            self.g['flag_AI_Trace'] = False
        else:
            self.g['flag_AI_Trace'] = True
        try:
            self.g['model_satellites_track'].predictor.trackers[0].reset()
            self.g['model_craters_track'].predictor.trackers[0].reset()
        except:
            pass
        self.g['track_satellite_history'] = defaultdict(lambda: [])
        self.g['track_crater_history'] = defaultdict(lambda: [])

    # =========================================================================
    # Misc Callbacks
    # =========================================================================

    def choix_demo_left(self, event=None):
        """Set demo mode display to left side."""
        self.g['flag_demo_side'] = "Left"

    def choix_demo_right(self, event=None):
        """Set demo mode display to right side."""
        self.g['flag_demo_side'] = "Right"

    def choix_SAT_Vid(self):
        """Set satellite capture to video mode."""
        self.g['flag_SAT_Image'] = False

    def choix_SAT_Img(self):
        """Set satellite capture to image mode."""
        self.g['flag_SAT_Image'] = True

    def commande_autoexposure(self):
        """Toggle auto-exposure mode."""
        if self.g['choix_autoexposure'].get() == 0:
            self.g['flag_autoexposure_exposition'] = False
            if self.g.get('flag_camera_ok', False):
                camera = self.g['camera']
                asi = self.g['asi']
                camera.set_control_value(asi.ASI_EXPOSURE, self.g['exposition'], auto=False)
        else:
            self.g['flag_autoexposure_exposition'] = True
            if self.g.get('flag_camera_ok', False):
                camera = self.g['camera']
                asi = self.g['asi']
                camera.set_control_value(asi.ASI_EXPOSURE, self.g['exposition'], auto=True)
                camera.set_control_value(asi.ASI_AUTO_MAX_EXP, 250)  # 250ms max

    def commande_autogain(self):
        """Toggle auto-gain mode."""
        if self.g['choix_autogain'].get() == 0:
            self.g['flag_autoexposure_gain'] = False
            if self.g.get('flag_camera_ok', False):
                camera = self.g['camera']
                asi = self.g['asi']
                camera.set_control_value(asi.ASI_GAIN, self.g['val_gain'], auto=False)
        else:
            self.g['flag_autoexposure_gain'] = True
            if self.g.get('flag_camera_ok', False):
                camera = self.g['camera']
                asi = self.g['asi']
                camera.set_control_value(asi.ASI_GAIN, self.g['val_gain'], auto=True)
                camera.set_control_value(asi.ASI_AUTO_MAX_GAIN, self.g['val_maxgain'])

    def commande_16bLL(self):
        """Toggle 16-bit low light mode."""
        if self.g.get('flag_camera_ok', False):
            if not self.g.get('flag_cap_video', False):
                if self.g['choix_16bLL'].get() == 0:
                    self.g['flag_16b'] = False
                else:
                    self.g['flag_16b'] = True

    def commande_IMQE(self):
        """Toggle image quality estimation."""
        if self.g['choix_IMQE'].get() == 0:
            self.g['flag_IQE'] = False
        else:
            self.g['flag_IQE'] = True
            for x in range(1, 258):
                self.g['quality'][x] = 0
            self.g['max_quality'] = 1
            self.g['quality_pos'] = 1

    def choix_nb_captures(self, event=None):
        """Handle number of captures change."""
        self.g['val_nb_captures'] = self.g['echelle8'].get()

    def choix_deltat(self, event=None):
        """Handle delta time change."""
        self.g['val_deltat'] = self.g['echelle65'].get()

    def choix_nb_video(self, event=None):
        """Handle number of video captures change."""
        self.g['val_nb_capt_video'] = self.g['echelle11'].get()

    def choix_position_frame(self, event=None):
        """Handle video frame position change."""
        if not self.g.get('flag_cap_video', False) and not self.g.get('flag_image_mode', False):
            self.g['flag_new_frame_position'] = True
            video_frame_position = self.g['echelle210'].get()
            self.g['video_frame_position'] = video_frame_position
            if not self.g.get('flag_SER_file', False):
                self.g['video'].set(cv2.CAP_PROP_POS_FRAMES, video_frame_position)
            else:
                self.g['video'].setCurrentPosition(video_frame_position)

    def raz_framecount(self):
        """Reset frame counter."""
        self.g['frame_number'] = 0

    def raz_tracking(self):
        """Reset tracking state."""
        self.g['flag_nouvelle_resolution'] = True

    def stop_tracking(self):
        """Stop all tracking modes."""
        self.g['flag_TRKSAT'] = 0
        self.g['flag_DETECT_STARS'] = 0
        self.g['flag_nouvelle_resolution'] = True
        self.g['choix_DETECT_STARS'].set(0)
        self.g['choix_TRKSAT'].set(0)

    def Capture_Ref_Img(self):
        """Capture reference image."""
        self.g['flag_capture_image_reference'] = True

    def commande_sub_img_ref(self):
        """Toggle image reference subtraction."""
        if self.g['choix_sub_img_ref'].get() == 0:
            self.g['flag_image_ref_sub'] = False
        else:
            self.g['flag_image_ref_sub'] = True

    def commande_Blur_img_ref(self):
        """Toggle blur on image reference subtraction."""
        if self.g['choix_Blur_img_ref'].get() == 0:
            self.g['flag_blur_image_ref_sub'] = False
        else:
            self.g['flag_blur_image_ref_sub'] = True

    def commande_GBL(self):
        """Toggle Gaussian blur."""
        if self.g['choix_GBL'].get() == 0:
            self.g['flag_GaussBlur'] = False
        else:
            self.g['flag_GaussBlur'] = True

    def reset_general_FS(self):
        """Reset all filter states to defaults."""
        # Reference subtraction
        self.g['choix_sub_img_ref'].set(0)
        self.g['flag_capture_image_reference'] = False
        self.g['flag_image_reference_OK'] = False
        self.g['flag_image_ref_sub'] = False
        self.g['flag_image_disponible'] = False

        # Frame stacking
        self.g['val_FS'] = 1
        self.g['compteur_FS'] = 0
        self.g['Im1OK'] = False
        self.g['Im2OK'] = False
        self.g['Im3OK'] = False
        self.g['Im4OK'] = False
        self.g['Im5OK'] = False
        self.g['echelle20'].set(1)

        # AANR filters
        self.g['choix_AANR'].set(0)
        self.g['choix_AANRB'].set(0)
        self.g['flag_AANR'] = False
        self.g['compteur_AANR'] = 0
        self.g['Im1fsdnOK'] = False
        self.g['Im2fsdnOK'] = False
        self.g['flag_AANRB'] = False
        self.g['compteur_AANRB'] = 0
        self.g['Im1fsdnOKB'] = False
        self.g['Im2fsdnOKB'] = False

        # Stabilization
        self.g['flag_STAB'] = False
        self.g['flag_Template'] = False
        self.g['choix_STAB'].set(0)

        # Reduce variation
        self.g['compteur_RV'] = 0
        self.g['Im1rvOK'] = False
        self.g['Im2rvOK'] = False
        self.g['flag_reduce_variation'] = False
        self.g['choix_reduce_variation'].set(0)

        # 3FNR filters
        self.g['choix_3FNR'].set(0)
        self.g['flag_3FNR'] = False
        self.g['compteur_3FNR'] = 0
        self.g['img1_3FNROK'] = False
        self.g['img2_3FNROK'] = False
        self.g['img3_3FNROK'] = False
        self.g['FNR_First_Start'] = True

        self.g['choix_3FNRB'].set(0)
        self.g['flag_3FNRB'] = False
        self.g['compteur_3FNRB'] = 0
        self.g['img1_3FNRBOK'] = False
        self.g['img2_3FNRBOK'] = False
        self.g['img3_3FNRBOK'] = False
        self.g['FNRB_First_Start'] = True

        self.g['choix_3FNR2'].set(0)
        self.g['flag_3FNR2'] = False
        self.g['compteur_3FNR2'] = 0
        self.g['img1_3FNR2OK'] = False
        self.g['img2_3FNR2OK'] = False
        self.g['img3_3FNR2OK'] = False
        self.g['FNR2_First_Start'] = True

        self.g['choix_3FNR2B'].set(0)
        self.g['flag_3FNR2B'] = False
        self.g['compteur_3FNR2B'] = 0
        self.g['img1_3FNR2BOK'] = False
        self.g['img2_3FNR2BOK'] = False
        self.g['img3_3FNR2BOK'] = False
        self.g['FNR2B_First_Start'] = True

        # Best frame recording
        self.g['flag_BFR'] = False
        self.g['choix_BFR'].set(0)
        self.g['choix_IMQE'].set(0)
        self.g['flag_IQE'] = False

        # RGB shifting
        self.g['delta_RX'] = 0
        self.g['delta_RY'] = 0
        self.g['delta_BX'] = 0
        self.g['delta_BY'] = 0

        # Satellite tracking
        self.g['flag_first_sat_pass'] = True
        self.g['flag_first_sat_pass_AI'] = True
        self.g['flag_img_sat_buf1'] = False
        self.g['flag_img_sat_buf2'] = False
        self.g['flag_img_sat_buf3'] = False
        self.g['flag_img_sat_buf4'] = False
        self.g['flag_img_sat_buf5'] = False
        self.g['flag_img_sat_buf1_AI'] = False
        self.g['flag_img_sat_buf2_AI'] = False
        self.g['flag_img_sat_buf3_AI'] = False
        self.g['flag_img_sat_buf4_AI'] = False
        self.g['flag_img_sat_buf5_AI'] = False
        self.g['sat_frame_count'] = 0
        self.g['sat_frame_count_AI'] = 0
        self.g['nb_sat'] = -1

        self.g['labelInfo10'].config(text=" ")

        # Tracking flags
        self.g['flag_CONST'] = 0
        self.g['choix_CONST'].set(0)
        self.g['flag_TRKSAT'] = 0
        self.g['choix_TRKSAT'].set(0)
        self.g['flag_REMSAT'] = 0
        self.g['choix_REMSAT'].set(0)
        self.g['flag_DETECT_STARS'] = 0
        self.g['choix_DETECT_STARS'].set(0)

        time.sleep(0.5)

        # AI detection
        self.g['flag_AI_Craters'] = False
        self.g['choix_AI_Craters'].set(0)
        self.g['flag_reduce_variation_post_treatment'] = False
        self.g['flag_BFREFPT'] = False
        self.g['flag_BFREF_image'] = False
        self.g['max_qual_PT'] = 0
        self.g['choix_HOTPIX'].set(0)
        self.g['flag_hot_pixels'] = False
        self.g['choix_HDR'].set(0)
        self.g['flag_HDR'] = False

        try:
            self.g['model_craters_track'].predictor.trackers[0].reset()
            self.g['model_satellites_track'].predictor.trackers[0].reset()
        except:
            pass

        self.g['track_crater_history'] = defaultdict(lambda: [])
        self.g['track_satelitte_history'] = defaultdict(lambda: [])
        self.g['flag_AI_Craters'] = False
        self.g['choix_AI_Satellites'].set(0)
        self.g['flag_AI_Satellites'] = False

        time.sleep(0.2)


# Module-level function for backward compatibility
def init_gui_callbacks(main_globals: dict, app_state=None) -> GUICallbacks:
    """
    Initialize GUI callbacks and register them into the main namespace.

    This function creates a GUICallbacks instance and registers all callbacks
    into the main globals namespace, providing a drop-in replacement for the
    exec() pattern.

    Args:
        main_globals: globals() dictionary from main script
        app_state: ApplicationState instance (optional, for state sync)

    Returns:
        GUICallbacks instance (for reference, callbacks are already registered)

    Usage:
        from gui_callbacks_class import init_gui_callbacks

        gui_callbacks_obj = init_gui_callbacks(globals(), app_state)
    """
    callbacks = GUICallbacks(main_globals, app_state)
    callbacks.register_all()
    return callbacks
