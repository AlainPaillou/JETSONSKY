"""
Filter Pipeline Module for JetsonSky (Refactored)

This module contains the GPU-accelerated image filtering pipeline:
- FilterState: Dataclass holding mutable state across frames
- FilterParams: Dataclass holding filter configuration flags/values
- FilterKernels: Dataclass holding references to CUDA kernels
- FilterPipeline: Class containing filter processing methods

The filter pipeline processes frames in this fixed order:
1. RGB software adjustment
2. Image negative
3. Luminance estimate (mono from color)
4. 2-5 images SUM or MEAN stacking
5. Reduce consecutive variation
6. 3FNR1 front (3-frame noise removal)
7. AANR front (Adaptive Absorber Noise Removal)
8. NR P1 (Noise Removal Paillou 1)
9. NR P2 (Noise Removal Paillou 2)
10. NLM2 (Non-Local Means 2)
11. KNN (K-Nearest Neighbors)
12. Luminance adjust
13. Image amplification (Linear/Gaussian)
14. Star amplification
15. Gradient/vignetting management
16. CLL (Contrast Low Light)
17. CLAHE contrast
18. Color saturation enhancement (2-pass option)
19. 3FNR2 back
20. AANR back (High dynamic only)
21. Sharpen 1
22. Sharpen 2

Copyright Alain Paillou 2018-2025
"""

from dataclasses import dataclass, field
from typing import Optional, Any, List
import numpy as np

# These will be imported when module is loaded
# import cupy as cp
# import cv2


@dataclass
class FilterState:
    """
    Holds mutable state that persists across frames for temporal filters.

    This includes frame buffers for noise reduction, stacking, and other
    multi-frame operations.
    """
    # Frame stacking state (2-5 frame sum/mean)
    compteur_FS: int = 0
    Im1OK: bool = False
    Im2OK: bool = False
    Im3OK: bool = False
    Im4OK: bool = False
    Im5OK: bool = False

    # Stacking buffers - color (will be CuPy arrays)
    b1_sm: Any = None
    b2_sm: Any = None
    b3_sm: Any = None
    b4_sm: Any = None
    b5_sm: Any = None
    g1_sm: Any = None
    g2_sm: Any = None
    g3_sm: Any = None
    g4_sm: Any = None
    g5_sm: Any = None
    r1_sm: Any = None
    r2_sm: Any = None
    r3_sm: Any = None
    r4_sm: Any = None
    r5_sm: Any = None

    # Reduce variation state
    compteur_RV: int = 0
    Im1rvOK: bool = False
    Im2rvOK: bool = False
    res_b2: Any = None
    res_g2: Any = None
    res_r2: Any = None

    # AANR front state
    compteur_AANR: int = 0
    Im1fsdnOK: bool = False
    Im2fsdnOK: bool = False

    # AANR back state
    compteur_AANRB: int = 0
    Im1fsdnOKB: bool = False
    Im2fsdnOKB: bool = False
    res_b2B: Any = None
    res_g2B: Any = None
    res_r2B: Any = None

    # 3FNR front state
    compteur_3FNR: int = 0
    img1_3FNROK: bool = False
    img2_3FNROK: bool = False
    img3_3FNROK: bool = False
    FNR_First_Start: bool = True
    imgb1: Any = None
    imgg1: Any = None
    imgr1: Any = None
    imgb2: Any = None
    imgg2: Any = None
    imgr2: Any = None
    imgb3: Any = None
    imgg3: Any = None
    imgr3: Any = None

    # 3FNR back state
    compteur_3FNRB: int = 0
    img1_3FNROKB: bool = False
    img2_3FNROKB: bool = False
    img3_3FNROKB: bool = False
    FNRB_First_Start: bool = True
    imgb1B: Any = None
    imgg1B: Any = None
    imgr1B: Any = None
    imgb2B: Any = None
    imgg2B: Any = None
    imgr2B: Any = None
    imgb3B: Any = None
    imgg3B: Any = None
    imgr3B: Any = None

    # 3FNR2 front state
    compteur_3FNR2: int = 0
    img1_3FNR2OK: bool = False
    img2_3FNR2OK: bool = False
    img3_3FNR2OK: bool = False
    FNR2_First_Start: bool = True
    imgb21: Any = None
    imgg21: Any = None
    imgr21: Any = None
    imgb22: Any = None
    imgg22: Any = None
    imgr22: Any = None
    imgb23: Any = None
    imgg23: Any = None
    imgr23: Any = None

    # 3FNR2 back state
    compteur_3FNR2B: int = 0
    img1_3FNR2OKB: bool = False
    img2_3FNR2OKB: bool = False
    img3_3FNR2OKB: bool = False
    FNR2B_First_Start: bool = True
    imgb21B: Any = None
    imgg21B: Any = None
    imgr21B: Any = None
    imgb22B: Any = None
    imgg22B: Any = None
    imgr22B: Any = None
    imgb23B: Any = None
    imgg23B: Any = None
    imgr23B: Any = None

    # Best frame reference state
    BFREF_image: Any = None
    flag_BFREF_image: bool = False
    BFREF_image_PT: Any = None
    flag_BFREF_image_PT: bool = False
    max_qual_PT: int = 0

    # Transfer function arrays
    trsf_r: Any = None
    trsf_g: Any = None
    trsf_b: Any = None

    # Timing queue
    TTQueue: List[float] = field(default_factory=list)
    curTT: float = 0.0

    # Output
    image_traitee: Any = None
    time_exec_test: int = 0

    def reset_stacking(self):
        """Reset frame stacking state"""
        self.compteur_FS = 0
        self.Im1OK = False
        self.Im2OK = False
        self.Im3OK = False
        self.Im4OK = False
        self.Im5OK = False

    def reset_3fnr_front(self):
        """Reset 3FNR front state"""
        self.compteur_3FNR = 0
        self.img1_3FNROK = False
        self.img2_3FNROK = False
        self.img3_3FNROK = False
        self.FNR_First_Start = True

    def reset_3fnr_back(self):
        """Reset 3FNR back state"""
        self.compteur_3FNRB = 0
        self.img1_3FNROKB = False
        self.img2_3FNROKB = False
        self.img3_3FNROKB = False
        self.FNRB_First_Start = True

    def reset_aanr(self):
        """Reset AANR state"""
        self.compteur_AANR = 0
        self.Im1fsdnOK = False
        self.Im2fsdnOK = False
        self.compteur_AANRB = 0
        self.Im1fsdnOKB = False
        self.Im2fsdnOKB = False


@dataclass
class FilterParams:
    """
    Holds filter configuration parameters (flags and values).

    These are typically set by GUI controls and don't change during
    a single frame's processing.
    """
    # General flags
    flag_filtrage_ON: bool = True
    flag_IsColor: bool = True
    flag_image_mode: bool = False
    flag_TRSF: int = 0
    flag_DEMO: int = 0
    flag_demo_side: str = "Left"
    flag_reverse_RB: int = 0

    # Gaussian blur
    flag_GaussBlur: bool = False

    # RGB adjustment
    val_reds: float = 1.0
    val_greens: float = 1.0
    val_blues: float = 1.0

    # Image negative
    ImageNeg: int = 0

    # Luminance estimation
    flag_NB_estime: int = 0

    # Frame stacking
    val_FS: int = 1
    stack_div: int = 1  # 1=sum, 0=median

    # Reduce variation
    flag_reduce_variation: bool = False
    val_reduce_variation: float = 50.0
    flag_BFReference: str = "PreviousFrame"  # "PreviousFrame" or "BestFrame"

    # 3FNR
    flag_3FNR: bool = False
    flag_3FNRB: bool = False
    flag_3FNR2: bool = False
    flag_3FNR2B: bool = False
    val_3FNR_Thres: float = 1.0
    val_3FNR_Threshold: float = 1.0  # Alias used by GUI callbacks

    # AANR
    flag_AANR: bool = False
    flag_AANRB: bool = False
    flag_dyn_AANR: int = 0
    flag_ghost_reducer: bool = False  # GUI callbacks use bool
    val_ghost_reducer: int = 0

    # Denoise Paillou
    flag_denoise_Paillou: int = 0
    flag_denoise_Paillou2: int = 0

    # NLM2
    flag_NLM2: int = 0
    val_denoise: float = 10.0

    # KNN
    flag_denoise_KNN: int = 0
    flag_KNN: bool = False  # Alias used by GUI callbacks
    val_denoise_KNN: float = 10.0

    # Best frame reference progressive tracking
    flag_BFREFPT: bool = False

    # Histogram operations
    flag_histogram_stretch: bool = False
    flag_histogram_equalize: bool = False  # Used by GUI callbacks
    flag_histogram_equalize2: int = 0  # Deprecated alias
    flag_histogram_phitheta: bool = False
    val_histo_min: float = 0.0
    val_histo_max: float = 255.0
    val_heq2: float = 1.0
    val_phi: float = 1.0
    val_theta: float = 128.0

    # Amplification
    flag_AmpSoft: int = 0
    flag_lin_gauss: int = 1  # 1=linear, 2=gaussian, 3=star
    val_ampl: float = 1.0
    val_Mu: float = 0.5
    val_Ro: float = 0.5
    Corr_GS: Any = None  # Correction curve

    # Gradient removal
    flag_GR: bool = False
    grad_vignet: int = 1  # 1=vignetting, 0=gradient
    val_SGR: float = 50.0
    val_NGB: int = 5
    val_AGR: float = 50.0

    # Contrast Low Light
    flag_CLL: int = 0
    Corr_CLL: Any = None  # CLL correction curve

    # CLAHE
    flag_contrast_CLAHE: int = 0
    val_contrast_CLAHE: float = 2.0
    val_grid_CLAHE: int = 8
    flag_OpenCvCuda: bool = False

    # Saturation
    flag_SAT: bool = False
    flag_SAT_Image: bool = False
    flag_SAT2PASS: bool = False
    val_SAT: float = 1.0

    # Sharpen
    flag_sharpen_soft1: int = 0
    flag_sharpen_soft2: int = 0
    val_sharpen: float = 1.0
    val_sharpen2: float = 1.0
    val_sigma_sharpen: int = 3
    val_sigma_sharpen2: int = 3

    # Camera resolution (for BFREFPT calculations)
    res_cam_x: int = 1920
    res_cam_y: int = 1080
    delta_tx: int = 0
    delta_ty: int = 0

    # Image quality method
    IQ_Method: int = 0

    # Thread configuration
    nb_ThreadsX: int = 32
    nb_ThreadsY: int = 32


@dataclass
class FilterKernels:
    """
    Holds references to CUDA kernels used by the filter pipeline.

    These are compiled CuPy RawKernel objects that perform the actual
    GPU-accelerated image processing.
    """
    # RGB operations
    Set_RGB: Any = None
    color_estimate_Mono: Any = None
    grey_estimate_Mono: Any = None

    # Variation reduction
    reduce_variation_Color: Any = None
    reduce_variation_Mono: Any = None

    # 3FNR
    FNR_Color: Any = None
    FNR_Mono: Any = None
    FNR2_Color: Any = None
    FNR2_Mono: Any = None

    # AANR
    adaptative_absorber_denoise_Color: Any = None
    adaptative_absorber_denoise_Mono: Any = None

    # Denoise Paillou
    Denoise_Paillou_Colour: Any = None
    Denoise_Paillou_Mono: Any = None
    reduce_noise_Color: Any = None
    reduce_noise_Mono: Any = None

    # NLM2
    NLM2_Colour_GPU: Any = None
    NLM2_Mono_GPU: Any = None

    # KNN
    KNN_Colour_GPU: Any = None
    KNN_Mono_GPU: Any = None

    # Histogram
    Histo_Color: Any = None
    Histo_Mono: Any = None

    # Amplification
    Colour_ampsoft_GPU: Any = None
    Mono_ampsoft_GPU: Any = None
    Colour_staramp_GPU: Any = None
    Mono_staramp_GPU: Any = None

    # Contrast
    Contrast_Low_Light_Colour_GPU: Any = None
    Contrast_Low_Light_Mono_GPU: Any = None

    # Saturation
    Saturation_Colour: Any = None
    Saturation_Combine_Colour: Any = None


class FilterPipeline:
    """
    GPU-accelerated image filter pipeline for astronomy imaging.

    This class encapsulates all filter operations and manages state
    for temporal filters (noise reduction, stacking, etc.).

    Usage:
        # Initialize with dependencies
        pipeline = FilterPipeline(cp, cv2, kernels, utility_funcs)

        # Process a color frame
        result = pipeline.process_color(res_b, res_g, res_r, params, state)

        # Process a mono frame
        result = pipeline.process_mono(res_b, params, state)
    """

    def __init__(self, cp_module, cv2_module, kernels: FilterKernels,
                 utility_funcs: dict):
        """
        Initialize the filter pipeline.

        Args:
            cp_module: CuPy module for GPU operations
            cv2_module: OpenCV module for image operations
            kernels: FilterKernels instance with compiled CUDA kernels
            utility_funcs: Dictionary of utility functions:
                - gaussianblur_colour: (b, g, r, size) -> (b, g, r)
                - gaussianblur_mono: (img, size) -> img
                - image_negative_colour: (b, g, r) -> (b, g, r)
                - cupy_separateRGB_2_numpy_RGBimage: (b, g, r) -> numpy_rgb
                - numpy_RGBImage_2_cupy_separateRGB: (numpy_rgb) -> (r, g, b)
                - Image_Quality: (img, method) -> quality_score
        """
        self.cp = cp_module
        self.cv2 = cv2_module
        self.kernels = kernels
        self.utils = utility_funcs

    def _compute_grid_dims(self, width: int, height: int,
                           threads_x: int = 32, threads_y: int = 32):
        """Compute CUDA grid dimensions for given image size."""
        nb_blocksX = (width // threads_x) + 1
        nb_blocksY = (height // threads_y) + 1
        return nb_blocksX, nb_blocksY

    # ========================================================================
    # Individual Filter Methods - Color
    # ========================================================================

    def apply_rgb_adjustment_color(self, res_b, res_g, res_r,
                                   val_reds: float, val_greens: float,
                                   val_blues: float, params: FilterParams):
        """Apply RGB channel adjustment to color image."""
        if val_reds == 1.0 and val_greens == 1.0 and val_blues == 1.0:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        b_gpu = res_b
        g_gpu = res_g
        r_gpu = res_r

        self.kernels.Set_RGB(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             np.int_(width), np.int_(height),
             np.float32(val_reds), np.float32(val_greens), np.float32(val_blues))
        )

        return b_gpu, g_gpu, r_gpu

    def apply_negative_color(self, res_b, res_g, res_r):
        """Apply image negative to color image."""
        return self.utils['image_negative_colour'](res_b, res_g, res_r)

    def apply_luminance_estimate_color(self, res_b, res_g, res_r,
                                       params: FilterParams):
        """Apply luminance estimation for mono sensor with color filter."""
        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        b_gpu = res_b
        g_gpu = res_g
        r_gpu = res_r

        self.kernels.color_estimate_Mono(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             np.int_(width), np.int_(height))
        )

        return b_gpu, g_gpu, r_gpu

    def apply_frame_stacking_color(self, res_b, res_g, res_r,
                                   params: FilterParams, state: FilterState):
        """Apply 2-5 frame sum or median stacking."""
        if params.val_FS <= 1 or params.flag_image_mode:
            return res_b, res_g, res_r

        state.compteur_FS += 1
        if state.compteur_FS > params.val_FS:
            state.compteur_FS = 1

        # Store current frame in appropriate buffer
        if state.compteur_FS == 1:
            state.b1_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.g1_sm = self.cp.asarray(res_g).astype(self.cp.int16)
            state.r1_sm = self.cp.asarray(res_r).astype(self.cp.int16)
            state.Im1OK = True
        elif state.compteur_FS == 2:
            state.b2_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.g2_sm = self.cp.asarray(res_g).astype(self.cp.int16)
            state.r2_sm = self.cp.asarray(res_r).astype(self.cp.int16)
            state.Im2OK = True
        elif state.compteur_FS == 3:
            state.b3_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.g3_sm = self.cp.asarray(res_g).astype(self.cp.int16)
            state.r3_sm = self.cp.asarray(res_r).astype(self.cp.int16)
            state.Im3OK = True
        elif state.compteur_FS == 4:
            state.b4_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.g4_sm = self.cp.asarray(res_g).astype(self.cp.int16)
            state.r4_sm = self.cp.asarray(res_r).astype(self.cp.int16)
            state.Im4OK = True
        elif state.compteur_FS == 5:
            state.b5_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.g5_sm = self.cp.asarray(res_g).astype(self.cp.int16)
            state.r5_sm = self.cp.asarray(res_r).astype(self.cp.int16)
            state.Im5OK = True

        # Perform stacking if enough frames collected
        def stack_sum(buffers):
            result = buffers[0]
            for buf in buffers[1:]:
                result = result + buf
            result = self.cp.clip(result, 0, 255)
            return self.cp.asarray(result, dtype=self.cp.uint8)

        def stack_median(buffers):
            imgs = self.cp.asarray(buffers)
            return self.cp.asarray(self.cp.median(imgs, axis=0), dtype=self.cp.uint8)

        stack_func = stack_sum if params.stack_div == 1 else stack_median

        if params.val_FS == 2 and state.Im2OK:
            res_b = stack_func([state.b1_sm, state.b2_sm])
            res_g = stack_func([state.g1_sm, state.g2_sm])
            res_r = stack_func([state.r1_sm, state.r2_sm])
        elif params.val_FS == 3 and state.Im3OK:
            res_b = stack_func([state.b1_sm, state.b2_sm, state.b3_sm])
            res_g = stack_func([state.g1_sm, state.g2_sm, state.g3_sm])
            res_r = stack_func([state.r1_sm, state.r2_sm, state.r3_sm])
        elif params.val_FS == 4 and state.Im4OK:
            res_b = stack_func([state.b1_sm, state.b2_sm, state.b3_sm, state.b4_sm])
            res_g = stack_func([state.g1_sm, state.g2_sm, state.g3_sm, state.g4_sm])
            res_r = stack_func([state.r1_sm, state.r2_sm, state.r3_sm, state.r4_sm])
        elif params.val_FS == 5 and state.Im5OK:
            res_b = stack_func([state.b1_sm, state.b2_sm, state.b3_sm, state.b4_sm, state.b5_sm])
            res_g = stack_func([state.g1_sm, state.g2_sm, state.g3_sm, state.g4_sm, state.g5_sm])
            res_r = stack_func([state.r1_sm, state.r2_sm, state.r3_sm, state.r4_sm, state.r5_sm])

        return res_b, res_g, res_r

    def apply_reduce_variation_color(self, res_b, res_g, res_r,
                                     params: FilterParams, state: FilterState):
        """Apply turbulence reduction filter using previous or best frame."""
        if not params.flag_reduce_variation or params.flag_image_mode:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if params.flag_BFReference == "PreviousFrame":
            state.compteur_RV += 1
            if state.compteur_RV < 3:
                if state.compteur_RV == 1:
                    state.res_b2 = res_b.copy()
                    state.res_g2 = res_g.copy()
                    state.res_r2 = res_r.copy()
                    state.Im1rvOK = True
                if state.compteur_RV == 2:
                    state.Im2rvOK = True

            if state.Im2rvOK:
                b_gpu = res_b
                g_gpu = res_g
                r_gpu = res_r
                variation = int(255 / 100 * params.val_reduce_variation)

                self.kernels.reduce_variation_Color(
                    (nb_blocksX, nb_blocksY),
                    (params.nb_ThreadsX, params.nb_ThreadsY),
                    (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
                     state.res_r2, state.res_g2, state.res_b2,
                     np.int_(width), np.int_(height), np.int_(variation))
                )

                state.res_b2 = res_b.copy()
                state.res_g2 = res_g.copy()
                state.res_r2 = res_r.copy()
                return b_gpu, g_gpu, r_gpu

        elif params.flag_BFReference == "BestFrame" and state.flag_BFREF_image:
            res_r2, res_g2, res_b2 = self.utils['numpy_RGBImage_2_cupy_separateRGB'](
                state.BFREF_image)
            variation = int(255 / 100 * params.val_reduce_variation)

            b_gpu = res_b
            g_gpu = res_g
            r_gpu = res_r

            self.kernels.reduce_variation_Color(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
                 res_r2, res_g2, res_b2,
                 np.int_(width), np.int_(height), np.int_(variation))
            )

            return b_gpu, g_gpu, r_gpu

        return res_b, res_g, res_r

    def apply_3fnr_front_color(self, res_b, res_g, res_r,
                               params: FilterParams, state: FilterState):
        """Apply 3-frame noise reduction (front position in pipeline)."""
        if not params.flag_3FNR or params.flag_image_mode:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_3FNR < 4 and state.FNR_First_Start:
            state.compteur_3FNR += 1
            if state.compteur_3FNR == 1:
                state.imgb1 = res_b.copy()
                state.imgg1 = res_g.copy()
                state.imgr1 = res_r.copy()
                state.img1_3FNROK = True
            elif state.compteur_3FNR == 2:
                state.imgb2 = res_b.copy()
                state.imgg2 = res_g.copy()
                state.imgr2 = res_r.copy()
                state.img2_3FNROK = True
            elif state.compteur_3FNR == 3:
                state.imgb3 = res_b.copy()
                state.imgg3 = res_g.copy()
                state.imgr3 = res_r.copy()
                state.img3_3FNROK = True

        if state.img3_3FNROK:
            if not state.FNR_First_Start:
                state.imgb3 = res_b.copy()
                state.imgg3 = res_g.copy()
                state.imgr3 = res_r.copy()

            state.FNR_First_Start = False
            b_gpu = res_b
            g_gpu = res_g
            r_gpu = res_r

            self.kernels.FNR_Color(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu,
                 state.imgr1, state.imgg1, state.imgb1,
                 state.imgr2, state.imgg2, state.imgb2,
                 state.imgr3, state.imgg3, state.imgb3,
                 np.int_(width), np.int_(height), np.float32(params.val_3FNR_Thres))
            )

            # Shift buffers
            state.imgb1 = state.imgb2.copy()
            state.imgg1 = state.imgg2.copy()
            state.imgr1 = state.imgr2.copy()
            state.imgr2 = r_gpu.copy()
            state.imgg2 = g_gpu.copy()
            state.imgb2 = b_gpu.copy()

            return b_gpu, g_gpu, r_gpu

        return res_b, res_g, res_r

    def apply_aanr_front_color(self, res_b, res_g, res_r,
                               params: FilterParams, state: FilterState):
        """Apply Adaptive Absorber Noise Reduction (front position)."""
        if not params.flag_AANR or params.flag_image_mode:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_AANR < 3:
            state.compteur_AANR += 1
            if state.compteur_AANR == 1:
                state.res_b2 = res_b.copy()
                state.res_g2 = res_g.copy()
                state.res_r2 = res_r.copy()
                state.Im1fsdnOK = True
            elif state.compteur_AANR == 2:
                state.Im2fsdnOK = True

        if state.Im2fsdnOK:
            b_gpu = res_b
            g_gpu = res_g
            r_gpu = res_r

            self.kernels.adaptative_absorber_denoise_Color(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
                 state.res_r2, state.res_g2, state.res_b2,
                 np.int_(width), np.int_(height),
                 np.intc(params.flag_dyn_AANR),
                 np.intc(params.flag_ghost_reducer),
                 np.intc(params.val_ghost_reducer))
            )

            state.res_b2 = res_b.copy()
            state.res_g2 = res_g.copy()
            state.res_r2 = res_r.copy()

            # Apply 5% brightness boost
            tmp = self.cp.asarray(r_gpu).astype(self.cp.float64) * 1.05
            tmp = self.cp.clip(tmp, 0, 255)
            res_r = self.cp.asarray(tmp, dtype=self.cp.uint8)

            tmp = self.cp.asarray(g_gpu).astype(self.cp.float64) * 1.05
            tmp = self.cp.clip(tmp, 0, 255)
            res_g = self.cp.asarray(tmp, dtype=self.cp.uint8)

            tmp = self.cp.asarray(b_gpu).astype(self.cp.float64) * 1.05
            tmp = self.cp.clip(tmp, 0, 255)
            res_b = self.cp.asarray(tmp, dtype=self.cp.uint8)

            return res_b, res_g, res_r

        return res_b, res_g, res_r

    def apply_denoise_paillou1_color(self, res_b, res_g, res_r,
                                     params: FilterParams):
        """Apply Paillou denoise filter 1."""
        if params.flag_denoise_Paillou != 1:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        cell_size = 5
        sqr_cell_size = cell_size * cell_size

        b_gpu = res_b
        g_gpu = res_g
        r_gpu = res_r

        self.kernels.Denoise_Paillou_Colour(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             np.intc(width), np.intc(height),
             np.intc(cell_size), np.intc(sqr_cell_size))
        )

        return b_gpu, g_gpu, r_gpu

    def apply_denoise_paillou2_color(self, res_b, res_g, res_r,
                                     params: FilterParams):
        """Apply Paillou denoise filter 2."""
        if params.flag_denoise_Paillou2 != 1:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        b_gpu = res_b
        g_gpu = res_g
        r_gpu = res_r

        self.kernels.reduce_noise_Color(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             np.intc(width), np.intc(height))
        )

        return b_gpu, g_gpu, r_gpu

    def apply_nlm2_color(self, res_b, res_g, res_r, params: FilterParams):
        """Apply Non-Local Means 2 denoising."""
        if params.flag_NLM2 != 1:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_ThreadsXs = 8
        nb_ThreadsYs = 8
        nb_blocksXs = (width // nb_ThreadsXs) + 1
        nb_blocksYs = (height // nb_ThreadsYs) + 1

        param = float(params.val_denoise)
        Noise = 1.0 / (param * param)
        lerpC = 0.4

        b_gpu = res_b
        g_gpu = res_g
        r_gpu = res_r

        self.kernels.NLM2_Colour_GPU(
            (nb_blocksXs, nb_blocksYs),
            (nb_ThreadsXs, nb_ThreadsYs),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             np.intc(width), np.intc(height),
             np.float32(Noise), np.float32(lerpC))
        )

        return b_gpu, g_gpu, r_gpu

    def apply_knn_color(self, res_b, res_g, res_r, params: FilterParams):
        """Apply K-Nearest Neighbors denoising."""
        if params.flag_denoise_KNN != 1:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        param = float(params.val_denoise_KNN)
        Noise = 1.0 / (param * param)
        lerpC = 0.4

        b_gpu = res_b
        g_gpu = res_g
        r_gpu = res_r

        self.kernels.KNN_Colour_GPU(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             np.intc(width), np.intc(height),
             np.float32(Noise), np.float32(lerpC))
        )

        return b_gpu, g_gpu, r_gpu

    def apply_histogram_color(self, res_b, res_g, res_r, params: FilterParams):
        """Apply histogram operations (stretch, equalize, phi-theta)."""
        if not (params.flag_histogram_stretch == 1 or
                params.flag_histogram_equalize2 == 1 or
                params.flag_histogram_phitheta == 1):
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        b_gpu = res_b
        g_gpu = res_g
        r_gpu = res_r

        self.kernels.Histo_Color(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             np.int_(width), np.int_(height),
             np.intc(params.flag_histogram_stretch),
             np.float32(params.val_histo_min), np.float32(params.val_histo_max),
             np.intc(params.flag_histogram_equalize2), np.float32(params.val_heq2),
             np.intc(params.flag_histogram_phitheta),
             np.float32(params.val_phi), np.float32(params.val_theta))
        )

        return b_gpu, g_gpu, r_gpu

    def apply_amplification_color(self, res_b, res_g, res_r,
                                  params: FilterParams):
        """Apply image amplification (linear or gaussian)."""
        if params.flag_AmpSoft != 1:
            return res_b, res_g, res_r

        if params.flag_lin_gauss not in (1, 2):
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        correction = self.cp.asarray(params.Corr_GS)

        b_gpu = res_b
        g_gpu = res_g
        r_gpu = res_r

        self.kernels.Colour_ampsoft_GPU(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             np.int_(width), np.int_(height),
             np.float32(params.val_ampl), correction)
        )

        return b_gpu, g_gpu, r_gpu

    def apply_star_amplification_color(self, res_b, res_g, res_r,
                                       params: FilterParams):
        """Apply star amplification."""
        if params.flag_AmpSoft != 1 or params.flag_lin_gauss != 3:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        # Convert to grayscale for star detection
        image_brute_grey = self.cv2.cvtColor(
            self.utils['cupy_separateRGB_2_numpy_RGBimage'](res_b, res_g, res_r),
            self.cv2.COLOR_RGB2GRAY
        )
        imagegrey = self.cp.asarray(image_brute_grey)

        niveau_blur = 7
        imagegreyblur = self.utils['gaussianblur_mono'](imagegrey, niveau_blur)
        correction = self.cp.asarray(params.Corr_GS)

        r_gpu = res_r
        g_gpu = res_g
        b_gpu = res_b

        self.kernels.Colour_staramp_GPU(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             imagegrey, imagegreyblur,
             np.int_(width), np.int_(height),
             np.float32(params.val_Mu), np.float32(params.val_Ro),
             np.float32(params.val_ampl), correction)
        )

        return b_gpu, g_gpu, r_gpu

    def apply_gradient_removal_color(self, res_b, res_g, res_r,
                                     params: FilterParams):
        """Apply gradient removal or vignetting correction."""
        if not params.flag_GR:
            return res_b, res_g, res_r

        if params.grad_vignet == 1:
            # Vignetting correction
            seuilb = int(self.cp.percentile(res_b, params.val_SGR))
            seuilg = int(self.cp.percentile(res_g, params.val_SGR))
            seuilr = int(self.cp.percentile(res_r, params.val_SGR))

            img_b = res_b.copy()
            img_g = res_g.copy()
            img_r = res_r.copy()

            img_b[img_b > seuilb] = seuilb
            img_g[img_g > seuilg] = seuilg
            img_r[img_r > seuilr] = seuilr

            niveau_blur = params.val_NGB * 2 + 3
            img_b, img_g, img_r = self.utils['gaussianblur_colour'](
                img_b, img_g, img_r, niveau_blur)

            att_b = self.cp.asarray(img_b) * ((100.0 - params.val_AGR) / 100.0)
            att_g = self.cp.asarray(img_g) * ((100.0 - params.val_AGR) / 100.0)
            att_r = self.cp.asarray(img_r) * ((100.0 - params.val_AGR) / 100.0)

            resb = self.cp.subtract(self.cp.asarray(res_b), att_b)
            resg = self.cp.subtract(self.cp.asarray(res_g), att_g)
            resr = self.cp.subtract(self.cp.asarray(res_r), att_r)

            resb = self.cp.clip(resb, 0, 255)
            resg = self.cp.clip(resg, 0, 255)
            resr = self.cp.clip(resr, 0, 255)

            return (self.cp.asarray(resb, dtype=self.cp.uint8),
                    self.cp.asarray(resg, dtype=self.cp.uint8),
                    self.cp.asarray(resr, dtype=self.cp.uint8))
        else:
            # Gradient removal
            seuilb = int(self.cp.percentile(res_b, params.val_SGR))
            seuilg = int(self.cp.percentile(res_g, params.val_SGR))
            seuilr = int(self.cp.percentile(res_r, params.val_SGR))

            fd_b = res_b.copy()
            fd_g = res_g.copy()
            fd_r = res_r.copy()

            fd_b[fd_b > seuilb] = seuilb
            fd_g[fd_g > seuilg] = seuilg
            fd_r[fd_r > seuilr] = seuilr

            niveau_blur = params.val_NGB * 2 + 3
            fd_b, fd_g, fd_r = self.utils['gaussianblur_colour'](
                fd_b, fd_g, fd_r, niveau_blur)

            pivot_b = int(self.cp.percentile(self.cp.asarray(res_b), params.val_AGR))
            pivot_g = int(self.cp.percentile(self.cp.asarray(res_g), params.val_AGR))
            pivot_r = int(self.cp.percentile(self.cp.asarray(res_r), params.val_AGR))

            corr_b = (self.cp.asarray(res_b).astype(self.cp.int16) -
                      self.cp.asarray(fd_b).astype(self.cp.int16) + pivot_b)
            corr_g = (self.cp.asarray(res_g).astype(self.cp.int16) -
                      self.cp.asarray(fd_g).astype(self.cp.int16) + pivot_g)
            corr_r = (self.cp.asarray(res_r).astype(self.cp.int16) -
                      self.cp.asarray(fd_r).astype(self.cp.int16) + pivot_r)

            corr_b = self.cp.clip(corr_b, 0, 255)
            corr_g = self.cp.clip(corr_g, 0, 255)
            corr_r = self.cp.clip(corr_r, 0, 255)

            return (self.cp.asarray(corr_b, dtype=self.cp.uint8),
                    self.cp.asarray(corr_g, dtype=self.cp.uint8),
                    self.cp.asarray(corr_r, dtype=self.cp.uint8))

    def apply_cll_color(self, res_b, res_g, res_r, params: FilterParams):
        """Apply Contrast Low Light enhancement."""
        if params.flag_CLL != 1:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        correction_CLL = self.cp.asarray(params.Corr_CLL, dtype=self.cp.uint8)

        r_gpu = res_r
        g_gpu = res_g
        b_gpu = res_b

        self.kernels.Contrast_Low_Light_Colour_GPU(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
             np.int_(width), np.int_(height), correction_CLL)
        )

        return b_gpu, g_gpu, r_gpu

    def apply_clahe_color(self, res_b, res_g, res_r, params: FilterParams):
        """Apply CLAHE contrast enhancement."""
        if params.flag_contrast_CLAHE != 1:
            return res_b, res_g, res_r

        if params.flag_OpenCvCuda:
            clahe = self.cv2.cuda.createCLAHE(
                clipLimit=params.val_contrast_CLAHE,
                tileGridSize=(params.val_grid_CLAHE, params.val_grid_CLAHE)
            )

            srcb = self.cv2.cuda_GpuMat()
            srcb.upload(res_b.get())
            resb = clahe.apply(srcb, self.cv2.cuda_Stream.Null())
            resbb = resb.download()

            srcg = self.cv2.cuda_GpuMat()
            srcg.upload(res_g.get())
            resg = clahe.apply(srcg, self.cv2.cuda_Stream.Null())
            resgg = resg.download()

            srcr = self.cv2.cuda_GpuMat()
            srcr.upload(res_r.get())
            resr = clahe.apply(srcr, self.cv2.cuda_Stream.Null())
            resrr = resr.download()

            return (self.cp.asarray(resbb),
                    self.cp.asarray(resgg),
                    self.cp.asarray(resrr))
        else:
            clahe = self.cv2.createCLAHE(
                clipLimit=params.val_contrast_CLAHE,
                tileGridSize=(params.val_grid_CLAHE, params.val_grid_CLAHE)
            )
            b = clahe.apply(res_b.get())
            g = clahe.apply(res_g.get())
            r = clahe.apply(res_r.get())

            return (self.cp.asarray(b),
                    self.cp.asarray(g),
                    self.cp.asarray(r))

    def apply_saturation_color(self, res_b, res_g, res_r, params: FilterParams):
        """Apply color saturation enhancement."""
        if not params.flag_SAT:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        flag_neg_sat = 1 if params.ImageNeg == 1 else 0

        def apply_saturation_pass(r_in, g_in, b_in, blur_size):
            r_gpu = r_in.copy()
            g_gpu = g_in.copy()
            b_gpu = b_in.copy()
            init_r = r_in.copy()
            init_g = g_in.copy()
            init_b = b_in.copy()

            coul_r, coul_g, coul_b = self.utils['gaussianblur_colour'](
                r_gpu, g_gpu, b_gpu, blur_size)

            self.kernels.Saturation_Colour(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu, coul_r, coul_g, coul_b,
                 np.int_(width), np.int_(height),
                 np.float32(params.val_SAT), np.int_(flag_neg_sat))
            )

            coul_gauss2_r = r_gpu.copy()
            coul_gauss2_g = g_gpu.copy()
            coul_gauss2_b = b_gpu.copy()
            coul_gauss2_r, coul_gauss2_g, coul_gauss2_b = self.utils['gaussianblur_colour'](
                coul_gauss2_r, coul_gauss2_g, coul_gauss2_b, 7)

            self.kernels.Saturation_Combine_Colour(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu, init_r, init_g, init_b,
                 coul_gauss2_r, coul_gauss2_g, coul_gauss2_b,
                 np.int_(width), np.int_(height))
            )

            return r_gpu.copy(), g_gpu.copy(), b_gpu.copy()

        if params.flag_SAT_Image:
            res_r, res_g, res_b = apply_saturation_pass(res_r, res_g, res_b, 5)
            if params.flag_SAT2PASS:
                res_r, res_g, res_b = apply_saturation_pass(res_r, res_g, res_b, 11)
        else:
            r_gpu = res_r.copy()
            g_gpu = res_g.copy()
            b_gpu = res_b.copy()

            coul_gauss_r, coul_gauss_g, coul_gauss_b = self.utils['gaussianblur_colour'](
                res_r.copy(), res_g.copy(), res_b.copy(), 5)

            self.kernels.Saturation_Colour(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu, coul_gauss_r, coul_gauss_g, coul_gauss_b,
                 np.int_(width), np.int_(height),
                 np.float32(params.val_SAT), np.int_(flag_neg_sat))
            )

            coul_gauss_r = r_gpu.copy()
            coul_gauss_g = g_gpu.copy()
            coul_gauss_b = b_gpu.copy()
            coul_gauss_r, coul_gauss_g, coul_gauss_b = self.utils['gaussianblur_colour'](
                coul_gauss_r, coul_gauss_g, coul_gauss_b, 7)

            self.kernels.Saturation_Combine_Colour(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
                 coul_gauss_r, coul_gauss_g, coul_gauss_b,
                 np.int_(width), np.int_(height))
            )

            res_r = r_gpu.copy()
            res_g = g_gpu.copy()
            res_b = b_gpu.copy()

            if params.flag_SAT2PASS:
                r_gpu = res_r.copy()
                g_gpu = res_g.copy()
                b_gpu = res_b.copy()

                coul_gauss_r, coul_gauss_g, coul_gauss_b = self.utils['gaussianblur_colour'](
                    res_r.copy(), res_g.copy(), res_b.copy(), 11)

                self.kernels.Saturation_Colour(
                    (nb_blocksX, nb_blocksY),
                    (params.nb_ThreadsX, params.nb_ThreadsY),
                    (r_gpu, g_gpu, b_gpu, coul_gauss_r, coul_gauss_g, coul_gauss_b,
                     np.int_(width), np.int_(height),
                     np.float32(params.val_SAT), np.int_(flag_neg_sat))
                )

                coul_gauss_r = r_gpu.copy()
                coul_gauss_g = g_gpu.copy()
                coul_gauss_b = b_gpu.copy()
                coul_gauss_r, coul_gauss_g, coul_gauss_b = self.utils['gaussianblur_colour'](
                    coul_gauss_r, coul_gauss_g, coul_gauss_b, 7)

                self.kernels.Saturation_Combine_Colour(
                    (nb_blocksX, nb_blocksY),
                    (params.nb_ThreadsX, params.nb_ThreadsY),
                    (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
                     coul_gauss_r, coul_gauss_g, coul_gauss_b,
                     np.int_(width), np.int_(height))
                )

                res_r = r_gpu.copy()
                res_g = g_gpu.copy()
                res_b = b_gpu.copy()

        return res_b, res_g, res_r

    def apply_3fnr2_front_color(self, res_b, res_g, res_r,
                                params: FilterParams, state: FilterState):
        """Apply 3-frame noise reduction 2 (front position in pipeline)."""
        if not params.flag_3FNR2 or params.flag_image_mode:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_3FNR2 < 4 and state.FNR2_First_Start:
            state.compteur_3FNR2 += 1
            if state.compteur_3FNR2 == 1:
                state.imgb21 = res_b.copy()
                state.imgg21 = res_g.copy()
                state.imgr21 = res_r.copy()
                state.img1_3FNR2OK = True
            elif state.compteur_3FNR2 == 2:
                state.imgb22 = res_b.copy()
                state.imgg22 = res_g.copy()
                state.imgr22 = res_r.copy()
                state.img2_3FNR2OK = True
            elif state.compteur_3FNR2 == 3:
                state.imgb23 = res_b.copy()
                state.imgg23 = res_g.copy()
                state.imgr23 = res_r.copy()
                state.img3_3FNR2OK = True

        if state.img3_3FNR2OK:
            if not state.FNR2_First_Start:
                state.imgb23 = res_b.copy()
                state.imgg23 = res_g.copy()
                state.imgr23 = res_r.copy()

            state.FNR2_First_Start = False
            b_gpu = res_b
            g_gpu = res_g
            r_gpu = res_r

            self.kernels.FNR2_Color(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu,
                 state.imgr21, state.imgg21, state.imgb21,
                 state.imgr22, state.imgg22, state.imgb22,
                 state.imgr23, state.imgg23, state.imgb23,
                 np.int_(width), np.int_(height))
            )

            # Shift buffers
            state.imgb21 = state.imgb22.copy()
            state.imgg21 = state.imgg22.copy()
            state.imgr21 = state.imgr22.copy()
            state.imgr22 = r_gpu.copy()
            state.imgg22 = g_gpu.copy()
            state.imgb22 = b_gpu.copy()

            return b_gpu, g_gpu, r_gpu

        return res_b, res_g, res_r

    def apply_3fnr_back_color(self, res_b, res_g, res_r,
                              params: FilterParams, state: FilterState):
        """Apply 3-frame noise reduction (back position in pipeline)."""
        if not params.flag_3FNRB or params.flag_image_mode:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_3FNRB < 4 and state.FNRB_First_Start:
            state.compteur_3FNRB += 1
            if state.compteur_3FNRB == 1:
                state.imgb1B = res_b.copy()
                state.imgg1B = res_g.copy()
                state.imgr1B = res_r.copy()
                state.img1_3FNROKB = True
            elif state.compteur_3FNRB == 2:
                state.imgb2B = res_b.copy()
                state.imgg2B = res_g.copy()
                state.imgr2B = res_r.copy()
                state.img2_3FNROKB = True
            elif state.compteur_3FNRB == 3:
                state.imgb3B = res_b.copy()
                state.imgg3B = res_g.copy()
                state.imgr3B = res_r.copy()
                state.img3_3FNROKB = True

        if state.img3_3FNROKB:
            if not state.FNRB_First_Start:
                state.imgb3B = res_b.copy()
                state.imgg3B = res_g.copy()
                state.imgr3B = res_r.copy()

            state.FNRB_First_Start = False
            b_gpu = res_b
            g_gpu = res_g
            r_gpu = res_r

            self.kernels.FNR_Color(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu,
                 state.imgr1B, state.imgg1B, state.imgb1B,
                 state.imgr2B, state.imgg2B, state.imgb2B,
                 state.imgr3B, state.imgg3B, state.imgb3B,
                 np.int_(width), np.int_(height), np.float32(params.val_3FNR_Thres))
            )

            # Shift buffers
            state.imgb1B = state.imgb2B.copy()
            state.imgg1B = state.imgg2B.copy()
            state.imgr1B = state.imgr2B.copy()
            state.imgr2B = r_gpu.copy()
            state.imgg2B = g_gpu.copy()
            state.imgb2B = b_gpu.copy()

            return b_gpu, g_gpu, r_gpu

        return res_b, res_g, res_r

    def apply_3fnr2_back_color(self, res_b, res_g, res_r,
                               params: FilterParams, state: FilterState):
        """Apply 3-frame noise reduction 2 (back position in pipeline)."""
        if not params.flag_3FNR2B or params.flag_image_mode:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_3FNR2B < 4 and state.FNR2B_First_Start:
            state.compteur_3FNR2B += 1
            if state.compteur_3FNR2B == 1:
                state.imgb21B = res_b.copy()
                state.imgg21B = res_g.copy()
                state.imgr21B = res_r.copy()
                state.img1_3FNR2OKB = True
            elif state.compteur_3FNR2B == 2:
                state.imgb22B = res_b.copy()
                state.imgg22B = res_g.copy()
                state.imgr22B = res_r.copy()
                state.img2_3FNR2OKB = True
            elif state.compteur_3FNR2B == 3:
                state.imgb23B = res_b.copy()
                state.imgg23B = res_g.copy()
                state.imgr23B = res_r.copy()
                state.img3_3FNR2OKB = True

        if state.img3_3FNR2OKB:
            if not state.FNR2B_First_Start:
                state.imgb23B = res_b.copy()
                state.imgg23B = res_g.copy()
                state.imgr23B = res_r.copy()

            state.FNR2B_First_Start = False
            b_gpu = res_b
            g_gpu = res_g
            r_gpu = res_r

            self.kernels.FNR2_Color(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu,
                 state.imgr21B, state.imgg21B, state.imgb21B,
                 state.imgr22B, state.imgg22B, state.imgb22B,
                 state.imgr23B, state.imgg23B, state.imgb23B,
                 np.int_(width), np.int_(height))
            )

            # Shift buffers
            state.imgb21B = state.imgb22B.copy()
            state.imgg21B = state.imgg22B.copy()
            state.imgr21B = state.imgr22B.copy()
            state.imgr22B = r_gpu.copy()
            state.imgg22B = g_gpu.copy()
            state.imgb22B = b_gpu.copy()

            return b_gpu, g_gpu, r_gpu

        return res_b, res_g, res_r

    def apply_aanr_back_color(self, res_b, res_g, res_r,
                              params: FilterParams, state: FilterState):
        """Apply Adaptive Absorber Noise Reduction (back position)."""
        if not params.flag_AANRB or params.flag_image_mode:
            return res_b, res_g, res_r

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_AANRB < 3:
            state.compteur_AANRB += 1
            if state.compteur_AANRB == 1:
                state.res_b2B = res_b.copy()
                state.res_g2B = res_g.copy()
                state.res_r2B = res_r.copy()
                state.Im1fsdnOKB = True
            elif state.compteur_AANRB == 2:
                state.Im2fsdnOKB = True

        if state.Im2fsdnOKB:
            b_gpu = res_b
            g_gpu = res_g
            r_gpu = res_r

            # Back position uses fixed settings: high dynamic, no ghost reduction
            local_dyn = 1
            local_GR = 0
            local_VGR = 0

            self.kernels.adaptative_absorber_denoise_Color(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, g_gpu, b_gpu, res_r, res_g, res_b,
                 state.res_r2B, state.res_g2B, state.res_b2B,
                 np.int_(width), np.int_(height),
                 np.intc(local_dyn), np.intc(local_GR), np.intc(local_VGR))
            )

            state.res_b2B = res_b.copy()
            state.res_g2B = res_g.copy()
            state.res_r2B = res_r.copy()

            # Apply 5% brightness boost
            tmp = self.cp.asarray(r_gpu).astype(self.cp.float64) * 1.05
            tmp = self.cp.clip(tmp, 0, 255)
            res_r = self.cp.asarray(tmp, dtype=self.cp.uint8)

            tmp = self.cp.asarray(g_gpu).astype(self.cp.float64) * 1.05
            tmp = self.cp.clip(tmp, 0, 255)
            res_g = self.cp.asarray(tmp, dtype=self.cp.uint8)

            tmp = self.cp.asarray(b_gpu).astype(self.cp.float64) * 1.05
            tmp = self.cp.clip(tmp, 0, 255)
            res_b = self.cp.asarray(tmp, dtype=self.cp.uint8)

            return res_b, res_g, res_r

        return res_b, res_g, res_r

    def apply_sharpen_color(self, res_b, res_g, res_r, params: FilterParams,
                            cupy_context):
        """Apply sharpening filters 1 and/or 2."""
        res_s1_b1 = res_s1_g1 = res_s1_r1 = None
        res_s2_b1 = res_s2_g1 = res_s2_r1 = None

        # Sharpen 1
        if params.flag_sharpen_soft1 == 1:
            cupy_context.use()
            res_b1_blur, res_g1_blur, res_r1_blur = self.utils['gaussianblur_colour'](
                res_b, res_g, res_r, params.val_sigma_sharpen)

            tmp_b1 = self.cp.asarray(res_b).astype(self.cp.int16)
            tmp_g1 = self.cp.asarray(res_g).astype(self.cp.int16)
            tmp_r1 = self.cp.asarray(res_r).astype(self.cp.int16)

            tmp_b1 = tmp_b1 + params.val_sharpen * (tmp_b1 - res_b1_blur)
            tmp_g1 = tmp_g1 + params.val_sharpen * (tmp_g1 - res_g1_blur)
            tmp_r1 = tmp_r1 + params.val_sharpen * (tmp_r1 - res_r1_blur)

            tmp_b1 = self.cp.clip(tmp_b1, 0, 255)
            tmp_g1 = self.cp.clip(tmp_g1, 0, 255)
            tmp_r1 = self.cp.clip(tmp_r1, 0, 255)

            if params.flag_sharpen_soft2 == 1:
                res_s1_b1 = self.cp.asarray(tmp_b1, dtype=self.cp.uint8)
                res_s1_g1 = self.cp.asarray(tmp_g1, dtype=self.cp.uint8)
                res_s1_r1 = self.cp.asarray(tmp_r1, dtype=self.cp.uint8)
            else:
                res_b = self.cp.asarray(tmp_b1, dtype=self.cp.uint8)
                res_g = self.cp.asarray(tmp_g1, dtype=self.cp.uint8)
                res_r = self.cp.asarray(tmp_r1, dtype=self.cp.uint8)

        # Sharpen 2
        if params.flag_sharpen_soft2 == 1:
            cupy_context.use()
            res_b1_blur, res_g1_blur, res_r1_blur = self.utils['gaussianblur_colour'](
                res_b, res_g, res_r, params.val_sigma_sharpen2)

            tmp_b1 = self.cp.asarray(res_b).astype(self.cp.int16)
            tmp_g1 = self.cp.asarray(res_g).astype(self.cp.int16)
            tmp_r1 = self.cp.asarray(res_r).astype(self.cp.int16)

            tmp_b1 = tmp_b1 + params.val_sharpen2 * (tmp_b1 - res_b1_blur)
            tmp_g1 = tmp_g1 + params.val_sharpen2 * (tmp_g1 - res_g1_blur)
            tmp_r1 = tmp_r1 + params.val_sharpen2 * (tmp_r1 - res_r1_blur)

            tmp_b1 = self.cp.clip(tmp_b1, 0, 255)
            tmp_g1 = self.cp.clip(tmp_g1, 0, 255)
            tmp_r1 = self.cp.clip(tmp_r1, 0, 255)

            if params.flag_sharpen_soft1 == 1:
                res_s2_b1 = self.cp.asarray(tmp_b1, dtype=self.cp.uint8)
                res_s2_g1 = self.cp.asarray(tmp_g1, dtype=self.cp.uint8)
                res_s2_r1 = self.cp.asarray(tmp_r1, dtype=self.cp.uint8)
            else:
                res_b = self.cp.asarray(tmp_b1, dtype=self.cp.uint8)
                res_g = self.cp.asarray(tmp_g1, dtype=self.cp.uint8)
                res_r = self.cp.asarray(tmp_r1, dtype=self.cp.uint8)

        # Combine both sharpens with minimum
        if params.flag_sharpen_soft1 == 1 and params.flag_sharpen_soft2 == 1:
            res_b = self.cp.minimum(res_s1_b1, res_s2_b1)
            res_g = self.cp.minimum(res_s1_g1, res_s2_g1)
            res_r = self.cp.minimum(res_s1_r1, res_s2_r1)

        return res_b, res_g, res_r

    # ========================================================================
    # Main Processing Methods
    # ========================================================================

    def process_color(self, res_b, res_g, res_r,
                      params: FilterParams, state: FilterState,
                      cupy_context) -> np.ndarray:
        """
        Process a color frame through the complete filter pipeline.

        Args:
            res_b: Blue channel (CuPy array)
            res_g: Green channel (CuPy array)
            res_r: Red channel (CuPy array)
            params: Filter parameters
            state: Filter state for temporal operations
            cupy_context: CuPy CUDA stream context

        Returns:
            Processed RGB image as numpy array
        """
        start_time = self.cv2.getTickCount()

        with cupy_context:
            if not params.flag_filtrage_ON:
                state.image_traitee = self.utils['cupy_separateRGB_2_numpy_RGBimage'](
                    res_b, res_g, res_r)
                return state.image_traitee

            if not params.flag_IsColor:
                # Should use process_mono instead
                return self.process_mono(res_b, params, state, cupy_context)

            height, width = res_b.shape

            # Store original for demo mode
            image_base = None
            if params.flag_DEMO == 1:
                image_base = self.utils['cupy_separateRGB_2_numpy_RGBimage'](
                    res_b, res_g, res_r)

            # Gaussian blur (pre-filter)
            if params.flag_GaussBlur:
                res_b, res_g, res_r = self.utils['gaussianblur_colour'](
                    res_b, res_g, res_r, 3)

            # 1. RGB adjustment
            res_b, res_g, res_r = self.apply_rgb_adjustment_color(
                res_b, res_g, res_r,
                params.val_reds, params.val_greens, params.val_blues, params)

            # 2. Image negative
            if params.ImageNeg == 1:
                res_b, res_g, res_r = self.apply_negative_color(res_b, res_g, res_r)

            # 3. Luminance estimation
            if params.flag_NB_estime == 1:
                res_b, res_g, res_r = self.apply_luminance_estimate_color(
                    res_b, res_g, res_r, params)

            # 4. Frame stacking
            res_b, res_g, res_r = self.apply_frame_stacking_color(
                res_b, res_g, res_r, params, state)

            # 5. Reduce variation
            res_b, res_g, res_r = self.apply_reduce_variation_color(
                res_b, res_g, res_r, params, state)

            # 6. 3FNR front
            res_b, res_g, res_r = self.apply_3fnr_front_color(
                res_b, res_g, res_r, params, state)

            # 6b. 3FNR2 front
            res_b, res_g, res_r = self.apply_3fnr2_front_color(
                res_b, res_g, res_r, params, state)

            # 7. AANR front
            res_b, res_g, res_r = self.apply_aanr_front_color(
                res_b, res_g, res_r, params, state)

            # 8. Denoise Paillou 1
            res_b, res_g, res_r = self.apply_denoise_paillou1_color(
                res_b, res_g, res_r, params)

            # 9. Denoise Paillou 2
            res_b, res_g, res_r = self.apply_denoise_paillou2_color(
                res_b, res_g, res_r, params)

            # 10. NLM2
            res_b, res_g, res_r = self.apply_nlm2_color(res_b, res_g, res_r, params)

            # 11. KNN
            res_b, res_g, res_r = self.apply_knn_color(res_b, res_g, res_r, params)

            # 12-13. Histogram and amplification
            res_b, res_g, res_r = self.apply_histogram_color(
                res_b, res_g, res_r, params)
            res_b, res_g, res_r = self.apply_amplification_color(
                res_b, res_g, res_r, params)

            # 14. Star amplification
            res_b, res_g, res_r = self.apply_star_amplification_color(
                res_b, res_g, res_r, params)

            # 15. Gradient/vignetting
            res_b, res_g, res_r = self.apply_gradient_removal_color(
                res_b, res_g, res_r, params)

            # 16. CLL
            res_b, res_g, res_r = self.apply_cll_color(res_b, res_g, res_r, params)

            # 17. CLAHE
            res_b, res_g, res_r = self.apply_clahe_color(res_b, res_g, res_r, params)

            # 18. Saturation
            res_b, res_g, res_r = self.apply_saturation_color(
                res_b, res_g, res_r, params)

            # 19. 3FNR back
            res_b, res_g, res_r = self.apply_3fnr_back_color(
                res_b, res_g, res_r, params, state)

            # 19b. 3FNR2 back
            res_b, res_g, res_r = self.apply_3fnr2_back_color(
                res_b, res_g, res_r, params, state)

            # 20. AANR back
            res_b, res_g, res_r = self.apply_aanr_back_color(
                res_b, res_g, res_r, params, state)

            # 21-22. Sharpen
            res_b, res_g, res_r = self.apply_sharpen_color(
                res_b, res_g, res_r, params, cupy_context)

            # Convert to output format
            if params.flag_reverse_RB == 0:
                state.image_traitee = self.utils['cupy_separateRGB_2_numpy_RGBimage'](
                    res_b, res_g, res_r)
            else:
                state.image_traitee = self.utils['cupy_separateRGB_2_numpy_RGBimage'](
                    res_r, res_g, res_b)

            # Demo mode overlay
            if params.flag_DEMO == 1 and image_base is not None:
                if params.flag_demo_side == "Left":
                    state.image_traitee[0:height, 0:width//2] = image_base[0:height, 0:width//2]
                elif params.flag_demo_side == "Right":
                    state.image_traitee[0:height, width//2:width] = image_base[0:height, width//2:width]

        # Update timing
        stop_time = self.cv2.getTickCount()
        state.time_exec_test = int((stop_time - start_time) / self.cv2.getTickFrequency() * 1000)

        state.TTQueue.append(state.time_exec_test)
        if len(state.TTQueue) > 10:
            state.TTQueue.pop(0)
        state.curTT = sum(state.TTQueue) / len(state.TTQueue)

        return state.image_traitee

    # ========================================================================
    # Individual Filter Methods - Mono
    # ========================================================================

    def apply_negative_mono(self, res_b):
        """Apply image negative to mono image."""
        return self.cp.invert(res_b, dtype=self.cp.uint8)

    def apply_luminance_estimate_mono(self, res_b, params: FilterParams):
        """Apply luminance estimation for mono sensor."""
        if params.flag_NB_estime != 1:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        r_gpu = res_b.copy()

        self.kernels.grey_estimate_Mono(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, res_b, np.int_(width), np.int_(height))
        )

        return r_gpu.copy()

    def apply_frame_stacking_mono(self, res_b, params: FilterParams,
                                  state: FilterState):
        """Apply 2-5 frame sum or median stacking for mono."""
        if params.val_FS <= 1 or params.flag_image_mode:
            return res_b

        state.compteur_FS += 1
        if state.compteur_FS > params.val_FS:
            state.compteur_FS = 1

        # Store current frame in appropriate buffer
        if state.compteur_FS == 1:
            state.b1_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.Im1OK = True
        elif state.compteur_FS == 2:
            state.b2_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.Im2OK = True
        elif state.compteur_FS == 3:
            state.b3_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.Im3OK = True
        elif state.compteur_FS == 4:
            state.b4_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.Im4OK = True
        elif state.compteur_FS == 5:
            state.b5_sm = self.cp.asarray(res_b).astype(self.cp.int16)
            state.Im5OK = True

        def stack_sum(buffers):
            result = buffers[0]
            for buf in buffers[1:]:
                result = result + buf
            result = self.cp.clip(result, 0, 255)
            return self.cp.asarray(result, dtype=self.cp.uint8)

        def stack_median(buffers):
            imgs = self.cp.asarray(buffers)
            return self.cp.asarray(self.cp.median(imgs, axis=0), dtype=self.cp.uint8)

        stack_func = stack_sum if params.stack_div == 1 else stack_median

        if params.val_FS == 2 and state.Im2OK:
            res_b = stack_func([state.b1_sm, state.b2_sm])
        elif params.val_FS == 3 and state.Im3OK:
            res_b = stack_func([state.b1_sm, state.b2_sm, state.b3_sm])
        elif params.val_FS == 4 and state.Im4OK:
            res_b = stack_func([state.b1_sm, state.b2_sm, state.b3_sm, state.b4_sm])
        elif params.val_FS == 5 and state.Im5OK:
            res_b = stack_func([state.b1_sm, state.b2_sm, state.b3_sm,
                               state.b4_sm, state.b5_sm])

        return res_b

    def apply_reduce_variation_mono(self, res_b, params: FilterParams,
                                    state: FilterState):
        """Apply turbulence reduction filter for mono."""
        if not params.flag_reduce_variation or params.flag_image_mode:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if params.flag_BFReference == "PreviousFrame":
            state.compteur_RV += 1
            if state.compteur_RV < 3:
                if state.compteur_RV == 1:
                    state.res_b2 = res_b.copy()
                    state.Im1rvOK = True
                if state.compteur_RV == 2:
                    state.Im2rvOK = True

            if state.Im2rvOK:
                b_gpu = res_b
                variation = int(255 / 100 * params.val_reduce_variation)

                self.kernels.reduce_variation_Mono(
                    (nb_blocksX, nb_blocksY),
                    (params.nb_ThreadsX, params.nb_ThreadsY),
                    (b_gpu, res_b, state.res_b2,
                     np.int_(width), np.int_(height), np.int_(variation))
                )

                state.res_b2 = res_b.copy()
                return b_gpu

        elif params.flag_BFReference == "BestFrame" and state.flag_BFREF_image:
            res_b2 = self.cp.asarray(state.BFREF_image, dtype=self.cp.uint8)
            variation = int(255 / 100 * params.val_reduce_variation)
            b_gpu = res_b

            self.kernels.reduce_variation_Mono(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (b_gpu, res_b, res_b2,
                 np.int_(width), np.int_(height), np.int_(variation))
            )

            return b_gpu

        return res_b

    def apply_3fnr_front_mono(self, res_b, params: FilterParams,
                              state: FilterState):
        """Apply 3-frame noise reduction (front position) for mono."""
        if not params.flag_3FNR or params.flag_image_mode:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_3FNR < 4 and state.FNR_First_Start:
            state.compteur_3FNR += 1
            if state.compteur_3FNR == 1:
                state.imgb1 = res_b.copy()
                state.img1_3FNROK = True
            elif state.compteur_3FNR == 2:
                state.imgb2 = res_b.copy()
                state.img2_3FNROK = True
            elif state.compteur_3FNR == 3:
                state.imgb3 = res_b.copy()
                state.img3_3FNROK = True

        if state.img3_3FNROK:
            if not state.FNR_First_Start:
                state.imgb3 = res_b.copy()

            state.FNR_First_Start = False
            b_gpu = res_b

            self.kernels.FNR_Mono(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (b_gpu, state.imgb1, state.imgb2, state.imgb3,
                 np.int_(width), np.int_(height), np.float32(params.val_3FNR_Thres))
            )

            state.imgb1 = state.imgb2.copy()
            state.imgb2 = b_gpu.copy()

            return b_gpu

        return res_b

    def apply_3fnr2_front_mono(self, res_b, params: FilterParams,
                               state: FilterState):
        """Apply 3-frame noise reduction 2 (front position) for mono."""
        if not params.flag_3FNR2 or params.flag_image_mode:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_3FNR2 < 4 and state.FNR2_First_Start:
            state.compteur_3FNR2 += 1
            if state.compteur_3FNR2 == 1:
                state.imgb21 = res_b.copy()
                state.img1_3FNR2OK = True
            elif state.compteur_3FNR2 == 2:
                state.imgb22 = res_b.copy()
                state.img2_3FNR2OK = True
            elif state.compteur_3FNR2 == 3:
                state.imgb23 = res_b.copy()
                state.img3_3FNR2OK = True

        if state.img3_3FNR2OK:
            if not state.FNR2_First_Start:
                state.imgb23 = res_b.copy()

            state.FNR2_First_Start = False
            b_gpu = res_b

            self.kernels.FNR2_Mono(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (b_gpu, state.imgb21, state.imgb22, state.imgb23,
                 np.int_(width), np.int_(height))
            )

            state.imgb21 = state.imgb22.copy()
            state.imgb22 = b_gpu.copy()

            return b_gpu

        return res_b

    def apply_aanr_front_mono(self, res_b, params: FilterParams,
                              state: FilterState):
        """Apply Adaptive Absorber Noise Reduction (front) for mono."""
        if not params.flag_AANR or params.flag_image_mode:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_AANR < 3:
            state.compteur_AANR += 1
            if state.compteur_AANR == 1:
                state.res_b2 = res_b.copy()
                state.Im1fsdnOK = True
            elif state.compteur_AANR == 2:
                state.Im2fsdnOK = True

        r_gpu = res_b

        if state.Im2fsdnOK:
            self.kernels.adaptative_absorber_denoise_Mono(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, res_b, state.res_b2,
                 np.int_(width), np.int_(height),
                 np.intc(params.flag_dyn_AANR),
                 np.intc(params.flag_ghost_reducer),
                 np.intc(params.val_ghost_reducer))
            )

            state.res_b2 = res_b.copy()

            # Apply 5% brightness boost
            tmp = self.cp.asarray(r_gpu).astype(self.cp.float64) * 1.05
            tmp = self.cp.clip(tmp, 0, 255)
            return self.cp.asarray(tmp, dtype=self.cp.uint8)

        return res_b

    def apply_denoise_paillou1_mono(self, res_b, params: FilterParams):
        """Apply Paillou denoise filter 1 for mono."""
        if params.flag_denoise_Paillou != 1:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        cell_size = 3
        sqr_cell_size = cell_size * cell_size
        r_gpu = res_b

        self.kernels.Denoise_Paillou_Mono(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, res_b, np.intc(width), np.intc(height),
             np.intc(cell_size), np.intc(sqr_cell_size))
        )

        return r_gpu

    def apply_denoise_paillou2_mono(self, res_b, params: FilterParams):
        """Apply Paillou denoise filter 2 for mono."""
        if params.flag_denoise_Paillou2 != 1:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        r_gpu = res_b

        self.kernels.reduce_noise_Mono(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, res_b, np.intc(width), np.intc(height))
        )

        return r_gpu

    def apply_nlm2_mono(self, res_b, params: FilterParams):
        """Apply Non-Local Means 2 denoising for mono."""
        if params.flag_NLM2 != 1:
            return res_b

        height, width = res_b.shape
        nb_ThreadsXs = 8
        nb_ThreadsYs = 8
        nb_blocksXs = (width // nb_ThreadsXs) + 1
        nb_blocksYs = (height // nb_ThreadsYs) + 1

        param = float(params.val_denoise)
        Noise = 1.0 / (param * param)
        lerpC = 0.4

        r_gpu = res_b

        self.kernels.NLM2_Mono_GPU(
            (nb_blocksXs, nb_blocksYs),
            (nb_ThreadsXs, nb_ThreadsYs),
            (r_gpu, res_b, np.intc(width), np.intc(height),
             np.float32(Noise), np.float32(lerpC))
        )

        return r_gpu

    def apply_knn_mono(self, res_b, params: FilterParams):
        """Apply K-Nearest Neighbors denoising for mono."""
        if params.flag_denoise_KNN != 1:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        param = float(params.val_denoise_KNN)
        Noise = 1.0 / (param * param)
        lerpC = 0.4

        r_gpu = res_b

        self.kernels.KNN_Mono_GPU(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, res_b, np.intc(width), np.intc(height),
             np.float32(Noise), np.float32(lerpC))
        )

        return r_gpu

    def apply_histogram_mono(self, res_b, params: FilterParams):
        """Apply histogram operations for mono."""
        if not (params.flag_histogram_stretch == 1 or
                params.flag_histogram_equalize2 == 1 or
                params.flag_histogram_phitheta == 1):
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        r_gpu = res_b

        self.kernels.Histo_Mono(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, res_b, np.int_(width), np.int_(height),
             np.intc(params.flag_histogram_stretch),
             np.float32(params.val_histo_min), np.float32(params.val_histo_max),
             np.intc(params.flag_histogram_equalize2), np.float32(params.val_heq2),
             np.intc(params.flag_histogram_phitheta),
             np.float32(params.val_phi), np.float32(params.val_theta))
        )

        return r_gpu

    def apply_amplification_mono(self, res_b, params: FilterParams):
        """Apply image amplification for mono."""
        if params.flag_AmpSoft != 1:
            return res_b

        if params.flag_lin_gauss not in (1, 2):
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        correction = self.cp.asarray(params.Corr_GS)
        r_gpu = res_b

        self.kernels.Mono_ampsoft_GPU(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, res_b, np.int_(width), np.int_(height),
             np.float32(params.val_ampl), correction)
        )

        return r_gpu

    def apply_star_amplification_mono(self, res_b, params: FilterParams):
        """Apply star amplification for mono."""
        if params.flag_AmpSoft != 1 or params.flag_lin_gauss != 3:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        niveau_blur = 7
        imagegreyblur = self.utils['gaussianblur_mono'](res_b, niveau_blur)
        correction = self.cp.asarray(params.Corr_GS)
        r_gpu = res_b

        self.kernels.Mono_staramp_GPU(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, res_b, imagegreyblur,
             np.int_(width), np.int_(height),
             np.float32(params.val_Mu), np.float32(params.val_Ro),
             np.float32(params.val_ampl), correction)
        )

        return r_gpu

    def apply_gradient_removal_mono(self, res_b, params: FilterParams):
        """Apply gradient removal or vignetting correction for mono."""
        if not params.flag_GR:
            return res_b

        if params.grad_vignet == 1:
            # Vignetting correction
            seuilb = int(self.cp.percentile(res_b, params.val_SGR))
            img_b = res_b.copy()
            img_b[img_b > seuilb] = seuilb

            niveau_blur = params.val_NGB * 2 + 3
            img_b = self.utils['gaussianblur_mono'](img_b, niveau_blur)

            att_b = self.cp.asarray(img_b) * ((100.0 - params.val_AGR) / 100.0)
            resb = self.cp.subtract(self.cp.asarray(res_b), att_b)
            resb = self.cp.clip(resb, 0, 255)

            return self.cp.asarray(resb, dtype=self.cp.uint8)
        else:
            # Gradient removal
            seuilb = int(self.cp.percentile(res_b, params.val_SGR))
            fd_b = res_b.copy()
            fd_b[fd_b > seuilb] = seuilb

            niveau_blur = params.val_NGB * 2 + 3
            fd_b = self.utils['gaussianblur_mono'](fd_b, niveau_blur)

            pivot_b = int(self.cp.percentile(self.cp.asarray(res_b), params.val_AGR))
            corr_b = (self.cp.asarray(res_b).astype(self.cp.int16) -
                      self.cp.asarray(fd_b).astype(self.cp.int16) + pivot_b)
            corr_b = self.cp.clip(corr_b, 0, 255)

            return self.cp.asarray(corr_b, dtype=self.cp.uint8)

    def apply_cll_mono(self, res_b, params: FilterParams):
        """Apply Contrast Low Light for mono."""
        if params.flag_CLL != 1:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        correction_CLL = self.cp.asarray(params.Corr_CLL, dtype=self.cp.uint8)
        r_gpu = res_b

        self.kernels.Contrast_Low_Light_Mono_GPU(
            (nb_blocksX, nb_blocksY),
            (params.nb_ThreadsX, params.nb_ThreadsY),
            (r_gpu, res_b, np.int_(width), np.int_(height), correction_CLL)
        )

        return r_gpu

    def apply_clahe_mono(self, res_b, params: FilterParams):
        """Apply CLAHE contrast enhancement for mono."""
        if params.flag_contrast_CLAHE != 1:
            return res_b

        if params.flag_OpenCvCuda:
            clahe = self.cv2.cuda.createCLAHE(
                clipLimit=params.val_contrast_CLAHE,
                tileGridSize=(params.val_grid_CLAHE, params.val_grid_CLAHE)
            )
            srcb = self.cv2.cuda_GpuMat()
            srcb.upload(res_b.get())
            resb = clahe.apply(srcb, self.cv2.cuda_Stream.Null())
            resbb = resb.download()
            return self.cp.asarray(resbb, dtype=self.cp.uint8)
        else:
            clahe = self.cv2.createCLAHE(
                clipLimit=params.val_contrast_CLAHE,
                tileGridSize=(params.val_grid_CLAHE, params.val_grid_CLAHE)
            )
            b = clahe.apply(res_b.get())
            return self.cp.asarray(b, dtype=self.cp.uint8)

    def apply_3fnr_back_mono(self, res_b, params: FilterParams,
                             state: FilterState):
        """Apply 3-frame noise reduction (back position) for mono."""
        if not params.flag_3FNRB or params.flag_image_mode:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_3FNRB < 4 and state.FNRB_First_Start:
            state.compteur_3FNRB += 1
            if state.compteur_3FNRB == 1:
                state.imgb1B = res_b.copy()
                state.img1_3FNROKB = True
            elif state.compteur_3FNRB == 2:
                state.imgb2B = res_b.copy()
                state.img2_3FNROKB = True
            elif state.compteur_3FNRB == 3:
                state.imgb3B = res_b.copy()
                state.img3_3FNROKB = True

        if state.img3_3FNROKB:
            if not state.FNRB_First_Start:
                state.imgb3B = res_b.copy()

            state.FNRB_First_Start = False
            b_gpu = res_b

            self.kernels.FNR_Mono(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (b_gpu, state.imgb1B, state.imgb2B, state.imgb3B,
                 np.int_(width), np.int_(height), np.float32(params.val_3FNR_Thres))
            )

            state.imgb1B = state.imgb2B.copy()
            state.imgb2B = b_gpu.copy()

            return b_gpu.copy()

        return res_b

    def apply_aanr_back_mono(self, res_b, params: FilterParams,
                             state: FilterState):
        """Apply Adaptive Absorber Noise Reduction (back) for mono."""
        if not params.flag_AANRB or params.flag_image_mode:
            return res_b

        height, width = res_b.shape
        nb_blocksX, nb_blocksY = self._compute_grid_dims(
            width, height, params.nb_ThreadsX, params.nb_ThreadsY)

        if state.compteur_AANRB < 3:
            state.compteur_AANRB += 1
            if state.compteur_AANRB == 1:
                state.res_b2B = res_b.copy()
                state.Im1fsdnOKB = True
            elif state.compteur_AANRB == 2:
                state.Im2fsdnOKB = True

        r_gpu = res_b

        if state.Im2fsdnOKB:
            local_dyn = 1
            local_GR = 0
            local_VGR = 0

            self.kernels.adaptative_absorber_denoise_Mono(
                (nb_blocksX, nb_blocksY),
                (params.nb_ThreadsX, params.nb_ThreadsY),
                (r_gpu, res_b, state.res_b2B,
                 np.int_(width), np.int_(height),
                 np.intc(local_dyn), np.intc(local_GR), np.intc(local_VGR))
            )

            state.res_b2B = res_b.copy()

            tmp = self.cp.asarray(r_gpu).astype(self.cp.float64) * 1.05
            tmp = self.cp.clip(tmp, 0, 255)
            return self.cp.asarray(tmp, dtype=self.cp.uint8)

        return res_b

    def apply_sharpen_mono(self, res_b, params: FilterParams, cupy_context):
        """Apply sharpening filters for mono."""
        res_s1_b1 = None
        res_s2_b1 = None

        # Sharpen 1
        if params.flag_sharpen_soft1 == 1:
            cupy_context.use()
            res_b1_blur = self.utils['gaussianblur_mono'](
                res_b, params.val_sigma_sharpen)

            tmp_b1 = self.cp.asarray(res_b).astype(self.cp.int16)
            tmp_b1 = tmp_b1 + params.val_sharpen * (tmp_b1 - res_b1_blur)
            tmp_b1 = self.cp.clip(tmp_b1, 0, 255)

            if params.flag_sharpen_soft2 == 1:
                res_s1_b1 = self.cp.asarray(tmp_b1, dtype=self.cp.uint8)
            else:
                res_b = self.cp.asarray(tmp_b1, dtype=self.cp.uint8)

        # Sharpen 2
        if params.flag_sharpen_soft2 == 1:
            cupy_context.use()
            res_b1_blur = self.utils['gaussianblur_mono'](
                res_b, params.val_sigma_sharpen2)

            tmp_b1 = self.cp.asarray(res_b).astype(self.cp.int16)
            tmp_b1 = tmp_b1 + params.val_sharpen2 * (tmp_b1 - res_b1_blur)
            tmp_b1 = self.cp.clip(tmp_b1, 0, 255)

            if params.flag_sharpen_soft1 == 1:
                res_s2_b1 = self.cp.asarray(tmp_b1, dtype=self.cp.uint8)
            else:
                res_b = self.cp.asarray(tmp_b1, dtype=self.cp.uint8)

        # Combine both sharpens
        if params.flag_sharpen_soft1 == 1 and params.flag_sharpen_soft2 == 1:
            res_b = res_s1_b1 // 2 + res_s2_b1 // 2

        return res_b

    def process_mono(self, res_b, params: FilterParams, state: FilterState,
                     cupy_context) -> np.ndarray:
        """
        Process a monochrome frame through the complete filter pipeline.

        Args:
            res_b: Mono image (CuPy array)
            params: Filter parameters
            state: Filter state for temporal operations
            cupy_context: CuPy CUDA stream context

        Returns:
            Processed grayscale image as numpy array
        """
        start_time = self.cv2.getTickCount()

        with cupy_context:
            if not params.flag_filtrage_ON:
                state.image_traitee = res_b.get()
                return state.image_traitee

            if params.flag_IsColor:
                # Should use process_color instead
                return state.image_traitee

            height, width = res_b.shape

            # Store original for demo mode
            image_base = None
            if params.flag_DEMO == 1:
                image_base = res_b.get()

            # Gaussian blur (pre-filter)
            if params.flag_GaussBlur:
                res_b = self.utils['gaussianblur_mono'](res_b, 3)

            # 1. Image negative
            if params.ImageNeg == 1:
                res_b = self.apply_negative_mono(res_b)

            # 2. Luminance estimation
            res_b = self.apply_luminance_estimate_mono(res_b, params)

            # 3. Frame stacking
            res_b = self.apply_frame_stacking_mono(res_b, params, state)

            # 4. Reduce variation
            res_b = self.apply_reduce_variation_mono(res_b, params, state)

            # 5. 3FNR front
            res_b = self.apply_3fnr_front_mono(res_b, params, state)

            # 5b. 3FNR2 front
            res_b = self.apply_3fnr2_front_mono(res_b, params, state)

            # 6. AANR front
            res_b = self.apply_aanr_front_mono(res_b, params, state)

            # 7. Denoise Paillou 1
            res_b = self.apply_denoise_paillou1_mono(res_b, params)

            # 8. Denoise Paillou 2
            res_b = self.apply_denoise_paillou2_mono(res_b, params)

            # 9. NLM2
            res_b = self.apply_nlm2_mono(res_b, params)

            # 10. KNN
            res_b = self.apply_knn_mono(res_b, params)

            # 11. Histogram
            res_b = self.apply_histogram_mono(res_b, params)

            # 12. Amplification
            res_b = self.apply_amplification_mono(res_b, params)

            # 13. Star amplification
            res_b = self.apply_star_amplification_mono(res_b, params)

            # 14. Gradient/vignetting
            res_b = self.apply_gradient_removal_mono(res_b, params)

            # 15. CLL
            res_b = self.apply_cll_mono(res_b, params)

            # 16. CLAHE
            res_b = self.apply_clahe_mono(res_b, params)

            # 17. 3FNR back
            res_b = self.apply_3fnr_back_mono(res_b, params, state)

            # 18. AANR back
            res_b = self.apply_aanr_back_mono(res_b, params, state)

            # 19-20. Sharpen
            res_b = self.apply_sharpen_mono(res_b, params, cupy_context)

            # Convert to output
            state.image_traitee = res_b.get()

            # Demo mode overlay
            if params.flag_DEMO == 1 and image_base is not None:
                if params.flag_demo_side == "Left":
                    state.image_traitee[0:height, 0:width//2] = image_base[0:height, 0:width//2]
                elif params.flag_demo_side == "Right":
                    state.image_traitee[0:height, width//2:width] = image_base[0:height, width//2:width]

        # Update timing
        stop_time = self.cv2.getTickCount()
        state.time_exec_test = int((stop_time - start_time) / self.cv2.getTickFrequency() * 1000)

        state.TTQueue.append(state.time_exec_test)
        if len(state.TTQueue) > 10:
            state.TTQueue.pop(0)
        state.curTT = sum(state.TTQueue) / len(state.TTQueue)

        return state.image_traitee


# ============================================================================
# Backward-Compatible Wrapper
# ============================================================================
#
# This section provides the same interface as the old exec()-based module,
# but internally uses the new FilterPipeline class.
#
# Usage in main script:
#   from filter_pipeline_new import init_filter_pipeline, application_filtrage_color, application_filtrage_mono
#   init_filter_pipeline(globals())  # Call once after CUDA kernels are loaded
#
# Then use application_filtrage_color() and application_filtrage_mono() as before.
# ============================================================================

# Global instances for backward compatibility
_pipeline_instance = None
_filter_state = None
_filter_params = None
_main_globals = None
_app_state = None  # New: ApplicationState reference for explicit state passing


def init_filter_pipeline(main_globals: dict, app_state=None):
    """
    Initialize the filter pipeline with references to the main script's globals.

    This must be called once after CUDA kernels are loaded and before
    calling application_filtrage_color() or application_filtrage_mono().

    Args:
        main_globals: The globals() dict from the main script
        app_state: Optional ApplicationState object. If provided, the pipeline
                   will use app_state.filter_state and app_state.filter_params
                   instead of syncing with globals.
    """
    global _pipeline_instance, _filter_state, _filter_params, _main_globals, _app_state

    _main_globals = main_globals
    _app_state = app_state  # Store reference to ApplicationState if provided

    # Import required modules from main globals
    cp = main_globals.get('cp')
    cv2 = main_globals.get('cv2')

    if cp is None or cv2 is None:
        raise RuntimeError("CuPy (cp) and OpenCV (cv2) must be available in globals")

    # Create FilterKernels from main globals
    kernels = FilterKernels(
        # RGB operations
        Set_RGB=main_globals.get('Set_RGB'),
        color_estimate_Mono=main_globals.get('color_estimate_Mono'),
        grey_estimate_Mono=main_globals.get('grey_estimate_Mono'),

        # Variation reduction
        reduce_variation_Color=main_globals.get('reduce_variation_Color'),
        reduce_variation_Mono=main_globals.get('reduce_variation_Mono'),

        # 3FNR
        FNR_Color=main_globals.get('FNR_Color'),
        FNR_Mono=main_globals.get('FNR_Mono'),
        FNR2_Color=main_globals.get('FNR2_Color'),
        FNR2_Mono=main_globals.get('FNR2_Mono'),

        # AANR
        adaptative_absorber_denoise_Color=main_globals.get('adaptative_absorber_denoise_Color'),
        adaptative_absorber_denoise_Mono=main_globals.get('adaptative_absorber_denoise_Mono'),

        # Denoise
        Denoise_Paillou_Colour=main_globals.get('Denoise_Paillou_Colour'),
        Denoise_Paillou_Mono=main_globals.get('Denoise_Paillou_Mono'),
        reduce_noise_Color=main_globals.get('reduce_noise_Color'),
        reduce_noise_Mono=main_globals.get('reduce_noise_Mono'),
        NLM2_Colour_GPU=main_globals.get('NLM2_Colour_GPU'),
        NLM2_Mono_GPU=main_globals.get('NLM2_Mono_GPU'),
        KNN_Colour_GPU=main_globals.get('KNN_Colour_GPU'),
        KNN_Mono_GPU=main_globals.get('KNN_Mono_GPU'),

        # Histogram & Amplification
        Histo_Color=main_globals.get('Histo_Color'),
        Histo_Mono=main_globals.get('Histo_Mono'),
        Colour_ampsoft_GPU=main_globals.get('Colour_ampsoft_GPU'),
        Mono_ampsoft_GPU=main_globals.get('Mono_ampsoft_GPU'),
        Colour_staramp_GPU=main_globals.get('Colour_staramp_GPU'),
        Mono_staramp_GPU=main_globals.get('Mono_staramp_GPU'),

        # Contrast
        Contrast_Low_Light_Colour_GPU=main_globals.get('Contrast_Low_Light_Colour_GPU'),
        Contrast_Low_Light_Mono_GPU=main_globals.get('Contrast_Low_Light_Mono_GPU'),

        # Saturation
        Saturation_Colour=main_globals.get('Saturation_Colour'),
        Saturation_Combine_Colour=main_globals.get('Saturation_Combine_Colour'),
    )

    # Create utility functions dict
    utility_funcs = {
        'gaussianblur_colour': main_globals.get('gaussianblur_colour'),
        'gaussianblur_mono': main_globals.get('gaussianblur_mono'),
        'image_negative_colour': main_globals.get('image_negative_colour'),
        'cupy_separateRGB_2_numpy_RGBimage': main_globals.get('cupy_separateRGB_2_numpy_RGBimage'),
        'numpy_RGBImage_2_cupy_separateRGB': main_globals.get('numpy_RGBImage_2_cupy_separateRGB'),
        'Image_Quality': main_globals.get('Image_Quality'),
    }

    # Create pipeline instance
    _pipeline_instance = FilterPipeline(cp, cv2, kernels, utility_funcs)

    # Create or use existing filter state
    if app_state is not None and hasattr(app_state, 'filter_state') and app_state.filter_state is not None:
        # Use app_state's filter_state
        _filter_state = app_state.filter_state
    else:
        # Create new filter state
        _filter_state = FilterState()

    # Create or use existing filter params
    if app_state is not None and hasattr(app_state, 'filter_params') and app_state.filter_params is not None:
        # Use app_state's filter_params
        _filter_params = app_state.filter_params
    else:
        # Will be created from globals on each call
        _filter_params = None

    # Initialize transfer function arrays if not already set
    if _filter_state.trsf_r is None:
        _filter_state.trsf_r = np.zeros(256, dtype=np.int32)
    if _filter_state.trsf_g is None:
        _filter_state.trsf_g = np.zeros(256, dtype=np.int32)
    if _filter_state.trsf_b is None:
        _filter_state.trsf_b = np.zeros(256, dtype=np.int32)

    # Sync filter params from globals to app_state.filter_params (one-time init)
    # After this, callbacks are responsible for keeping filter_params updated
    sync_filter_params_from_globals(main_globals, app_state)


def sync_filter_params_from_globals(main_globals: dict = None, app_state=None):
    """
    Sync filter params from main globals to app_state.filter_params.

    This is called once at initialization to populate app_state.filter_params
    with the initial values from globals. After this, callbacks are responsible
    for keeping app_state.filter_params updated directly.

    Args:
        main_globals: The globals() dict from the main script. If None, uses _main_globals.
        app_state: The ApplicationState object. If None, uses _app_state.
    """
    g = main_globals if main_globals is not None else _main_globals
    state = app_state if app_state is not None else _app_state

    if g is None or state is None or state.filter_params is None:
        return

    fp = state.filter_params

    # Sync all filter params from globals
    fp.flag_filtrage_ON = g.get('flag_filtrage_ON', True)
    fp.flag_IsColor = g.get('flag_IsColor', True)
    fp.flag_image_mode = g.get('flag_image_mode', False)
    fp.flag_TRSF = g.get('flag_TRSF', 0)
    fp.flag_DEMO = g.get('flag_DEMO', 0)
    fp.flag_demo_side = g.get('flag_demo_side', 'Left')
    fp.flag_reverse_RB = g.get('flag_reverse_RB', 0)
    fp.flag_GaussBlur = g.get('flag_GaussBlur', False)
    fp.val_reds = g.get('val_reds', 1.0)
    fp.val_greens = g.get('val_greens', 1.0)
    fp.val_blues = g.get('val_blues', 1.0)
    fp.ImageNeg = g.get('ImageNeg', 0)
    fp.flag_NB_estime = g.get('flag_NB_estime', 0)
    fp.val_FS = g.get('val_FS', 1)
    fp.stack_div = g.get('stack_div', 1)
    fp.flag_reduce_variation = g.get('flag_reduce_variation', False)
    fp.val_reduce_variation = g.get('val_reduce_variation', 50.0)
    fp.flag_BFReference = g.get('flag_BFReference', 'PreviousFrame')
    fp.flag_3FNR = g.get('flag_3FNR', False)
    fp.flag_3FNRB = g.get('flag_3FNRB', False)
    fp.flag_3FNR2 = g.get('flag_3FNR2', False)
    fp.flag_3FNR2B = g.get('flag_3FNR2B', False)
    fp.val_3FNR_Thres = g.get('val_3FNR_Thres', 1.0)
    fp.val_3FNR_Threshold = g.get('val_3FNR_Thres', 1.0)  # Alias
    fp.flag_AANR = g.get('flag_AANR', False)
    fp.flag_AANRB = g.get('flag_AANRB', False)
    fp.flag_dyn_AANR = g.get('flag_dyn_AANR', 0)
    fp.flag_ghost_reducer = g.get('flag_ghost_reducer', 0)
    fp.val_ghost_reducer = g.get('val_ghost_reducer', 0)
    fp.flag_denoise_Paillou = g.get('flag_denoise_Paillou', 0)
    fp.flag_denoise_Paillou2 = g.get('flag_denoise_Paillou2', 0)
    fp.flag_NLM2 = g.get('flag_NLM2', 0)
    fp.val_denoise = g.get('val_denoise', 10.0)
    fp.flag_denoise_KNN = g.get('flag_denoise_KNN', 0)
    fp.flag_KNN = bool(g.get('flag_denoise_KNN', 0))  # Alias
    fp.val_denoise_KNN = g.get('val_denoise_KNN', 10.0)
    fp.flag_BFREFPT = g.get('flag_BFREFPT', False)
    fp.flag_histogram_stretch = g.get('flag_histogram_stretch', 0)
    fp.flag_histogram_equalize = bool(g.get('flag_histogram_equalize2', 0))  # Alias
    fp.flag_histogram_equalize2 = g.get('flag_histogram_equalize2', 0)
    fp.flag_histogram_phitheta = g.get('flag_histogram_phitheta', 0)
    fp.val_histo_min = g.get('val_histo_min', 0.0)
    fp.val_histo_max = g.get('val_histo_max', 255.0)
    fp.val_heq2 = g.get('val_heq2', 1.0)
    fp.val_phi = g.get('val_phi', 1.0)
    fp.val_theta = g.get('val_theta', 128.0)
    fp.flag_AmpSoft = g.get('flag_AmpSoft', 0)
    fp.flag_lin_gauss = g.get('flag_lin_gauss', 1)
    fp.val_ampl = g.get('val_ampl', 1.0)
    fp.val_Mu = g.get('val_Mu', 0.5)
    fp.val_Ro = g.get('val_Ro', 0.5)
    fp.Corr_GS = g.get('Corr_GS')
    fp.flag_GR = g.get('flag_GR', False)
    fp.grad_vignet = g.get('grad_vignet', 1)
    fp.val_SGR = g.get('val_SGR', 50.0)
    fp.val_NGB = g.get('val_NGB', 5)
    fp.val_AGR = g.get('val_AGR', 50.0)
    fp.flag_CLL = g.get('flag_CLL', 0)
    fp.Corr_CLL = g.get('Corr_CLL')
    fp.flag_contrast_CLAHE = g.get('flag_contrast_CLAHE', 0)
    fp.val_contrast_CLAHE = g.get('val_contrast_CLAHE', 2.0)
    fp.val_grid_CLAHE = g.get('val_grid_CLAHE', 8)
    fp.flag_OpenCvCuda = g.get('flag_OpenCvCuda', False)
    fp.flag_SAT = g.get('flag_SAT', False)
    fp.flag_SAT_Image = g.get('flag_SAT_Image', False)
    fp.flag_SAT2PASS = g.get('flag_SAT2PASS', False)
    fp.val_SAT = g.get('val_SAT', 1.0)
    fp.flag_sharpen_soft1 = g.get('flag_sharpen_soft1', 0)
    fp.flag_sharpen_soft2 = g.get('flag_sharpen_soft2', 0)
    fp.val_sharpen = g.get('val_sharpen', 1.0)
    fp.val_sharpen2 = g.get('val_sharpen2', 1.0)
    fp.val_sigma_sharpen = g.get('val_sigma_sharpen', 3)
    fp.val_sigma_sharpen2 = g.get('val_sigma_sharpen2', 3)
    fp.res_cam_x = g.get('res_cam_x', 1920)
    fp.res_cam_y = g.get('res_cam_y', 1080)
    fp.delta_tx = g.get('delta_tx', 0)
    fp.delta_ty = g.get('delta_ty', 0)
    fp.IQ_Method = g.get('IQ_Method', 0)
    fp.nb_ThreadsX = g.get('nb_ThreadsX', 32)
    fp.nb_ThreadsY = g.get('nb_ThreadsY', 32)


def _get_filter_params() -> FilterParams:
    """
    Get FilterParams for the filter pipeline.

    Returns app_state.filter_params as the single source of truth.
    The params are initialized from globals once at startup via
    sync_filter_params_from_globals(), then updated directly by callbacks.
    """
    if _app_state is not None and _app_state.filter_params is not None:
        return _app_state.filter_params

    # Fallback: create minimal FilterParams (should not happen in normal operation)
    return FilterParams()


def _sync_state_to_globals():
    """
    Sync filter state back to main globals after processing.

    When app_state.filter_state is being used as the source of truth,
    this function only syncs timing info (which may still be read from globals
    by the refresh loop). In the future, this can be removed entirely.
    """
    # Skip full sync if using app_state.filter_state as source of truth
    if _app_state is not None and _app_state.filter_state is _filter_state:
        # Only sync timing info and output image to globals
        g = _main_globals
        s = _filter_state
        g['time_exec_test'] = s.time_exec_test
        g['curTT'] = s.curTT
        g['TTQueue'] = s.TTQueue
        # Critical: sync the processed image output
        if s.image_traitee is not None:
            g['image_traitee'] = s.image_traitee
        return

    g = _main_globals
    s = _filter_state

    # Timing
    g['time_exec_test'] = s.time_exec_test
    g['curTT'] = s.curTT
    g['TTQueue'] = s.TTQueue

    # Frame stacking state
    g['compteur_FS'] = s.compteur_FS
    g['Im1OK'] = s.Im1OK
    g['Im2OK'] = s.Im2OK
    g['Im3OK'] = s.Im3OK
    g['Im4OK'] = s.Im4OK
    g['Im5OK'] = s.Im5OK
    g['b1_sm'] = s.b1_sm
    g['b2_sm'] = s.b2_sm
    g['b3_sm'] = s.b3_sm
    g['b4_sm'] = s.b4_sm
    g['b5_sm'] = s.b5_sm
    g['g1_sm'] = s.g1_sm
    g['g2_sm'] = s.g2_sm
    g['g3_sm'] = s.g3_sm
    g['g4_sm'] = s.g4_sm
    g['g5_sm'] = s.g5_sm
    g['r1_sm'] = s.r1_sm
    g['r2_sm'] = s.r2_sm
    g['r3_sm'] = s.r3_sm
    g['r4_sm'] = s.r4_sm
    g['r5_sm'] = s.r5_sm

    # Reduce variation state
    g['compteur_RV'] = s.compteur_RV
    g['Im1rvOK'] = s.Im1rvOK
    g['Im2rvOK'] = s.Im2rvOK
    g['res_b2'] = s.res_b2
    g['res_g2'] = s.res_g2
    g['res_r2'] = s.res_r2

    # AANR state
    g['compteur_AANR'] = s.compteur_AANR
    g['Im1fsdnOK'] = s.Im1fsdnOK
    g['Im2fsdnOK'] = s.Im2fsdnOK
    g['compteur_AANRB'] = s.compteur_AANRB
    g['Im1fsdnOKB'] = s.Im1fsdnOKB
    g['Im2fsdnOKB'] = s.Im2fsdnOKB
    g['res_b2B'] = s.res_b2B
    g['res_g2B'] = s.res_g2B
    g['res_r2B'] = s.res_r2B

    # 3FNR state
    g['compteur_3FNR'] = s.compteur_3FNR
    g['img1_3FNROK'] = s.img1_3FNROK
    g['img2_3FNROK'] = s.img2_3FNROK
    g['img3_3FNROK'] = s.img3_3FNROK
    g['FNR_First_Start'] = s.FNR_First_Start
    g['imgb1'] = s.imgb1
    g['imgg1'] = s.imgg1
    g['imgr1'] = s.imgr1
    g['imgb2'] = s.imgb2
    g['imgg2'] = s.imgg2
    g['imgr2'] = s.imgr2
    g['imgb3'] = s.imgb3
    g['imgg3'] = s.imgg3
    g['imgr3'] = s.imgr3

    # 3FNR back state
    g['compteur_3FNRB'] = s.compteur_3FNRB
    g['img1_3FNROKB'] = s.img1_3FNROKB
    g['img2_3FNROKB'] = s.img2_3FNROKB
    g['img3_3FNROKB'] = s.img3_3FNROKB
    g['FNRB_First_Start'] = s.FNRB_First_Start
    g['imgb1B'] = s.imgb1B
    g['imgg1B'] = s.imgg1B
    g['imgr1B'] = s.imgr1B
    g['imgb2B'] = s.imgb2B
    g['imgg2B'] = s.imgg2B
    g['imgr2B'] = s.imgr2B
    g['imgb3B'] = s.imgb3B
    g['imgg3B'] = s.imgg3B
    g['imgr3B'] = s.imgr3B

    # 3FNR2 state
    g['compteur_3FNR2'] = s.compteur_3FNR2
    g['img1_3FNR2OK'] = s.img1_3FNR2OK
    g['img2_3FNR2OK'] = s.img2_3FNR2OK
    g['img3_3FNR2OK'] = s.img3_3FNR2OK
    g['FNR2_First_Start'] = s.FNR2_First_Start
    g['imgb21'] = s.imgb21
    g['imgg21'] = s.imgg21
    g['imgr21'] = s.imgr21
    g['imgb22'] = s.imgb22
    g['imgg22'] = s.imgg22
    g['imgr22'] = s.imgr22
    g['imgb23'] = s.imgb23
    g['imgg23'] = s.imgg23
    g['imgr23'] = s.imgr23

    # 3FNR2 back state
    g['compteur_3FNR2B'] = s.compteur_3FNR2B
    g['img1_3FNR2OKB'] = s.img1_3FNR2OKB
    g['img2_3FNR2OKB'] = s.img2_3FNR2OKB
    g['img3_3FNR2OKB'] = s.img3_3FNR2OKB
    g['FNR2B_First_Start'] = s.FNR2B_First_Start
    g['imgb21B'] = s.imgb21B
    g['imgg21B'] = s.imgg21B
    g['imgr21B'] = s.imgr21B
    g['imgb22B'] = s.imgb22B
    g['imgg22B'] = s.imgg22B
    g['imgr22B'] = s.imgr22B
    g['imgb23B'] = s.imgb23B
    g['imgg23B'] = s.imgg23B
    g['imgr23B'] = s.imgr23B

    # Best frame reference
    g['BFREF_image_PT'] = s.BFREF_image_PT
    g['flag_BFREF_image_PT'] = s.flag_BFREF_image_PT
    g['max_qual_PT'] = s.max_qual_PT

    # Transfer functions
    g['trsf_r'] = s.trsf_r
    g['trsf_g'] = s.trsf_g
    g['trsf_b'] = s.trsf_b

    # Output image
    g['image_traitee'] = s.image_traitee


def _sync_globals_to_state():
    """
    Sync main globals to filter state before processing.
    This includes re-syncing filter params from globals to ensure
    checkbox changes are reflected in the filter pipeline.
    """
    g = _main_globals
    s = _filter_state

    # Best frame reference (read from globals)
    s.BFREF_image = g.get('BFREF_image')
    s.flag_BFREF_image = g.get('flag_BFREF_image', False)
    
    # Re-sync filter params from globals (critical for checkbox changes)
    sync_filter_params_from_globals(_main_globals, _app_state)


def application_filtrage_color(res_b1, res_g1, res_r1):
    """
    Process a color frame through the filter pipeline.

    This is a backward-compatible wrapper that uses the new FilterPipeline
    class internally while maintaining the same function signature.

    Args:
        res_b1: Blue channel (CuPy array)
        res_g1: Green channel (CuPy array)
        res_r1: Red channel (CuPy array)
    """
    global _pipeline_instance, _filter_state, _main_globals

    if _pipeline_instance is None:
        raise RuntimeError(
            "Filter pipeline not initialized. "
            "Call init_filter_pipeline(globals()) first."
        )

    # Sync globals to state
    _sync_globals_to_state()

    # Get current params from globals
    params = _get_filter_params()

    # Get cupy context from globals
    cupy_context = _main_globals.get('cupy_context')

    # Process the frame
    _pipeline_instance.process_color(
        res_b1, res_g1, res_r1, params, _filter_state, cupy_context
    )

    # Sync state back to globals
    _sync_state_to_globals()


def application_filtrage_mono(res_b1):
    """
    Process a monochrome frame through the filter pipeline.

    This is a backward-compatible wrapper that uses the new FilterPipeline
    class internally while maintaining the same function signature.

    Args:
        res_b1: Mono image (CuPy array)
    """
    global _pipeline_instance, _filter_state, _main_globals

    if _pipeline_instance is None:
        raise RuntimeError(
            "Filter pipeline not initialized. "
            "Call init_filter_pipeline(globals()) first."
        )

    # Sync globals to state
    _sync_globals_to_state()

    # Get current params from globals
    params = _get_filter_params()
    params.flag_IsColor = False  # Force mono mode

    # Get cupy context from globals
    cupy_context = _main_globals.get('cupy_context')

    # Process the frame
    _pipeline_instance.process_mono(
        res_b1, params, _filter_state, cupy_context
    )

    # Sync state back to globals
    _sync_state_to_globals()


# Legacy create functions - now just return empty string since
# the wrapper functions above replace the need for exec()
def create_filter_pipeline_color():
    """
    Legacy function - returns empty string.
    Use init_filter_pipeline() and application_filtrage_color() instead.
    """
    return ''


def create_filter_pipeline_mono():
    """
    Legacy function - returns empty string.
    Use init_filter_pipeline() and application_filtrage_mono() instead.
    """
    return ''
