# JETSONSKY Architecture Analysis

## Current Monolithic Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│        JetsonSky_Linux_Windows_V53_07RC.py (11,301 lines)                  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        300+ Global Variables                          │ │
│  │  val_exposition, val_gain, res_cam_x, res_cam_y, FlipV, FlipH, ...  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        143 Module-Level Functions                     │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │ │
│  │  │  GUI Functions  │  │ Camera Functions │  │  Filter Functions  │ │ │
│  │  │  (~50 funcs)    │  │   (~30 funcs)    │  │    (~40 funcs)     │ │ │
│  │  │                 │  │                  │  │                    │ │ │
│  │  │ commande_flipV()│  │ init_camera()    │  │ application_       │ │ │
│  │  │ commande_flipH()│  │ camera_          │  │   filtrage_color() │ │ │
│  │  │ choix_BIN1()    │  │   acquisition()  │  │ application_       │ │ │
│  │  │ choix_BIN2()    │  │ start_           │  │   filtrage_mono()  │ │ │
│  │  │ load_video()    │  │   acquisition()  │  │ gaussianblur()     │ │ │
│  │  │ load_image()    │  │ stop_            │  │ sharpen_soft1()    │ │ │
│  │  │ ...             │  │   acquisition()  │  │ ...                │ │ │
│  │  └─────────────────┘  └──────────────────┘  └────────────────────┘ │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │ │
│  │  │  AI Functions   │  │  I/O Functions   │  │  Mount Functions   │ │ │
│  │  │  (~8 funcs)     │  │   (~10 funcs)    │  │    (~5 funcs)      │ │ │
│  │  │                 │  │                  │  │                    │ │ │
│  │  │ satellites_     │  │ video_capture()  │  │ mount_info()       │ │ │
│  │  │   tracking_AI() │  │ pic_capture()    │  │ calcul_AZ_HT_     │ │ │
│  │  │ stars_          │  │ start_video_     │  │   cible()          │ │ │
│  │  │   detection()   │  │   capture()      │  │ Mount_            │ │ │
│  │  │ ...             │  │ ...              │  │   calibration()    │ │ │
│  │  └─────────────────┘  └──────────────────┘  └────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                           3 Thread Classes                            │ │
│  │                                                                       │ │
│  │  keyboard_management(Thread)    acquisition_mount(Thread)            │ │
│  │  acquisition(Thread)                                                 │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    Tkinter GUI Code (2,500+ lines)                    │ │
│  │  Button(), Label(), Scale(), Frame() creation scattered throughout    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

            ↓ Dependencies ↓

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ zwoasi_cupy/ │  │   zwoefw/    │  │   synscan/   │  │   Serfile/   │
│ (1,052 lines)│  │ (264 lines)  │  │ (929 lines)  │  │ (339 lines)  │
│              │  │              │  │              │  │              │
│ Camera SDK   │  │ Filter Wheel │  │ Mount Control│  │ SER File I/O │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

### Problem Visualization

```
┌───────────────────────────────────────────────────────────────┐
│                     EVERYTHING IN ONE FILE                    │
│                                                               │
│  GUI ←──→ Business Logic ←──→ Hardware ←──→ File I/O        │
│   ↕          ↕                  ↕              ↕              │
│  ALL FUNCTIONS ACCESS ALL GLOBAL VARIABLES                    │
│   ↕          ↕                  ↕              ↕              │
│  NO BOUNDARIES • NO ENCAPSULATION • NO TESTABILITY            │
└───────────────────────────────────────────────────────────────┘
```

### Current Data Flow (Spaghetti Pattern)

```
                        Global State (300+ vars)
                                 │
                    ┌────────────┼────────────┐
                    ↓            ↓            ↓
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │   GUI    │ │  Camera  │ │ Filters  │
              │ Functions│ │ Functions│ │ Functions│
              └──────────┘ └──────────┘ └──────────┘
                    ↓            ↓            ↓
                    └────────────┼────────────┘
                                 │
                        Global State Modified
                                 │
                    ┌────────────┼────────────┐
                    ↓            ↓            ↓
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │   I/O    │ │   AI     │ │  Mount   │
              │ Functions│ │ Functions│ │ Functions│
              └──────────┘ └──────────┘ └──────────┘

   Problem: Any function can modify any state at any time!
   Result: Impossible to reason about program behavior
```

### Function Coupling Example

```
commande_flipV()
    │
    ├──> Modifies global: FlipV
    ├──> Modifies global: choix_flipV
    ├──> Reads global: flag_filtrage_ON
    ├──> Calls: Button1.config() (GUI)
    └──> No return value
         │
         ├──> camera_acquisition() reads FlipV
         ├──> refresh() reads FlipV
         ├──> application_filtrage_color() reads FlipV
         └──> reconstruction_image() reads FlipV

    TIGHT COUPLING: 1 function affects 10+ others through globals!
```

---

## Proposed Modular Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          JetsonSky Application                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
        ┌───────────────────────┐       ┌───────────────────────┐
        │      main.py          │       │   config.yaml         │
        │   (Entry Point)       │       │  (Configuration)      │
        │    < 100 lines        │       │                       │
        └───────────────────────┘       └───────────────────────┘
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
┌──────────┐  ┌──────────┐  ┌──────────┐
│   gui/   │  │  core/   │  │ hardware/│
│          │  │          │  │          │
└──────────┘  └──────────┘  └──────────┘
        ↓           ↓           ↓
┌──────────┐  ┌──────────┐  ┌──────────┐
│ filters/ │  │   ai/    │  │   io/    │
│          │  │          │  │          │
└──────────┘  └──────────┘  └──────────┘
        ↓
┌──────────┐
│  utils/  │
│          │
└──────────┘
```

### Detailed Module Structure

```
JetsonSky/
│
├── main.py (< 100 lines)
│   └─> Entry point, initialization, main loop
│
├── config.yaml
│   └─> Application configuration
│
├── gui/ (Presentation Layer)
│   ├── __init__.py
│   ├── main_window.py          (~200 lines)
│   │   └─> MainWindow class, layout, widgets
│   ├── controls.py             (~150 lines)
│   │   └─> Control panel widgets
│   ├── display.py              (~100 lines)
│   │   └─> Image display canvas
│   └── dialogs.py              (~100 lines)
│       └─> Dialog windows
│
├── core/ (Business Logic Layer)
│   ├── __init__.py
│   ├── controller.py           (~300 lines)
│   │   └─> JetsonSkyController (MVC Controller)
│   ├── camera.py               (~200 lines)
│   │   └─> Camera class (abstraction)
│   ├── camera_models.py        (~400 lines)
│   │   └─> CAMERA_MODELS registry
│   ├── image_processor.py      (~250 lines)
│   │   └─> ImageProcessor class
│   └── config.py               (~200 lines)
│       ├─> CameraConfig dataclass
│       ├─> ProcessingState dataclass
│       └─> AppState dataclass
│
├── filters/ (Image Processing Layer)
│   ├── __init__.py
│   ├── base.py                 (~50 lines)
│   │   └─> Filter abstract base class
│   ├── pipeline.py             (~100 lines)
│   │   └─> FilterPipeline class
│   ├── denoise.py              (~300 lines)
│   │   ├─> DenoiseKNNFilter
│   │   ├─> DenoiseNLM2Filter
│   │   ├─> DenoisePaillouFilter
│   │   └─> AdaptiveAbsorberFilter
│   ├── sharpen.py              (~150 lines)
│   │   ├─> SharpenSoft1Filter
│   │   └─> SharpenSoft2Filter
│   ├── contrast.py             (~200 lines)
│   │   ├─> ContrastCLAHEFilter
│   │   ├─> ContrastLowLightFilter
│   │   └─> HistogramStretchFilter
│   └── color.py                (~150 lines)
│       ├─> ColorBalanceFilter
│       └─> SaturationFilter
│
├── ai/ (AI/ML Layer)
│   ├── __init__.py
│   ├── detector.py             (~100 lines)
│   │   └─> BaseDetector abstract class
│   ├── crater_detector.py      (~150 lines)
│   │   └─> CraterDetector (YOLOv8)
│   └── satellite_detector.py   (~200 lines)
│       └─> SatelliteDetector (YOLOv8)
│
├── hardware/ (Hardware Abstraction Layer)
│   ├── __init__.py
│   ├── zwoasi_cupy/            (existing - 1,052 lines)
│   │   └─> Camera SDK interface
│   ├── zwoefw/                 (existing - 264 lines)
│   │   └─> Filter wheel interface
│   └── synscan/                (existing - 929 lines)
│       └─> Mount control interface
│
├── io/ (I/O Layer)
│   ├── __init__.py
│   ├── image_io.py             (~200 lines)
│   │   ├─> load_image()
│   │   ├─> save_image()
│   │   └─> ImageCapture class
│   ├── video_io.py             (~250 lines)
│   │   ├─> load_video()
│   │   ├─> save_video()
│   │   └─> VideoCapture class
│   └── serfile.py              (moved from Serfile/)
│       └─> SER format handler
│
└── utils/ (Utilities Layer)
    ├── __init__.py
    ├── constants.py            (~100 lines)
    │   └─> All magic numbers and constants
    ├── threading.py            (~150 lines)
    │   ├─> KeyboardThread
    │   ├─> AcquisitionThread
    │   └─> MountThread
    └── astronomy.py            (~150 lines)
        ├─> calc_jour_julien()
        ├─> calc_heure_siderale()
        └─> coordinate conversions
```

### Layer Architecture (Clean Separation)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │MainWindow  │  │ Controls   │  │  Display   │                │
│  │  (View)    │  │  (View)    │  │  (View)    │                │
│  └────────────┘  └────────────┘  └────────────┘                │
└────────────────────────────┬────────────────────────────────────┘
                             │ Events
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Business Logic Layer                        │
│  ┌──────────────────────────────────────────────────┐          │
│  │         JetsonSkyController (Controller)         │          │
│  └──────────────────────────────────────────────────┘          │
│                             │                                   │
│     ┌───────────────────────┼───────────────────────┐          │
│     ↓                       ↓                       ↓          │
│  ┌─────────┐         ┌──────────────┐        ┌──────────┐     │
│  │ Camera  │         │ImageProcessor│        │ AppState │     │
│  └─────────┘         └──────────────┘        └──────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ↓                ↓                ↓
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│  Filter Layer   │  │   AI Layer   │  │   I/O Layer  │
│                 │  │              │  │              │
│ ┌─────────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │
│ │   Denoise   │ │  │ │ Crater   │ │  │ │  Image   │ │
│ │   Sharpen   │ │  │ │ Satellite│ │  │ │  Video   │ │
│ │  Contrast   │ │  │ │ Detector │ │  │ │  I/O     │ │
│ └─────────────┘ │  │ └──────────┘ │  │ └──────────┘ │
└─────────────────┘  └──────────────┘  └──────────────┘
            │                │                │
            └────────────────┼────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Hardware Layer                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │  ZWO ASI   │  │ Filter     │  │  Mount     │                │
│  │  Camera    │  │ Wheel      │  │  Control   │                │
│  └────────────┘  └────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────┘

Key Principle: Each layer only depends on layers below it
              No circular dependencies!
```

### Data Flow (Clean Pattern)

```
User Interaction
      │
      ↓
┌──────────────┐
│     View     │ (GUI displays state, emits events)
└──────────────┘
      │
      │ Events (button clicks, sliders)
      ↓
┌──────────────┐
│  Controller  │ (Handles events, updates model)
└──────────────┘
      │
      │ Commands
      ↓
┌──────────────┐
│    Model     │ (Business logic, state management)
│  (AppState)  │
└──────────────┘
      │
      │ State changes
      ↓
┌──────────────┐
│     View     │ (Updates UI to reflect new state)
└──────────────┘

Example: User clicks "Flip Vertical" button
1. View: Button click event → Controller.toggle_flip_vertical()
2. Controller: Updates AppState.processing_state.flip_vertical = True
3. Controller: Calls View.update_flip_button(True)
4. View: Updates button color to turquoise
```

### Filter Pipeline Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    FilterPipeline                          │
│                                                            │
│  filters = [                                               │
│    DenoiseKNNFilter(enabled=True, strength=0.2),          │
│    SharpenSoft1Filter(enabled=True, amount=1.5),          │
│    ContrastCLAHEFilter(enabled=True, clip_limit=2.0),     │
│    SaturationFilter(enabled=True, saturation=1.2),        │
│    ...                                                     │
│  ]                                                         │
│                                                            │
│  def apply_all(image):                                     │
│    result = image                                          │
│    for filter in filters:                                  │
│      if filter.enabled:                                    │
│        result = filter.apply(result)                       │
│    return result                                           │
└────────────────────────────────────────────────────────────┘
                            │
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ DenoiseKNN   │    │ SharpenSoft1 │    │ Contrast     │
│  Filter      │    │   Filter     │    │ CLAHE Filter │
│              │    │              │    │              │
│ .enabled     │    │ .enabled     │    │ .enabled     │
│ .strength    │    │ .amount      │    │ .clip_limit  │
│ .apply()     │    │ .apply()     │    │ .apply()     │
└──────────────┘    └──────────────┘    └──────────────┘

Benefits:
- Each filter is independent and testable
- Easy to add/remove filters
- Clear execution order
- Can be configured dynamically
```

### Camera Model Registry Pattern

```
┌─────────────────────────────────────────────────────────────┐
│              CAMERA_MODELS Dictionary                       │
│                                                             │
│  "ZWO ASI1600MC": CameraConfig(                            │
│    resolution_x=4656,                                       │
│    resolution_y=3520,                                       │
│    sensor_factor="4_3",                                     │
│    supported_resolutions_bin1=[(4656,3520), ...],          │
│    ...                                                      │
│  ),                                                         │
│                                                             │
│  "ZWO ASI294MC": CameraConfig(                             │
│    resolution_x=4144,                                       │
│    resolution_y=2822,                                       │
│    ...                                                      │
│  ),                                                         │
│                                                             │
│  "ZWO ASI178MC": CameraConfig(...),                        │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Lookup by model name
                            ↓
                  get_camera_config(model_name)
                            │
                            │ Returns
                            ↓
                    CameraConfig instance
                            │
                            ↓
              ┌─────────────────────────┐
              │     Camera class        │
              │                         │
              │  def __init__(config):  │
              │    self.config = config │
              │    self.res_x = ...     │
              │    self.res_y = ...     │
              └─────────────────────────┘

Replaces: 1,500 lines of if-elif chains with a simple dictionary lookup!
```

---

## Component Interaction Diagrams

### Image Acquisition Flow

```
┌──────────┐                                    ┌──────────────┐
│   User   │                                    │   Camera     │
│          │                                    │  (Hardware)  │
└────┬─────┘                                    └──────┬───────┘
     │                                                 │
     │ 1. Click "Start Acquisition"                   │
     ↓                                                 │
┌────────────┐                                        │
│    View    │                                        │
│ (MainWindow)│                                       │
└────┬───────┘                                        │
     │                                                 │
     │ 2. on_start_acquisition()                      │
     ↓                                                 │
┌────────────────┐                                    │
│   Controller   │                                    │
│ (JetsonSky)    │                                    │
└────┬───────────┘                                    │
     │                                                 │
     │ 3. start_acquisition()                         │
     ↓                                                 │
┌────────────────┐                                    │
│ AcquisitionThread│                                  │
│   (Thread)     │                                    │
└────┬───────────┘                                    │
     │                                                 │
     │ 4. camera.capture()                            │
     │────────────────────────────────────────────────>│
     │                                                 │
     │ 5. raw_image                                   │
     │<────────────────────────────────────────────────│
     │                                                 │
     │ 6. process_image(raw_image)                    │
     ↓                                                 │
┌────────────────┐                                    │
│ ImageProcessor │                                    │
└────┬───────────┘                                    │
     │                                                 │
     │ 7. apply_filters(image)                        │
     ↓                                                 │
┌────────────────┐                                    │
│ FilterPipeline │                                    │
└────┬───────────┘                                    │
     │                                                 │
     │ 8. For each filter: apply()                    │
     │                                                 │
     │ 9. processed_image                             │
     ↓                                                 │
┌────────────────┐                                    │
│   Controller   │                                    │
└────┬───────────┘                                    │
     │                                                 │
     │ 10. update_display(image)                      │
     ↓                                                 │
┌────────────┐                                        │
│    View    │                                        │
│  (Display) │                                        │
└────┬───────┘                                        │
     │                                                 │
     │ 11. Show image to user                         │
     ↓                                                 │
┌──────────┐                                          │
│   User   │                                          │
└──────────┘                                          │
```

### Filter Configuration Flow

```
┌──────────┐
│   User   │
│          │
└────┬─────┘
     │
     │ Adjust "Denoise" slider
     ↓
┌────────────┐
│    View    │
│ (Controls) │
└────┬───────┘
     │
     │ on_denoise_changed(value=0.5)
     ↓
┌────────────────┐
│   Controller   │
└────┬───────────┘
     │
     │ Update filter config
     ↓
┌────────────────┐              ┌──────────────────┐
│ AppState       │              │ FilterPipeline   │
│                │──────────────>│                  │
│ .filters       │  Get filter  │ .filters[0]      │
│   [0] = Denoise│  instance    │   = DenoiseKNN   │
└────┬───────────┘              └────┬─────────────┘
     │                               │
     │ Set strength = 0.5            │
     │──────────────────────────────>│
     │                               │
     │                         DenoiseKNN
     │                         .enabled = True
     │                         .strength = 0.5
     │
     │ Trigger re-render
     ↓
┌────────────────┐
│ ImageProcessor │
│                │
│ Re-process with│
│ new settings   │
└────────────────┘
```

### Camera Initialization Flow

```
┌─────────────┐
│ Application │
│   Start     │
└──────┬──────┘
       │
       │ 1. Initialize
       ↓
┌──────────────┐
│     main.py  │
└──────┬───────┘
       │
       │ 2. init_camera()
       ↓
┌──────────────────┐
│  Camera class    │
└──────┬───────────┘
       │
       │ 3. asi.list_cameras()
       ↓
┌──────────────────┐
│  zwoasi_cupy     │
│  (Hardware SDK)  │
└──────┬───────────┘
       │
       │ 4. Returns: ["ZWO ASI178MC"]
       ↓
┌──────────────────┐
│  Camera class    │
└──────┬───────────┘
       │
       │ 5. get_camera_config("ZWO ASI178MC")
       ↓
┌──────────────────┐
│ camera_models.py │
│                  │
│ CAMERA_MODELS    │
│   ["ZWO ASI178MC"]│
└──────┬───────────┘
       │
       │ 6. Returns: CameraConfig(
       │      resolution_x=3096,
       │      resolution_y=2080,
       │      sensor_factor="4_3",
       │      ...
       │    )
       ↓
┌──────────────────┐
│  Camera class    │
│                  │
│  self.config =   │
│    CameraConfig  │
└──────┬───────────┘
       │
       │ 7. Camera ready
       ↓
┌──────────────────┐
│   Controller     │
│                  │
│ self.camera =    │
│   Camera(config) │
└──────────────────┘

OLD WAY: 1,500 lines of if-elif
NEW WAY: Dictionary lookup + dataclass
```

---

## Refactoring Migration Strategy

### Phase 1: Foundation (Week 1-2)

```
Current State                        Phase 1 Target
═══════════════                      ═══════════════

┌─────────────────┐                  ┌─────────────────┐
│ Monolithic File │                  │ Monolithic File │
│   11,301 lines  │                  │   11,301 lines  │
└─────────────────┘                  └─────────────────┘
                                              │
                                              │ (imports from)
                                              ↓
                                     ┌──────────────────┐
                                     │  core/config.py  │
                                     │                  │
                                     │ • CameraConfig   │
                                     │ • ProcessingState│
                                     │ • AppState       │
                                     └──────────────────┘
                                              │
                                              ↓
                                     ┌──────────────────┐
                                     │ core/           │
                                     │ camera_models.py │
                                     │                  │
                                     │ • CAMERA_MODELS  │
                                     │   dictionary     │
                                     └──────────────────┘
                                              │
                                              ↓
                                     ┌──────────────────┐
                                     │ utils/          │
                                     │ constants.py    │
                                     │                  │
                                     │ • All magic nums │
                                     └──────────────────┘

Action Items:
1. Create core/config.py with dataclasses
2. Create core/camera_models.py with registry
3. Create utils/constants.py with all constants
4. Update monolithic file to import these modules
5. Run tests to ensure functionality unchanged
```

### Phase 2: Extract Filters (Week 3-4)

```
Phase 1 State                        Phase 2 Target
═════════════                        ══════════════

┌─────────────────┐                  ┌─────────────────┐
│ Monolithic File │                  │ Reduced File    │
│   11,301 lines  │                  │   ~9,000 lines  │
└─────────────────┘                  └─────────────────┘
        │                                     │
        │                                     │ (imports from)
        ↓                                     ↓
┌─────────────────┐                  ┌─────────────────┐
│  Core modules   │                  │  Core modules   │
│  (from Phase 1) │                  │  (from Phase 1) │
└─────────────────┘                  └─────────────────┘
                                              │
                                              ↓
                                     ┌──────────────────┐
                                     │  filters/        │
                                     │                  │
                                     │ • base.py        │
                                     │ • pipeline.py    │
                                     │ • denoise.py     │
                                     │ • sharpen.py     │
                                     │ • contrast.py    │
                                     │ • color.py       │
                                     └──────────────────┘

Action Items:
1. Extract application_filtrage_color() → FilterPipeline
2. Extract application_filtrage_mono() → FilterPipeline
3. Create Filter base class
4. Create individual filter classes
5. Update main file to use FilterPipeline
6. Delete old filter functions (save ~2,000 lines)
```

### Phase 3: Extract Hardware & I/O (Week 5-6)

```
Phase 2 State                        Phase 3 Target
═════════════                        ══════════════

┌─────────────────┐                  ┌─────────────────┐
│ Reduced File    │                  │ Reduced File    │
│   ~9,000 lines  │                  │   ~6,000 lines  │
└─────────────────┘                  └─────────────────┘
        │                                     │
        ↓                                     ↓
┌─────────────────┐                  ┌─────────────────┐
│ Core + Filters  │                  │ Core + Filters  │
└─────────────────┘                  └─────────────────┘
                                              │
                                              ↓
                                     ┌──────────────────┐
                                     │  core/camera.py  │
                                     │                  │
                                     │ • Camera class   │
                                     │ • init_camera()  │
                                     │   (simplified)   │
                                     └──────────────────┘
                                              │
                                              ↓
                                     ┌──────────────────┐
                                     │  io/             │
                                     │                  │
                                     │ • image_io.py    │
                                     │ • video_io.py    │
                                     │ • serfile.py     │
                                     └──────────────────┘

Action Items:
1. Extract camera_acquisition() → Camera class
2. Extract init_camera() → Camera class (use registry)
3. Extract video_capture() → VideoCapture class
4. Extract pic_capture() → ImageCapture class
5. Move Serfile/ → io/serfile.py
6. Delete old functions (save ~3,000 lines)
```

### Phase 4: Extract AI & Utilities (Week 7-8)

```
Phase 3 State                        Phase 4 Target
═════════════                        ══════════════

┌─────────────────┐                  ┌─────────────────┐
│ Reduced File    │                  │ Reduced File    │
│   ~6,000 lines  │                  │   ~4,000 lines  │
└─────────────────┘                  └─────────────────┘
        │                                     │
        ↓                                     ↓
┌─────────────────┐                  ┌─────────────────┐
│ All modules     │                  │ All modules     │
│ (Phase 1-3)     │                  │ (Phase 1-3)     │
└─────────────────┘                  └─────────────────┘
                                              │
                                              ↓
                                     ┌──────────────────┐
                                     │  ai/             │
                                     │                  │
                                     │ • detector.py    │
                                     │ • crater_        │
                                     │   detector.py    │
                                     │ • satellite_     │
                                     │   detector.py    │
                                     └──────────────────┘
                                              │
                                              ↓
                                     ┌──────────────────┐
                                     │  utils/          │
                                     │                  │
                                     │ • threading.py   │
                                     │ • astronomy.py   │
                                     └──────────────────┘

Action Items:
1. Extract satellites_tracking_AI() → SatelliteDetector
2. Extract stars_detection() → StarDetector
3. Extract crater detection → CraterDetector
4. Extract Thread classes → utils/threading.py
5. Extract astronomy functions → utils/astronomy.py
6. Delete old functions (save ~2,000 lines)
```

### Phase 5: Create GUI Layer (Week 9-10)

```
Phase 4 State                        Phase 5 Target (FINAL)
═════════════                        ══════════════════════

┌─────────────────┐                  ┌─────────────────┐
│ Reduced File    │                  │    main.py      │
│   ~4,000 lines  │                  │   < 100 lines   │
│                 │                  │                 │
│ • GUI code      │                  │ Entry point only│
│ • Event handlers│                  └─────────────────┘
│ • State mgmt    │                           │
└─────────────────┘                           ↓
                                     ┌──────────────────┐
                                     │  gui/            │
                                     │                  │
                                     │ • main_window.py │
                                     │ • controls.py    │
                                     │ • display.py     │
                                     │ • dialogs.py     │
                                     └──────────────────┘
                                              │
                                              ↓
                                     ┌──────────────────┐
                                     │  core/           │
                                     │  controller.py   │
                                     │                  │
                                     │ JetsonSky        │
                                     │ Controller       │
                                     │ (MVC Controller) │
                                     └──────────────────┘
                                              │
                                              ↓
                              ┌───────────────┴───────────────┐
                              ↓                               ↓
                    All Business Logic              All Presentation
                    Modules (Phase 1-4)             Logic (GUI)

Action Items:
1. Create JetsonSkyController class (MVC pattern)
2. Extract all GUI code → gui/main_window.py
3. Extract controls → gui/controls.py
4. Extract display canvas → gui/display.py
5. Create main.py entry point
6. Wire everything together with controller
7. DELETE old monolithic file!
```

### Final Architecture Summary

```
BEFORE Refactoring:                  AFTER Refactoring:
═══════════════════                  ══════════════════

1 File: 11,301 lines                 25+ Files: ~5,500 total lines
                                     (49% reduction in code!)

300+ Global Variables                0 Global Variables
                                     All state encapsulated

143 Module-level Functions           ~10-20 functions per module
                                     Organized by responsibility

No separation of concerns            Clean layered architecture:
                                     • Presentation (GUI)
                                     • Business Logic (Core)
                                     • Domain Logic (Filters, AI)
                                     • Infrastructure (Hardware, I/O)

Impossible to test                   Fully testable:
                                     • Unit tests for filters
                                     • Integration tests for pipeline
                                     • Mock hardware for testing

4 duplicate versions                 1 version with configuration

No documentation                     Full documentation:
                                     • Module docstrings
                                     • Function type hints
                                     • API documentation

SOLID violations everywhere          SOLID principles followed:
                                     • Single Responsibility
                                     • Open/Closed
                                     • Liskov Substitution
                                     • Interface Segregation
                                     • Dependency Inversion
```

---

## Benefits Summary

### Maintainability

```
BEFORE: Change a filter → search 11K lines → hope you found all uses
AFTER:  Change a filter → edit filters/denoise.py → done

BEFORE: Add new camera → copy-paste 50 lines into if-elif chain
AFTER:  Add new camera → add entry to CAMERA_MODELS dictionary

BEFORE: Fix bug → might be anywhere in 11K lines
AFTER:  Fix bug → follow layer architecture to exact location
```

### Testability

```
BEFORE:
# Can't test because everything depends on global state
# Can't mock hardware because it's directly called
# No way to test individual filters

AFTER:
def test_denoise_filter():
    filter_obj = DenoiseKNNFilter(enabled=True, strength=0.5)
    input_img = cp.random.rand(100, 100)
    output = filter_obj.apply(input_img)
    assert output.shape == input_img.shape

def test_camera_init():
    config = get_camera_config("ZWO ASI178MC")
    assert config.resolution_x == 3096
    assert config.sensor_factor == "4_3"

def test_filter_pipeline():
    pipeline = FilterPipeline()
    pipeline.add_filter(DenoiseKNNFilter(enabled=True))
    pipeline.add_filter(SharpenFilter(enabled=True))

    input_img = create_test_image()
    output = pipeline.apply_all(input_img)
    assert output.shape == input_img.shape
```

### Scalability

```
BEFORE: Want to add new feature → fear touching anything
AFTER:  Want to add new feature → add new module, plug into pipeline

BEFORE: Want to support new camera → add 50 lines to if-elif hell
AFTER:  Want to support new camera → add 1 dictionary entry

BEFORE: Want new filter → insert into 1000-line function
AFTER:  Want new filter → create new Filter subclass, add to pipeline
```

### Team Collaboration

```
BEFORE:
• Only 1 person can work on code at a time
• Merge conflicts guaranteed
• Hard to review 1000-line function changes

AFTER:
• Team can work on different modules simultaneously
• Minimal merge conflicts (different files)
• Easy to review focused changes
```

---

## Metrics Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Lines per file (avg)** | 11,301 | ~200 | 98% ↓ |
| **Largest file** | 11,301 | 500 | 96% ↓ |
| **Global variables** | 300+ | 0 | 100% ↓ |
| **Cyclomatic complexity** | Very High | Low-Medium | 70% ↓ |
| **Test coverage** | 0% | 80%+ | ∞ ↑ |
| **Build time** | N/A | < 1 sec | N/A |
| **Duplicate code** | 44K lines (4 versions) | 5.5K lines (1 version) | 87% ↓ |
| **Number of modules** | 1 | 25+ | Better organization |
| **Dependencies per function** | 10-50 globals | 0-3 parameters | 90% ↓ |
| **Time to find code** | Minutes | Seconds | 95% ↓ |
| **Time to add feature** | Days | Hours | 75% ↓ |
| **Time to fix bug** | Hours | Minutes | 90% ↓ |

