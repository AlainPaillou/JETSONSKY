# JETSONSKY Architecture Diagrams (Mermaid Format)

This file contains architecture diagrams in Mermaid format that can be rendered in:
- GitHub/GitLab markdown
- VS Code (with Mermaid extension)
- Documentation sites
- Mermaid Live Editor: https://mermaid.live

---

## Current Monolithic Architecture

### File Structure - Current State

```mermaid
graph TD
    A[JetsonSky_Linux_Windows_V53_07RC.py<br/>11,301 lines] --> B[300+ Global Variables]
    A --> C[143 Module-Level Functions]
    A --> D[3 Thread Classes]
    A --> E[2,500+ lines GUI Code]

    C --> C1[GUI Functions ~50]
    C --> C2[Camera Functions ~30]
    C --> C3[Filter Functions ~40]
    C --> C4[AI Functions ~8]
    C --> C5[I/O Functions ~10]
    C --> C6[Mount Functions ~5]

    A --> F[Dependencies]
    F --> F1[zwoasi_cupy/<br/>1,052 lines]
    F --> F2[zwoefw/<br/>264 lines]
    F --> F3[synscan/<br/>929 lines]
    F --> F4[Serfile/<br/>339 lines]

    style A fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px
    style B fill:#ffd43b,stroke:#fab005
    style C fill:#ffd43b,stroke:#fab005
    style D fill:#ffd43b,stroke:#fab005
    style E fill:#ffd43b,stroke:#fab005
```

### Coupling Problem - Current State

```mermaid
graph LR
    GS[Global State<br/>300+ variables] -.->|read/write| GUI[GUI Functions]
    GS -.->|read/write| CAM[Camera Functions]
    GS -.->|read/write| FILT[Filter Functions]
    GS -.->|read/write| AI[AI Functions]
    GS -.->|read/write| IO[I/O Functions]
    GS -.->|read/write| MNT[Mount Functions]

    GUI -.->|modify| GS
    CAM -.->|modify| GS
    FILT -.->|modify| GS
    AI -.->|modify| GS
    IO -.->|modify| GS
    MNT -.->|modify| GS

    style GS fill:#ff6b6b,stroke:#c92a2a,stroke-width:4px
    style GUI fill:#ffd43b
    style CAM fill:#ffd43b
    style FILT fill:#ffd43b
    style AI fill:#ffd43b
    style IO fill:#ffd43b
    style MNT fill:#ffd43b
```

---

## Proposed Modular Architecture

### Directory Structure - Proposed

```mermaid
graph TD
    ROOT[JetsonSky/]

    ROOT --> MAIN[main.py<br/>Entry Point<br/>< 100 lines]
    ROOT --> CONFIG[config.yaml<br/>Configuration]

    ROOT --> GUI[gui/]
    ROOT --> CORE[core/]
    ROOT --> FILTERS[filters/]
    ROOT --> AI[ai/]
    ROOT --> HW[hardware/]
    ROOT --> IO[io/]
    ROOT --> UTILS[utils/]

    GUI --> GUI1[main_window.py<br/>~200 lines]
    GUI --> GUI2[controls.py<br/>~150 lines]
    GUI --> GUI3[display.py<br/>~100 lines]
    GUI --> GUI4[dialogs.py<br/>~100 lines]

    CORE --> CORE1[controller.py<br/>~300 lines]
    CORE --> CORE2[camera.py<br/>~200 lines]
    CORE --> CORE3[camera_models.py<br/>~400 lines]
    CORE --> CORE4[image_processor.py<br/>~250 lines]
    CORE --> CORE5[config.py<br/>~200 lines]

    FILTERS --> FILT1[base.py<br/>~50 lines]
    FILTERS --> FILT2[pipeline.py<br/>~100 lines]
    FILTERS --> FILT3[denoise.py<br/>~300 lines]
    FILTERS --> FILT4[sharpen.py<br/>~150 lines]
    FILTERS --> FILT5[contrast.py<br/>~200 lines]
    FILTERS --> FILT6[color.py<br/>~150 lines]

    AI --> AI1[detector.py<br/>~100 lines]
    AI --> AI2[crater_detector.py<br/>~150 lines]
    AI --> AI3[satellite_detector.py<br/>~200 lines]

    HW --> HW1[zwoasi_cupy/<br/>existing]
    HW --> HW2[zwoefw/<br/>existing]
    HW --> HW3[synscan/<br/>existing]

    IO --> IO1[image_io.py<br/>~200 lines]
    IO --> IO2[video_io.py<br/>~250 lines]
    IO --> IO3[serfile.py<br/>moved]

    UTILS --> UTILS1[constants.py<br/>~100 lines]
    UTILS --> UTILS2[threading.py<br/>~150 lines]
    UTILS --> UTILS3[astronomy.py<br/>~150 lines]

    style MAIN fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style CONFIG fill:#51cf66,stroke:#2f9e44
    style GUI fill:#74c0fc,stroke:#1971c2
    style CORE fill:#ffd43b,stroke:#f59f00
    style FILTERS fill:#ffc9c9,stroke:#f03e3e
    style AI fill:#d0bfff,stroke:#7950f2
    style HW fill:#a9e34b,stroke:#74b816
    style IO fill:#ffa8a8,stroke:#e03131
    style UTILS fill:#e7f5ff,stroke:#339af0
```

### Layered Architecture - Proposed

```mermaid
graph TB
    subgraph Presentation["Presentation Layer"]
        MW[MainWindow]
        CTRL_UI[Controls]
        DISP[Display]
    end

    subgraph Business["Business Logic Layer"]
        CONTROLLER[JetsonSkyController<br/>MVC Controller]
        CAMERA[Camera]
        IMGPROC[ImageProcessor]
        STATE[AppState]
    end

    subgraph Domain["Domain Logic Layer"]
        FILT_PIPE[FilterPipeline]
        DETECTORS[AI Detectors]
        FILEIO[File I/O]
    end

    subgraph Infrastructure["Infrastructure Layer"]
        HWCAM[ZWO Camera SDK]
        HWEFW[Filter Wheel]
        HWMNT[Mount Control]
    end

    MW -->|events| CONTROLLER
    CTRL_UI -->|events| CONTROLLER
    CONTROLLER -->|updates| MW
    CONTROLLER -->|updates| DISP

    CONTROLLER --> CAMERA
    CONTROLLER --> IMGPROC
    CONTROLLER --> STATE

    IMGPROC --> FILT_PIPE
    IMGPROC --> DETECTORS
    CAMERA --> FILEIO

    CAMERA --> HWCAM
    CAMERA --> HWEFW
    CAMERA --> HWMNT

    style Presentation fill:#e7f5ff,stroke:#1971c2,stroke-width:2px
    style Business fill:#fff4e6,stroke:#f59f00,stroke-width:2px
    style Domain fill:#ffe9ec,stroke:#f03e3e,stroke-width:2px
    style Infrastructure fill:#ebfbee,stroke:#2f9e44,stroke-width:2px
```

### MVC Pattern

```mermaid
graph LR
    subgraph View
        GUI[GUI Components<br/>MainWindow<br/>Controls<br/>Display]
    end

    subgraph Controller
        CTRL[JetsonSkyController<br/>Event Handlers<br/>State Management]
    end

    subgraph Model
        STATE[AppState<br/>CameraConfig<br/>ProcessingState]
        BUSINESS[Business Logic<br/>Camera<br/>ImageProcessor<br/>FilterPipeline]
    end

    GUI -->|User Events| CTRL
    CTRL -->|Update View| GUI
    CTRL -->|Update Model| STATE
    CTRL -->|Commands| BUSINESS
    STATE -->|State Changes| CTRL
    BUSINESS -->|Results| CTRL

    style View fill:#e7f5ff,stroke:#1971c2,stroke-width:2px
    style Controller fill:#fff4e6,stroke:#f59f00,stroke-width:2px
    style Model fill:#ebfbee,stroke:#2f9e44,stroke-width:2px
```

---

## Component Interaction Diagrams

### Image Acquisition Sequence

```mermaid
sequenceDiagram
    participant User
    participant View
    participant Controller
    participant AcqThread
    participant Camera
    participant ImgProc
    participant FilterPipe
    participant Display

    User->>View: Click "Start Acquisition"
    View->>Controller: on_start_acquisition()
    Controller->>AcqThread: start()

    loop Continuous Acquisition
        AcqThread->>Camera: capture()
        Camera-->>AcqThread: raw_image
        AcqThread->>ImgProc: process_image(raw_image)
        ImgProc->>FilterPipe: apply_all(image)

        loop For Each Enabled Filter
            FilterPipe->>FilterPipe: filter.apply(image)
        end

        FilterPipe-->>ImgProc: processed_image
        ImgProc-->>AcqThread: result
        AcqThread->>Controller: update(result)
        Controller->>Display: update_display(result)
        Display->>User: Show image
    end
```

### Filter Configuration Sequence

```mermaid
sequenceDiagram
    participant User
    participant ControlsView
    participant Controller
    participant AppState
    participant FilterPipeline
    participant DenoiseFilter

    User->>ControlsView: Adjust "Denoise" slider
    ControlsView->>Controller: on_denoise_changed(value=0.5)
    Controller->>AppState: Get current filter config
    AppState-->>Controller: filter_pipeline
    Controller->>FilterPipeline: Get denoise filter
    FilterPipeline-->>Controller: DenoiseFilter instance
    Controller->>DenoiseFilter: Set strength = 0.5
    Controller->>Controller: Trigger re-render
    Note over Controller: Image re-processed<br/>with new settings
```

### Camera Initialization Sequence

```mermaid
sequenceDiagram
    participant Main
    participant Camera
    participant SDK as ZWO SDK
    participant Registry as camera_models.py
    participant Controller

    Main->>Camera: init_camera()
    Camera->>SDK: asi.list_cameras()
    SDK-->>Camera: ["ZWO ASI178MC"]
    Camera->>Registry: get_camera_config("ZWO ASI178MC")
    Registry-->>Camera: CameraConfig(res_x=3096, res_y=2080, ...)
    Camera->>Camera: self.config = CameraConfig
    Camera-->>Main: Camera instance
    Main->>Controller: JetsonSkyController(camera)
    Note over Controller: Application ready
```

---

## Data Flow Diagrams

### Image Processing Pipeline

```mermaid
flowchart LR
    CAM[Camera<br/>Hardware] -->|Raw Bayer<br/>16-bit| ACQ[Acquisition<br/>Thread]
    ACQ -->|Raw Image| DEBAY[Debayer<br/>RGGB→RGB]
    DEBAY -->|RGB Image| PIPE[Filter<br/>Pipeline]

    subgraph FilterPipeline
        F1[Denoise KNN]
        F2[Sharpen]
        F3[Contrast CLAHE]
        F4[Saturation]
        F5[Color Balance]
        F1 --> F2 --> F3 --> F4 --> F5
    end

    PIPE -->|Filtered<br/>Image| CONV[Convert to<br/>Display Format]
    CONV -->|PIL Image| GUI[GUI Display]
    GUI --> USER[User]

    style CAM fill:#a9e34b,stroke:#74b816
    style ACQ fill:#74c0fc,stroke:#1971c2
    style DEBAY fill:#ffc9c9,stroke:#f03e3e
    style PIPE fill:#ffd43b,stroke:#f59f00
    style CONV fill:#d0bfff,stroke:#7950f2
    style GUI fill:#e7f5ff,stroke:#1971c2
```

### State Management Flow

```mermaid
flowchart TD
    USER[User Action<br/>Button Click / Slider]

    USER --> VIEW[View<br/>GUI Component]
    VIEW -->|Event| CTRL[Controller<br/>Event Handler]

    CTRL -->|Read| STATE[AppState<br/>Current State]
    CTRL -->|Update| STATE
    CTRL -->|Command| BUSINESS[Business Logic<br/>Camera, Processor, etc.]

    STATE -->|Notify| CTRL
    BUSINESS -->|Result| CTRL

    CTRL -->|Update UI| VIEW
    VIEW --> USER

    style USER fill:#e7f5ff,stroke:#1971c2
    style VIEW fill:#e7f5ff,stroke:#1971c2
    style CTRL fill:#fff4e6,stroke:#f59f00
    style STATE fill:#d3f9d8,stroke:#2f9e44
    style BUSINESS fill:#ffe9ec,stroke:#f03e3e
```

### Filter Pipeline Flow

```mermaid
flowchart TD
    START[Input Image<br/>CuPy Array]

    START --> CHECK1{Denoise<br/>Enabled?}
    CHECK1 -->|Yes| DENOISE[Apply Denoise<br/>KNN Filter]
    CHECK1 -->|No| CHECK2
    DENOISE --> CHECK2

    CHECK2{Sharpen<br/>Enabled?}
    CHECK2 -->|Yes| SHARPEN[Apply Sharpen<br/>Soft Filter]
    CHECK2 -->|No| CHECK3
    SHARPEN --> CHECK3

    CHECK3{Contrast<br/>Enabled?}
    CHECK3 -->|Yes| CONTRAST[Apply Contrast<br/>CLAHE Filter]
    CHECK3 -->|No| CHECK4
    CONTRAST --> CHECK4

    CHECK4{Saturation<br/>Enabled?}
    CHECK4 -->|Yes| SAT[Apply Saturation<br/>Enhancement]
    CHECK4 -->|No| CHECK5
    SAT --> CHECK5

    CHECK5{Color Balance<br/>Enabled?}
    CHECK5 -->|Yes| COLOR[Apply Color<br/>Balance Filter]
    CHECK5 -->|No| END
    COLOR --> END

    END[Output Image<br/>CuPy Array]

    style START fill:#51cf66,stroke:#2f9e44
    style END fill:#51cf66,stroke:#2f9e44
    style DENOISE fill:#ffc9c9,stroke:#f03e3e
    style SHARPEN fill:#ffc9c9,stroke:#f03e3e
    style CONTRAST fill:#ffc9c9,stroke:#f03e3e
    style SAT fill:#ffc9c9,stroke:#f03e3e
    style COLOR fill:#ffc9c9,stroke:#f03e3e
```

---

## Refactoring Migration Strategy

### 5-Phase Migration Plan

```mermaid
gantt
    title Refactoring Timeline (10 Weeks)
    dateFormat YYYY-MM-DD

    section Phase 1
    Create Config Classes       :p1a, 2025-01-01, 7d
    Create Camera Registry      :p1b, 2025-01-04, 7d
    Extract Constants           :p1c, 2025-01-08, 3d
    Testing & Validation        :p1d, 2025-01-11, 4d

    section Phase 2
    Extract Filter Base         :p2a, 2025-01-15, 5d
    Create Filter Classes       :p2b, 2025-01-18, 7d
    Create Pipeline             :p2c, 2025-01-22, 3d
    Testing & Validation        :p2d, 2025-01-25, 3d

    section Phase 3
    Extract Camera Class        :p3a, 2025-01-29, 7d
    Extract I/O Functions       :p3b, 2025-02-01, 7d
    Testing & Validation        :p3c, 2025-02-08, 4d

    section Phase 4
    Extract AI Detectors        :p4a, 2025-02-12, 7d
    Extract Utilities           :p4b, 2025-02-15, 5d
    Testing & Validation        :p4c, 2025-02-20, 2d

    section Phase 5
    Create GUI Layer            :p5a, 2025-02-22, 7d
    Create Controller           :p5b, 2025-02-25, 7d
    Integration & Testing       :p5c, 2025-03-04, 7d
    Documentation               :p5d, 2025-03-11, 3d
```

### Phase-by-Phase Code Reduction

```mermaid
graph LR
    START[Monolithic File<br/>11,301 lines]
    P1[Phase 1<br/>11,301 lines<br/>+Config Modules]
    P2[Phase 2<br/>~9,000 lines<br/>+Filter Modules]
    P3[Phase 3<br/>~6,000 lines<br/>+Camera & I/O]
    P4[Phase 4<br/>~4,000 lines<br/>+AI & Utils]
    P5[Phase 5<br/>main.py<br/>< 100 lines<br/>+GUI Modules]

    START -->|Extract Config| P1
    P1 -->|Extract Filters<br/>-2,000 lines| P2
    P2 -->|Extract Camera/IO<br/>-3,000 lines| P3
    P3 -->|Extract AI/Utils<br/>-2,000 lines| P4
    P4 -->|Extract GUI<br/>-3,900 lines| P5

    style START fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px
    style P1 fill:#ffd43b,stroke:#f59f00
    style P2 fill:#ffec99,stroke:#f59f00
    style P3 fill:#d3f9d8,stroke:#2f9e44
    style P4 fill:#b2f2bb,stroke:#2f9e44
    style P5 fill:#51cf66,stroke:#2b8a3e,stroke-width:3px
```

### Dependency Evolution

```mermaid
graph TB
    subgraph "Phase 1: Foundation"
        M1[Monolithic File]
        C1[core/config.py]
        CM1[core/camera_models.py]
        U1[utils/constants.py]
        M1 --> C1
        M1 --> CM1
        M1 --> U1
    end

    subgraph "Phase 2: Filters"
        M2[Reduced File<br/>~9K lines]
        F2[filters/*]
        M2 --> F2
        M2 --> C2[Core Modules]
    end

    subgraph "Phase 3: Hardware & I/O"
        M3[Reduced File<br/>~6K lines]
        CAM3[core/camera.py]
        IO3[io/*]
        M3 --> CAM3
        M3 --> IO3
        M3 --> F3[Filters]
        M3 --> C3[Core]
    end

    subgraph "Phase 4: AI & Utils"
        M4[Reduced File<br/>~4K lines]
        AI4[ai/*]
        UT4[utils/threading.py<br/>utils/astronomy.py]
        M4 --> AI4
        M4 --> UT4
        M4 --> ALL4[All Previous Modules]
    end

    subgraph "Phase 5: GUI Layer"
        MAIN5[main.py<br/>< 100 lines]
        GUI5[gui/*]
        CTRL5[core/controller.py]
        MAIN5 --> GUI5
        MAIN5 --> CTRL5
        CTRL5 --> ALL5[All Business Logic]
        GUI5 --> CTRL5
    end

    style M1 fill:#ff6b6b
    style M2 fill:#ffd43b
    style M3 fill:#ffec99
    style M4 fill:#d3f9d8
    style MAIN5 fill:#51cf66,stroke:#2b8a3e,stroke-width:3px
```

---

## Class Diagrams

### Camera Model Registry

```mermaid
classDiagram
    class CameraConfig {
        +str model
        +int resolution_x
        +int resolution_y
        +int display_x
        +int display_y
        +str sensor_factor
        +int sensor_bits
        +str bayer_pattern
        +List~Tuple~ supported_resolutions_bin1
        +List~Tuple~ supported_resolutions_bin2
        +int max_gain
        +int usb_bandwidth
    }

    class CameraModels {
        <<dictionary>>
        +Dict~str,CameraConfig~ CAMERA_MODELS
        +get_camera_config(model_name: str) CameraConfig
    }

    class Camera {
        -CameraConfig config
        -asi_camera hardware
        +__init__(model_name: str)
        +capture() ndarray
        +set_exposure(ms: int)
        +set_gain(value: int)
        +get_resolution() Tuple
    }

    CameraModels --> CameraConfig : contains
    Camera --> CameraConfig : uses
    Camera --> CameraModels : queries
```

### Filter Pipeline

```mermaid
classDiagram
    class Filter {
        <<abstract>>
        +bool enabled
        +__init__(enabled: bool)
        +apply(image: ndarray)* ndarray
        +get_name()* str
    }

    class DenoiseKNNFilter {
        +float strength
        +apply(image: ndarray) ndarray
        +get_name() str
    }

    class SharpenFilter {
        +float amount
        +float sigma
        +apply(image: ndarray) ndarray
        +get_name() str
    }

    class ContrastCLAHEFilter {
        +float clip_limit
        +int grid_size
        +apply(image: ndarray) ndarray
        +get_name() str
    }

    class FilterPipeline {
        -List~Filter~ filters
        +add_filter(filter: Filter)
        +remove_filter(index: int)
        +apply_all(image: ndarray) ndarray
        +get_active_filters() List~str~
    }

    Filter <|-- DenoiseKNNFilter
    Filter <|-- SharpenFilter
    Filter <|-- ContrastCLAHEFilter
    FilterPipeline o-- Filter : contains
```

### MVC Architecture

```mermaid
classDiagram
    class MainWindow {
        -JetsonSkyController controller
        -Tk root
        -Button flip_v_button
        -Scale gain_slider
        +__init__(controller: Controller)
        +build_ui()
        +update_flip_v_button(enabled: bool)
        +update_gain_slider(value: int)
    }

    class JetsonSkyController {
        -AppState state
        -MainWindow view
        -Camera camera
        -ImageProcessor processor
        +__init__(state: AppState, view: MainWindow)
        +toggle_flip_vertical()
        +set_gain(value: int)
        +start_acquisition()
        +stop_acquisition()
    }

    class AppState {
        +CameraConfig camera_config
        +ProcessingState processing_state
        +bool acquisition_running
        +bool mount_tracking
    }

    class ProcessingState {
        +int exposition
        +int gain
        +float denoise
        +bool flip_vertical
        +bool flip_horizontal
        +float saturation
    }

    class Camera {
        -CameraConfig config
        +capture() ndarray
        +set_exposure(ms: int)
        +set_gain(value: int)
    }

    class ImageProcessor {
        -FilterPipeline pipeline
        +process(image: ndarray) ndarray
        +set_filter_enabled(name: str, enabled: bool)
    }

    MainWindow --> JetsonSkyController : uses
    JetsonSkyController --> MainWindow : updates
    JetsonSkyController --> AppState : manages
    JetsonSkyController --> Camera : controls
    JetsonSkyController --> ImageProcessor : uses
    AppState --> ProcessingState : contains
    ImageProcessor --> FilterPipeline : uses
```

---

## Metrics Visualization

### Code Distribution - Before vs After

```mermaid
pie title Current Code Distribution (Monolithic)
    "GUI Code" : 2500
    "Image Processing" : 3000
    "Camera Control" : 2000
    "Video/Image I/O" : 1500
    "AI Integration" : 800
    "Mount Control" : 800
    "Utilities" : 701
```

```mermaid
pie title Proposed Code Distribution (Modular)
    "gui/" : 550
    "core/" : 1350
    "filters/" : 950
    "ai/" : 450
    "hardware/" : 0
    "io/" : 650
    "utils/" : 400
    "main.py" : 100
    "Code Reduction" : 5851
```

### Complexity Reduction

```mermaid
graph LR
    subgraph Before
        B1[Lines per File: 11,301]
        B2[Global Variables: 300+]
        B3[Function Dependencies: 10-50]
        B4[Cyclomatic Complexity: Very High]
    end

    subgraph After
        A1[Lines per File: ~200]
        A2[Global Variables: 0]
        A3[Function Dependencies: 0-3]
        A4[Cyclomatic Complexity: Low]
    end

    B1 -.->|96% reduction| A1
    B2 -.->|100% reduction| A2
    B3 -.->|90% reduction| A3
    B4 -.->|70% reduction| A4

    style B1 fill:#ff6b6b
    style B2 fill:#ff6b6b
    style B3 fill:#ff6b6b
    style B4 fill:#ff6b6b
    style A1 fill:#51cf66
    style A2 fill:#51cf66
    style A3 fill:#51cf66
    style A4 fill:#51cf66
```

---

## Testing Strategy

### Test Coverage Pyramid

```mermaid
graph TD
    subgraph "Test Pyramid"
        E2E[End-to-End Tests<br/>~10%<br/>Full application workflow]
        INT[Integration Tests<br/>~30%<br/>Module interactions]
        UNIT[Unit Tests<br/>~60%<br/>Individual functions/classes]
    end

    E2E --> INT
    INT --> UNIT

    style E2E fill:#e7f5ff,stroke:#1971c2
    style INT fill:#fff4e6,stroke:#f59f00
    style UNIT fill:#ebfbee,stroke:#2f9e44
```

### Test Organization

```mermaid
graph TB
    TESTS[tests/]

    TESTS --> UNIT[unit/]
    TESTS --> INT[integration/]
    TESTS --> E2E[e2e/]

    UNIT --> U1[test_filters.py<br/>Test each filter independently]
    UNIT --> U2[test_camera_models.py<br/>Test camera registry]
    UNIT --> U3[test_config.py<br/>Test data classes]

    INT --> I1[test_pipeline.py<br/>Test filter pipeline]
    INT --> I2[test_camera.py<br/>Test camera with SDK]
    INT --> I3[test_image_processor.py<br/>Test full processing chain]

    E2E --> E1[test_acquisition.py<br/>Test full acquisition flow]
    E2E --> E2[test_video_capture.py<br/>Test video recording]

    style TESTS fill:#e7f5ff,stroke:#1971c2,stroke-width:2px
    style UNIT fill:#ebfbee,stroke:#2f9e44
    style INT fill:#fff4e6,stroke:#f59f00
    style E2E fill:#ffe9ec,stroke:#f03e3e
```

---

## Deployment

### Package Structure

```mermaid
graph TB
    PKG[JetsonSky Package]

    PKG --> SRC[src/<br/>Source code]
    PKG --> TESTS[tests/<br/>Test suite]
    PKG --> DOCS[docs/<br/>Documentation]
    PKG --> CFG[Configuration files]

    SRC --> MODULES[All modules<br/>gui/, core/, filters/, etc.]

    TESTS --> UNIT_T[Unit tests]
    TESTS --> INT_T[Integration tests]

    DOCS --> API[API documentation]
    DOCS --> ARCH[Architecture docs]
    DOCS --> USER[User guide]

    CFG --> SETUP[setup.py]
    CFG --> REQ[requirements.txt]
    CFG --> CONFIG[config.yaml]
    CFG --> README[README.md]

    style PKG fill:#51cf66,stroke:#2b8a3e,stroke-width:3px
```

---

## Summary

These diagrams visualize the transformation from:
- **Monolithic nightmare**: 11,301 lines, 300+ globals, impossible to maintain
- **Clean architecture**: Modular, testable, maintainable, professional

**Key Improvements:**
- 96% reduction in largest file size
- 100% elimination of global variables
- 90% reduction in function dependencies
- Full test coverage capability
- Clear separation of concerns
- Industry-standard architecture patterns

You can view these diagrams:
1. In GitHub/GitLab (automatic Mermaid rendering)
2. In VS Code with Mermaid extension
3. At https://mermaid.live (paste the code blocks)
4. In your documentation site
