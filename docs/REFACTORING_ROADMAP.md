# JETSONSKY Refactoring Roadmap

## Executive Summary

**Current State**: Monolithic 11,301-line file with 300+ global variables
**Target State**: Clean modular architecture with 25+ focused modules
**Timeline**: 10 weeks (5 phases)
**Code Reduction**: 49% reduction in total lines
**Maintainability**: Impossible → Professional grade

---

## Quick Reference

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Largest file** | 11,301 lines | < 500 lines | **96%** ↓ |
| **Total lines** | ~11,300 | ~5,500 | **51%** ↓ |
| **Global variables** | 300+ | 0 | **100%** ↓ |
| **Modules** | 1 | 25+ | **Better organized** |
| **Test coverage** | 0% | 80%+ | **∞** ↑ |
| **Time to find code** | Minutes | Seconds | **95%** ↓ |
| **Time to add feature** | Days | Hours | **75%** ↓ |
| **Duplicate versions** | 4 (44K lines) | 1 (5.5K lines) | **87%** ↓ |

---

## Phase-by-Phase Breakdown

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Extract configuration and camera registry without breaking existing code

**Tasks**:

1. **Create `core/config.py`** (~200 lines)
   - [ ] Define `CameraConfig` dataclass
   - [ ] Define `ProcessingState` dataclass
   - [ ] Define `AppState` dataclass
   - [ ] Add type hints and docstrings

2. **Create `core/camera_models.py`** (~400 lines)
   - [ ] Define `CAMERA_MODELS` dictionary
   - [ ] Add all 20+ camera configurations
   - [ ] Create `get_camera_config()` function
   - [ ] Add validation

3. **Create `utils/constants.py`** (~100 lines)
   - [ ] Extract all magic numbers
   - [ ] Extract color constants
   - [ ] Extract default values
   - [ ] Organize by category

4. **Update monolithic file**
   - [ ] Import new modules
   - [ ] Replace hardcoded values with constants
   - [ ] Test that everything still works

**Deliverables**:
- `core/config.py` ✓
- `core/camera_models.py` ✓
- `utils/constants.py` ✓
- Updated imports in main file
- All tests pass

**Success Criteria**:
- Application runs without errors
- No functionality broken
- Code still 11,301 lines (adding modules, not removing yet)

---

### Phase 2: Extract Filters (Weeks 3-4)

**Goal**: Replace monolithic filter functions with modular filter pipeline

**Tasks**:

1. **Create filter base classes** (~150 lines)
   - [ ] Create `filters/base.py` with `Filter` abstract class
   - [ ] Create `filters/pipeline.py` with `FilterPipeline` class
   - [ ] Add filter registration mechanism
   - [ ] Add enable/disable functionality

2. **Extract denoise filters** (~300 lines)
   - [ ] Create `filters/denoise.py`
   - [ ] Implement `DenoiseKNNFilter`
   - [ ] Implement `DenoiseNLM2Filter`
   - [ ] Implement `DenoisePaillouFilter`
   - [ ] Implement `DenoisePaillou2Filter`
   - [ ] Implement `AdaptiveAbsorberFilter`
   - [ ] Implement `ThreeFrameNoiseRemovalFilter`

3. **Extract sharpen filters** (~150 lines)
   - [ ] Create `filters/sharpen.py`
   - [ ] Implement `SharpenSoft1Filter`
   - [ ] Implement `SharpenSoft2Filter`

4. **Extract contrast filters** (~200 lines)
   - [ ] Create `filters/contrast.py`
   - [ ] Implement `ContrastCLAHEFilter`
   - [ ] Implement `ContrastLowLightFilter`
   - [ ] Implement `HistogramStretchFilter`
   - [ ] Implement `HistogramPhiThetaFilter`
   - [ ] Implement `HistogramEqualizeFilter`

5. **Extract color filters** (~150 lines)
   - [ ] Create `filters/color.py`
   - [ ] Implement `ColorBalanceFilter`
   - [ ] Implement `SaturationFilter`
   - [ ] Implement `RGBBalanceFilter`

6. **Replace filter application functions**
   - [ ] Replace `application_filtrage_color()` with `FilterPipeline`
   - [ ] Replace `application_filtrage_mono()` with `FilterPipeline`
   - [ ] Update all filter toggle functions
   - [ ] Delete old filter code (~2,000 lines)

**Deliverables**:
- `filters/base.py` ✓
- `filters/pipeline.py` ✓
- `filters/denoise.py` ✓
- `filters/sharpen.py` ✓
- `filters/contrast.py` ✓
- `filters/color.py` ✓
- Main file reduced to ~9,000 lines
- Unit tests for each filter

**Success Criteria**:
- All filters work identically to before
- Main file reduced by ~2,000 lines
- Each filter has unit tests
- Can add/remove filters dynamically

---

### Phase 3: Extract Hardware & I/O (Weeks 5-6)

**Goal**: Abstract camera control and file I/O into clean interfaces

**Tasks**:

1. **Create Camera abstraction** (~200 lines)
   - [ ] Create `core/camera.py`
   - [ ] Implement `Camera` class
   - [ ] Move `init_camera()` logic (now 10 lines instead of 1,500!)
   - [ ] Move `camera_acquisition()` logic
   - [ ] Add camera control methods
   - [ ] Use camera registry for configuration

2. **Extract image I/O** (~200 lines)
   - [ ] Create `io/image_io.py`
   - [ ] Implement `load_image()` function
   - [ ] Implement `save_image()` function
   - [ ] Implement `ImageCapture` class
   - [ ] Move `pic_capture()` logic
   - [ ] Add format support (TIFF, JPEG, PNG)

3. **Extract video I/O** (~250 lines)
   - [ ] Create `io/video_io.py`
   - [ ] Implement `load_video()` function
   - [ ] Implement `save_video()` function
   - [ ] Implement `VideoCapture` class
   - [ ] Move `video_capture()` logic
   - [ ] Add codec support (MP4, AVI, MOV)

4. **Move SER file handler** (~339 lines)
   - [ ] Move `Serfile/` to `io/serfile.py`
   - [ ] Update imports
   - [ ] Clean up interface

5. **Update main file**
   - [ ] Replace camera init with `Camera` class
   - [ ] Replace I/O functions with new modules
   - [ ] Delete old code (~3,000 lines)

**Deliverables**:
- `core/camera.py` ✓
- `io/image_io.py` ✓
- `io/video_io.py` ✓
- `io/serfile.py` ✓
- Main file reduced to ~6,000 lines
- Integration tests for camera and I/O

**Success Criteria**:
- Camera initialization is 10 lines instead of 1,500
- Can mock camera for testing
- All file formats supported
- Main file reduced by ~3,000 lines

---

### Phase 4: Extract AI & Utilities (Weeks 7-8)

**Goal**: Separate AI detection and utility functions

**Tasks**:

1. **Create AI detector base** (~100 lines)
   - [ ] Create `ai/detector.py`
   - [ ] Implement `BaseDetector` abstract class
   - [ ] Add common detection utilities

2. **Extract crater detection** (~150 lines)
   - [ ] Create `ai/crater_detector.py`
   - [ ] Implement `CraterDetector` class
   - [ ] Move YOLOv8 model loading
   - [ ] Move crater detection logic
   - [ ] Add bounding box drawing

3. **Extract satellite detection** (~200 lines)
   - [ ] Create `ai/satellite_detector.py`
   - [ ] Implement `SatelliteDetector` class
   - [ ] Move `satellites_tracking_AI()` logic
   - [ ] Move `satellites_tracking()` logic
   - [ ] Move `remove_satellites()` logic
   - [ ] Add tracking functionality

4. **Extract threading utilities** (~150 lines)
   - [ ] Create `utils/threading.py`
   - [ ] Move `keyboard_management` thread
   - [ ] Move `acquisition_mount` thread
   - [ ] Move `acquisition` thread
   - [ ] Clean up thread interfaces

5. **Extract astronomy utilities** (~150 lines)
   - [ ] Create `utils/astronomy.py`
   - [ ] Move `calc_jour_julien()`
   - [ ] Move `calc_heure_siderale()`
   - [ ] Move `calcul_AZ_HT_cible()`
   - [ ] Move `calcul_ASD_DEC_cible()`
   - [ ] Move `Mount_calibration()`

6. **Update main file**
   - [ ] Replace AI functions with detector classes
   - [ ] Replace threads with utility classes
   - [ ] Replace astronomy functions
   - [ ] Delete old code (~2,000 lines)

**Deliverables**:
- `ai/detector.py` ✓
- `ai/crater_detector.py` ✓
- `ai/satellite_detector.py` ✓
- `utils/threading.py` ✓
- `utils/astronomy.py` ✓
- Main file reduced to ~4,000 lines
- Unit tests for AI detectors

**Success Criteria**:
- AI detection is modular and testable
- Can add new detectors easily
- Threads are clean and documented
- Main file reduced by ~2,000 lines

---

### Phase 5: Create GUI Layer (Weeks 9-10)

**Goal**: Complete separation of GUI from business logic using MVC pattern

**Tasks**:

1. **Create MVC Controller** (~300 lines)
   - [ ] Create `core/controller.py`
   - [ ] Implement `JetsonSkyController` class
   - [ ] Add event handlers for all GUI actions
   - [ ] Add state management
   - [ ] Connect to all business logic modules

2. **Extract main window** (~200 lines)
   - [ ] Create `gui/main_window.py`
   - [ ] Implement `MainWindow` class
   - [ ] Move Tkinter window creation
   - [ ] Move main layout
   - [ ] Connect to controller

3. **Extract control panels** (~150 lines)
   - [ ] Create `gui/controls.py`
   - [ ] Implement `ControlPanel` class
   - [ ] Move all buttons, sliders, checkboxes
   - [ ] Wire up to controller events

4. **Extract display canvas** (~100 lines)
   - [ ] Create `gui/display.py`
   - [ ] Implement `DisplayCanvas` class
   - [ ] Move image display logic
   - [ ] Move zoom/pan functionality
   - [ ] Add overlay drawing

5. **Extract dialogs** (~100 lines)
   - [ ] Create `gui/dialogs.py`
   - [ ] Move file dialogs
   - [ ] Move message dialogs
   - [ ] Add configuration dialogs

6. **Create main entry point** (~100 lines)
   - [ ] Create `main.py`
   - [ ] Initialize all components
   - [ ] Wire up MVC pattern
   - [ ] Start application loop

7. **Create configuration file** (~50 lines)
   - [ ] Create `config.yaml`
   - [ ] Add default settings
   - [ ] Add camera preferences
   - [ ] Add filter presets

8. **Delete monolithic file**
   - [ ] Verify all functionality moved
   - [ ] Delete `JetsonSky_Linux_Windows_V53_07RC.py`
   - [ ] Delete `JetsonSky_Linux_Windows_V53_06RC.py`
   - [ ] Delete `JetsonSky_Linux_Windows_V53_04RC.py`
   - [ ] Delete `JetsonSky_Linux_Windows_V51_05RC.py`

**Deliverables**:
- `core/controller.py` ✓
- `gui/main_window.py` ✓
- `gui/controls.py` ✓
- `gui/display.py` ✓
- `gui/dialogs.py` ✓
- `main.py` ✓
- `config.yaml` ✓
- Old monolithic files DELETED
- Full test suite
- API documentation

**Success Criteria**:
- Application works identically to before
- Clean MVC architecture
- GUI completely separated from logic
- Can swap GUI framework if needed
- All old files deleted
- 100% test coverage on business logic

---

## Final Directory Structure

```
JetsonSky/
├── main.py                          # Entry point (< 100 lines)
├── config.yaml                      # Configuration
├── requirements.txt                 # Dependencies
├── README.md                        # Updated documentation
│
├── core/                            # Business logic
│   ├── __init__.py
│   ├── controller.py                # MVC Controller (~300 lines)
│   ├── camera.py                    # Camera abstraction (~200 lines)
│   ├── camera_models.py             # Camera registry (~400 lines)
│   ├── image_processor.py           # Image processing (~250 lines)
│   └── config.py                    # Configuration classes (~200 lines)
│
├── gui/                             # Presentation layer
│   ├── __init__.py
│   ├── main_window.py               # Main window (~200 lines)
│   ├── controls.py                  # Control panels (~150 lines)
│   ├── display.py                   # Image display (~100 lines)
│   └── dialogs.py                   # Dialogs (~100 lines)
│
├── filters/                         # Image processing filters
│   ├── __init__.py
│   ├── base.py                      # Filter base class (~50 lines)
│   ├── pipeline.py                  # Filter pipeline (~100 lines)
│   ├── denoise.py                   # Denoise filters (~300 lines)
│   ├── sharpen.py                   # Sharpen filters (~150 lines)
│   ├── contrast.py                  # Contrast filters (~200 lines)
│   └── color.py                     # Color filters (~150 lines)
│
├── ai/                              # AI/ML integration
│   ├── __init__.py
│   ├── detector.py                  # Base detector (~100 lines)
│   ├── crater_detector.py           # Crater detection (~150 lines)
│   └── satellite_detector.py        # Satellite detection (~200 lines)
│
├── hardware/                        # Hardware interfaces
│   ├── __init__.py
│   ├── zwoasi_cupy/                 # Camera SDK (existing)
│   ├── zwoefw/                      # Filter wheel (existing)
│   └── synscan/                     # Mount control (existing)
│
├── io/                              # File I/O
│   ├── __init__.py
│   ├── image_io.py                  # Image I/O (~200 lines)
│   ├── video_io.py                  # Video I/O (~250 lines)
│   └── serfile.py                   # SER format (~339 lines)
│
├── utils/                           # Utilities
│   ├── __init__.py
│   ├── constants.py                 # Constants (~100 lines)
│   ├── threading.py                 # Threading utils (~150 lines)
│   └── astronomy.py                 # Astronomy calculations (~150 lines)
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── unit/                        # Unit tests
│   │   ├── test_filters.py
│   │   ├── test_camera_models.py
│   │   └── test_config.py
│   ├── integration/                 # Integration tests
│   │   ├── test_pipeline.py
│   │   └── test_camera.py
│   └── e2e/                         # End-to-end tests
│       └── test_acquisition.py
│
├── docs/                            # Documentation
│   ├── ARCHITECTURE_ANALYSIS.md     # This document
│   ├── ARCHITECTURE_DIAGRAMS_MERMAID.md
│   ├── REFACTORING_ROADMAP.md
│   ├── API.md                       # API documentation
│   └── USER_GUIDE.md                # User guide
│
├── AI_models/                       # AI models (existing)
│   ├── AI_craters_model6_8s_3c_180e.pt
│   ├── AI_Sat_model1_8n_3c_300e.pt
│   └── sattelite_custom_tracker.yaml
│
├── Videos/                          # User videos (existing)
├── Images/                          # User images (existing)
└── Lib/                             # ZWO SDK libraries (existing)
```

---

## Implementation Strategy

### Recommended Approach

**Option 1: Conservative (Recommended)**
- Work in a feature branch for each phase
- Keep old code working while building new modules
- Only delete old code when new code is fully tested
- Maintain backward compatibility throughout

**Option 2: Aggressive**
- Create new repository structure from scratch
- Port functionality piece by piece
- Test each piece before moving to next
- Switch over when complete

**Option 3: Hybrid (Best for Team)**
- Phase 1-3: Conservative (add modules, keep old code)
- Phase 4: Branch and test
- Phase 5: Complete rewrite of GUI layer
- Delete old code only when all tests pass

---

## Risk Management

### Potential Risks

1. **Breaking existing functionality**
   - Mitigation: Comprehensive testing at each phase
   - Keep old code until new code is proven

2. **Performance degradation**
   - Mitigation: Benchmark critical paths
   - Profile before and after refactoring

3. **Incomplete feature parity**
   - Mitigation: Feature checklist and user testing
   - Don't delete until verified equivalent

4. **Team resistance**
   - Mitigation: Clear documentation and training
   - Show benefits early (Phase 1-2)

5. **Timeline overrun**
   - Mitigation: Can stop at any phase
   - Each phase delivers value independently

---

## Testing Strategy

### Test Coverage Goals

- **Unit Tests**: 80%+ coverage
  - All filters individually tested
  - All data classes validated
  - All utility functions tested

- **Integration Tests**: 60%+ coverage
  - Filter pipeline tested end-to-end
  - Camera with SDK tested
  - I/O with different formats tested

- **End-to-End Tests**: Key workflows
  - Image acquisition workflow
  - Video capture workflow
  - Filter application workflow

### Test Infrastructure

```
tests/
├── conftest.py                      # Pytest configuration
├── fixtures/                        # Test data
│   ├── test_images/
│   ├── test_videos/
│   └── mock_camera_data/
├── unit/
│   ├── test_filters.py              # ~20 tests
│   ├── test_camera_models.py        # ~15 tests
│   ├── test_config.py               # ~10 tests
│   ├── test_astronomy.py            # ~10 tests
│   └── ...
├── integration/
│   ├── test_pipeline.py             # ~15 tests
│   ├── test_camera.py               # ~10 tests
│   ├── test_image_processor.py      # ~10 tests
│   └── ...
└── e2e/
    ├── test_acquisition_flow.py     # ~5 tests
    ├── test_video_capture_flow.py   # ~5 tests
    └── ...

Target: ~100 tests total
```

---

## Success Metrics

### Code Quality Metrics

- [ ] **Lines per file**: Average < 250 lines
- [ ] **Global variables**: 0
- [ ] **Cyclomatic complexity**: Average < 10
- [ ] **Function length**: Average < 30 lines
- [ ] **Test coverage**: > 80%
- [ ] **Documentation**: All public APIs documented

### Performance Metrics

- [ ] **Startup time**: < 3 seconds
- [ ] **Frame rate**: Maintained or improved
- [ ] **Memory usage**: No increase
- [ ] **Filter pipeline**: < 50ms overhead

### Maintainability Metrics

- [ ] **Time to find code**: < 10 seconds
- [ ] **Time to add filter**: < 30 minutes
- [ ] **Time to add camera**: < 15 minutes
- [ ] **Time to fix bug**: < 1 hour average
- [ ] **Code review time**: < 30 minutes per PR

---

## Quick Start Guide

### For Developers Starting Refactoring

1. **Read the architecture analysis**
   ```bash
   cd docs/
   cat ARCHITECTURE_ANALYSIS.md
   ```

2. **View the diagrams**
   - Open `ARCHITECTURE_DIAGRAMS_MERMAID.md` in GitHub
   - Or paste into https://mermaid.live

3. **Start with Phase 1**
   ```bash
   git checkout -b refactor/phase-1-foundation
   mkdir -p core utils
   touch core/config.py
   # Follow Phase 1 tasks...
   ```

4. **Run tests after each change**
   ```bash
   pytest tests/
   ```

5. **Commit frequently**
   ```bash
   git add .
   git commit -m "Phase 1: Add CameraConfig dataclass"
   ```

---

## Resources

### Documentation

- `ARCHITECTURE_ANALYSIS.md` - Detailed analysis of current vs proposed
- `ARCHITECTURE_DIAGRAMS_MERMAID.md` - Visual diagrams
- `REFACTORING_ROADMAP.md` - This document

### External Resources

- **Design Patterns**: "Design Patterns: Elements of Reusable Object-Oriented Software"
- **Clean Architecture**: "Clean Architecture" by Robert C. Martin
- **Python Best Practices**: PEP 8, PEP 257
- **Testing**: pytest documentation

---

## FAQ

**Q: Can we stop at Phase 3 and still get benefits?**
A: Yes! Each phase delivers value independently. Phase 1-3 gives you 75% of the benefits.

**Q: Will this break existing functionality?**
A: No, if done correctly. Each phase maintains compatibility until the switch.

**Q: How long will this really take?**
A: Depends on team size. One developer: 10 weeks. Two developers: 6 weeks. Three developers: 4 weeks.

**Q: What if we find issues during refactoring?**
A: Keep the old code working! Only delete when new code is proven equivalent.

**Q: Can we refactor just the filters first?**
A: Yes, but do Phase 1 first. You need the config classes as foundation.

**Q: Will performance be affected?**
A: Minimal impact. May even improve due to better cache locality. Benchmark to verify.

**Q: Do we need to rewrite everything?**
A: No! Hardware modules (zwoasi_cupy, zwoefw, synscan) are already well-structured. Keep them.

---

## Conclusion

This refactoring will transform JETSONSKY from an unmaintainable monolith into a professional, modular, testable application. The 10-week timeline is conservative and can be compressed with more developers or extended if needed.

**The key is starting with Phase 1**. Even just Phase 1 will make the code significantly more maintainable.

Ready to start? Begin with Phase 1, Task 1: Create `core/config.py`.

Good luck! 🚀
