# JETSONSKY Documentation

This directory contains comprehensive documentation for understanding and refactoring the JETSONSKY codebase.

## Documents Overview

### 📊 [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md)
**Complete analysis of current monolithic code and proposed modular architecture**

Contains:
- Current architecture breakdown (ASCII diagrams)
- Detailed problem identification
- Proposed modular structure
- Component interaction patterns
- Data flow diagrams
- Migration strategy with 5 phases
- Benefits and metrics comparison

**Read this first** to understand the scope of the monolithic problem and the solution.

---

### 🎨 [ARCHITECTURE_DIAGRAMS_MERMAID.md](./ARCHITECTURE_DIAGRAMS_MERMAID.md)
**Visual architecture diagrams in Mermaid format**

Contains:
- Current monolithic architecture diagrams
- Proposed modular architecture diagrams
- Sequence diagrams for key workflows
- Class diagrams for major components
- Data flow visualizations
- Migration timeline (Gantt chart)
- Testing pyramid
- Metrics comparisons

**View these diagrams** in:
- GitHub/GitLab (automatic rendering)
- VS Code (with Mermaid extension)
- https://mermaid.live (paste the code)
- Any markdown viewer with Mermaid support

---

### 🗺️ [REFACTORING_ROADMAP.md](./REFACTORING_ROADMAP.md)
**Step-by-step implementation guide for refactoring**

Contains:
- Executive summary with quick metrics
- Detailed 5-phase breakdown (10 weeks)
- Task checklists for each phase
- Deliverables and success criteria
- Final directory structure
- Implementation strategies
- Risk management
- Testing strategy
- Success metrics
- FAQ

**Use this as your implementation guide** when starting the refactoring work.

---

## Quick Navigation

### Understanding the Problem
1. Read: [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md) - Section "Current Monolithic Architecture"
2. View: [ARCHITECTURE_DIAGRAMS_MERMAID.md](./ARCHITECTURE_DIAGRAMS_MERMAID.md) - "Current Monolithic Architecture" diagrams

### Understanding the Solution
1. Read: [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md) - Section "Proposed Modular Architecture"
2. View: [ARCHITECTURE_DIAGRAMS_MERMAID.md](./ARCHITECTURE_DIAGRAMS_MERMAID.md) - "Proposed Modular Architecture" diagrams

### Planning the Refactoring
1. Read: [REFACTORING_ROADMAP.md](./REFACTORING_ROADMAP.md) - Complete implementation guide
2. View: [ARCHITECTURE_DIAGRAMS_MERMAID.md](./ARCHITECTURE_DIAGRAMS_MERMAID.md) - "Refactoring Migration Strategy" Gantt chart

### Starting Implementation
1. Read: [REFACTORING_ROADMAP.md](./REFACTORING_ROADMAP.md) - "Phase 1: Foundation"
2. Follow the task checklist
3. Run tests after each task
4. Move to next phase when complete

---

## Key Insights

### The Problem (In Numbers)

| Metric | Value |
|--------|-------|
| **Largest file** | 11,301 lines |
| **Global variables** | 300+ |
| **Module-level functions** | 143 |
| **Camera init function** | 1,500 lines |
| **Duplicate versions** | 4 files, 44K total lines |
| **Test coverage** | 0% |
| **Maintainability** | Impossible |

### The Solution (In Numbers)

| Metric | Value |
|--------|-------|
| **Largest file** | < 500 lines (96% reduction) |
| **Global variables** | 0 (100% elimination) |
| **Average file size** | ~200 lines |
| **Modules** | 25+ well-organized |
| **Camera init function** | 10 lines (99% reduction) |
| **Versions** | 1 with configuration |
| **Test coverage** | 80%+ target |
| **Maintainability** | Professional grade |

### The Impact

**Before Refactoring:**
- ❌ Can't test individual components
- ❌ Can't find code (search 11K lines)
- ❌ Can't add features without breaking others
- ❌ Can't work in parallel (merge conflicts)
- ❌ Can't understand code flow
- ❌ Can't onboard new developers

**After Refactoring:**
- ✅ Full test coverage
- ✅ Find code in seconds (navigate by module)
- ✅ Add features safely (isolated modules)
- ✅ Team can work in parallel (different modules)
- ✅ Clear architecture (MVC pattern)
- ✅ Easy onboarding (documented modules)

---

## Refactoring Timeline

```
Phase 1: Foundation (Weeks 1-2)
  └─> Extract config, camera registry, constants

Phase 2: Filters (Weeks 3-4)
  └─> Create filter pipeline and individual filters

Phase 3: Hardware & I/O (Weeks 5-6)
  └─> Abstract camera control and file operations

Phase 4: AI & Utilities (Weeks 7-8)
  └─> Separate AI detection and utility functions

Phase 5: GUI Layer (Weeks 9-10)
  └─> Implement MVC pattern, create main.py
```

**Total**: 10 weeks (can be parallelized with multiple developers)

---

## Code Reduction by Phase

```
Start:    11,301 lines (monolithic)
Phase 1:  11,301 lines (+ new modules, no deletion yet)
Phase 2:   9,000 lines (-2,000 lines of filters)
Phase 3:   6,000 lines (-3,000 lines of camera/IO)
Phase 4:   4,000 lines (-2,000 lines of AI/utils)
Phase 5:     100 lines (-3,900 lines, GUI extracted)
Final:    ~5,500 total lines across 25+ modules
```

**Total reduction**: 51% fewer lines, infinitely better organized!

---

## Architecture Patterns Used

### MVC (Model-View-Controller)
- **Model**: `AppState`, `Camera`, `ImageProcessor`
- **View**: `MainWindow`, `Controls`, `Display`
- **Controller**: `JetsonSkyController`

### Dependency Injection
- Components receive dependencies via constructor
- Easy to mock for testing
- Clear dependency graph

### Factory Pattern
- Camera registry creates camera configs
- Filter factory creates filter instances

### Strategy Pattern
- Filters implement common interface
- Can swap filter implementations

### Pipeline Pattern
- `FilterPipeline` chains filters sequentially
- Each filter transforms image and passes to next

### Observer Pattern
- State changes notify controller
- Controller updates view

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.x |
| **GUI** | Tkinter (can be swapped) |
| **Image Processing** | CuPy, NumPy, OpenCV |
| **AI/ML** | PyTorch, YOLOv8 |
| **Hardware** | ZWO SDK (ASI cameras) |
| **Testing** | pytest |
| **Documentation** | Markdown, docstrings |
| **Type Checking** | Python type hints |

---

## File Structure (Final)

```
JetsonSky/
├── main.py                          # Entry point (< 100 lines)
├── config.yaml                      # Configuration
│
├── core/                            # Business logic (1,350 lines)
│   ├── controller.py
│   ├── camera.py
│   ├── camera_models.py
│   ├── image_processor.py
│   └── config.py
│
├── gui/                             # Presentation (550 lines)
│   ├── main_window.py
│   ├── controls.py
│   ├── display.py
│   └── dialogs.py
│
├── filters/                         # Filters (950 lines)
│   ├── base.py
│   ├── pipeline.py
│   ├── denoise.py
│   ├── sharpen.py
│   ├── contrast.py
│   └── color.py
│
├── ai/                              # AI/ML (450 lines)
│   ├── detector.py
│   ├── crater_detector.py
│   └── satellite_detector.py
│
├── hardware/                        # Hardware (existing)
│   ├── zwoasi_cupy/
│   ├── zwoefw/
│   └── synscan/
│
├── io/                              # File I/O (650 lines)
│   ├── image_io.py
│   ├── video_io.py
│   └── serfile.py
│
├── utils/                           # Utilities (400 lines)
│   ├── constants.py
│   ├── threading.py
│   └── astronomy.py
│
├── tests/                           # Test suite (target: 100+ tests)
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── docs/                            # Documentation (this directory)
    ├── README.md
    ├── ARCHITECTURE_ANALYSIS.md
    ├── ARCHITECTURE_DIAGRAMS_MERMAID.md
    └── REFACTORING_ROADMAP.md
```

---

## Getting Started

### For Understanding the Codebase
1. Read `ARCHITECTURE_ANALYSIS.md` (30 minutes)
2. View diagrams in `ARCHITECTURE_DIAGRAMS_MERMAID.md` (15 minutes)
3. Review current code with new understanding

### For Planning Refactoring
1. Read `REFACTORING_ROADMAP.md` (45 minutes)
2. Review phase-by-phase breakdown
3. Estimate timeline for your team
4. Create project plan

### For Implementing Refactoring
1. Create feature branch: `git checkout -b refactor/phase-1`
2. Follow Phase 1 tasks in `REFACTORING_ROADMAP.md`
3. Test after each task
4. Commit frequently
5. Move to next phase when complete

---

## Key Principles

### SOLID Principles

**S - Single Responsibility**
- Each class has one reason to change
- `Camera` handles camera operations only
- `FilterPipeline` handles filter application only

**O - Open/Closed**
- Open for extension, closed for modification
- Add new filters without changing `FilterPipeline`
- Add new cameras without changing `Camera` class

**L - Liskov Substitution**
- All filters can be used interchangeably
- All detectors can be used interchangeably

**I - Interface Segregation**
- Clients don't depend on methods they don't use
- `Filter` interface is minimal
- Specific interfaces for specific needs

**D - Dependency Inversion**
- Depend on abstractions, not concretions
- Controller depends on `Camera` interface, not specific camera
- Pipeline depends on `Filter` interface, not specific filter

### Clean Code Principles

- **Meaningful names**: `CameraConfig` not `cc`
- **Small functions**: < 30 lines average
- **Single level of abstraction**: Don't mix high and low level
- **No side effects**: Functions do what their name says
- **DRY**: Don't repeat yourself
- **Comments**: Why, not what

### Testing Principles

- **Arrange-Act-Assert**: Clear test structure
- **One assertion per test**: Focus on one thing
- **Test behavior, not implementation**: Test what, not how
- **Fast tests**: Unit tests run in milliseconds
- **Independent tests**: No dependencies between tests

---

## Success Stories (After Refactoring)

### Adding a New Filter
**Before**: Search 11K lines, hope you don't break anything, 4 hours
**After**: Create new Filter subclass, 30 minutes

### Supporting a New Camera
**Before**: Copy-paste 50 lines into if-elif hell, 2 hours
**After**: Add entry to `CAMERA_MODELS` dictionary, 15 minutes

### Fixing a Bug
**Before**: Search everywhere, trace through globals, 4 hours
**After**: Navigate to module, fix, test, 30 minutes

### Onboarding a Developer
**Before**: "Good luck understanding this 11K line file", 2 weeks
**After**: "Read the docs, modules are self-contained", 2 days

### Adding Tests
**Before**: Impossible (global state, hardware dependencies)
**After**: Easy (mocked dependencies, isolated components)

---

## Maintenance

### Adding New Cameras
1. Add entry to `CAMERA_MODELS` in `core/camera_models.py`
2. Test with camera if available
3. Done!

### Adding New Filters
1. Create new class in appropriate `filters/*.py` file
2. Inherit from `Filter` base class
3. Implement `apply()` and `get_name()` methods
4. Add to filter panel in GUI
5. Write unit tests

### Adding New Features
1. Identify which layer the feature belongs to
2. Create module in appropriate directory
3. Wire into controller
4. Add GUI controls if needed
5. Write tests

---

## Resources

### Internal Documentation
- `ARCHITECTURE_ANALYSIS.md` - Architecture details
- `ARCHITECTURE_DIAGRAMS_MERMAID.md` - Visual diagrams
- `REFACTORING_ROADMAP.md` - Implementation guide
- Code docstrings - API documentation

### External Resources
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) by Robert C. Martin
- [Python Design Patterns](https://refactoring.guru/design-patterns/python)
- [PEP 8](https://peps.python.org/pep-0008/) - Python Style Guide
- [pytest Documentation](https://docs.pytest.org/)

---

## Contact & Support

For questions about this documentation or the refactoring process:
1. Review the FAQ in `REFACTORING_ROADMAP.md`
2. Check the diagrams in `ARCHITECTURE_DIAGRAMS_MERMAID.md`
3. Read the detailed analysis in `ARCHITECTURE_ANALYSIS.md`

---

## Version History

- **v1.0** (2025-10-21): Initial architecture analysis and refactoring plan
  - Analyzed monolithic codebase (11,301 lines)
  - Created comprehensive refactoring plan (5 phases, 10 weeks)
  - Generated architecture diagrams (ASCII and Mermaid)
  - Documented implementation roadmap

---

## License

This documentation is part of the JETSONSKY project.
Copyright Alain Paillou 2018-2025
Free for personal and non-commercial use.

---

**Ready to start refactoring? Begin with Phase 1 in [REFACTORING_ROADMAP.md](./REFACTORING_ROADMAP.md)!** 🚀
