# FAL Codegen Completion Plan

## Executive Summary

This plan outlines the remaining work to achieve 100% coverage of FAL endpoints in the nodetool-fal package. The current state is **54.43% coverage (584/1073 models configured)** with 519 nodes generated.

**Target Date:** TBD
**Current Coverage:** 54.43% (584/1073)
**Target Coverage:** 100% (1,073/1,073)
**Remaining Work:** 489 endpoints to configure

---

## Current State Analysis

### Completed (15 modules, 584 configs, 519 nodes)

| Module | Configured | Total | Coverage | Nodes Generated |
|--------|-----------|-------|----------|-----------------|
| speech-to-text | 9 | 9 | 100% ✅ | 9 |
| text-to-audio | 17 | 34 | 50% 🟢 | 19 |
| text-to-image | 69 | 138 | 50% 🟢 | 75 |
| text-to-video | 49 | 98 | 50% 🟢 | 46 |
| vision | 17 | 34 | 50% 🟢 | 17 |
| audio-to-video | 7 | 14 | 50% 🟢 | 6 |
| llm | 3 | 6 | 50% 🟢 | 3 |
| training | 15 | 31 | 48.4% 🟡 | 13 |
| image-to-3d | 12 | 25 | 48% 🟡 | 11 |
| image-to-image | 172 | 345 | 49.9% 🟡 | 175 |
| image-to-video | 68 | 137 | 49.6% 🟡 | 62 |
| video-to-video | 63 | 127 | 49.6% 🟡 | 63 |
| audio-to-audio | 8 | 17 | 47.1% 🟡 | 8 |
| text-to-speech | 10 | 25 | 40% 🟡 | 10 |
| 3d-to-3d | 2 | 5 | 40% 🟡 | 2 |

### Not Started (11 new modules, 0 configs)

| Category | Total | Priority | Notes |
|----------|-------|----------|-------|
| audio-to-text | 3 | Low | Speech recognition/diarization variants |
| image-to-json | 1 | Low | Image analysis to JSON output |
| json | 3 | Medium | JSON processing endpoints |
| speech-to-speech | 2 | Medium | Voice transformation/conversion |
| text-to-3d | 4 | Low | 3D model generation from text |
| text-to-json | 4 | Low | Text analysis to JSON |
| text-to-text | 1 | Low | Text transformation |
| unknown | 4 | Low | Uncategorized endpoints |
| video-to-audio | 4 | Medium | Audio extraction from video |
| video-to-text | 2 | Low | Video transcription |

---

## Phased Completion Plan

### Phase 1: Complete Existing Modules to 100% (Priority: HIGH)

**Goal:** Extend existing 15 modules from ~50% to 100% coverage

#### 1.1 High-Value Modules (Large remaining counts)

| Module | Need | Priority | Rationale |
|--------|------|----------|-----------|
| image-to-image | 173 | HIGH | Largest category, high demand |
| text-to-image | 69 | HIGH | Core generative AI capability |
| image-to-video | 69 | HIGH | Popular video animation use case |
| video-to-video | 64 | HIGH | Video editing workflows |
| text-to-video | 49 | HIGH | Text-to-video generation |

**Subtotal: 424 configs**

#### 1.2 Medium-Priority Modules

| Module | Need | Priority |
|--------|------|----------|
| vision | 17 | MEDIUM |
| text-to-audio | 17 | MEDIUM |
| text-to-speech | 15 | MEDIUM |
| training | 16 | MEDIUM |
| image-to-3d | 13 | MEDIUM |
| audio-to-audio | 9 | MEDIUM |
| audio-to-video | 7 | MEDIUM |
| 3d-to-3d | 3 | MEDIUM |

**Subtotal: 97 configs**

**Phase 1 Total: 521 configs**

### Phase 2: Create New Module Categories (Priority: MEDIUM)

**Goal:** Add support for 11 new categories

#### 2.1 Moderate Priority (Useful capabilities)

| Category | Total | Use Case |
|----------|-------|----------|
| video-to-audio | 4 | Audio extraction, music separation |
| speech-to-speech | 2 | Voice cloning, transformation |
| json | 3 | JSON processing/analysis |

**Subtotal: 9 configs**

#### 2.2 Low Priority (Specialized use cases)

| Category | Total | Use Case |
|----------|-------|----------|
| text-to-3d | 4 | 3D model generation |
| audio-to-text | 3 | Advanced speech features |
| text-to-json | 4 | Text analysis |
| video-to-text | 2 | Video transcription |
| image-to-json | 1 | Image analysis |
| text-to-text | 1 | Text processing |
| unknown | 4 | Uncategorized |

**Subtotal: 19 configs**

**Phase 2 Total: 28 configs**

---

## Implementation Strategy

### Step 1: Configuration Generation

For each batch of endpoints:

1. **Use auto_config_generator.py** to generate config templates
   ```bash
   python codegen/auto_config_generator.py
   ```

2. **Review and enhance templates** with:
   - Accurate docstrings from FAL descriptions
   - Relevant tags for searchability
   - Concrete use cases (5 per endpoint)
   - Field overrides for naming consistency
   - Enum overrides to prevent conflicts

3. **Merge into config files** using merge_configs.py
   ```bash
   python codegen/merge_configs.py
   ```

### Step 2: Node Generation

1. **Generate nodes for module**:
   ```bash
   python codegen/generate.py --module <module_name> --output-dir src/nodetool/nodes/fal
   ```

2. **Validate generated code**:
   - Check Python syntax: `python -m py_compile <file>`
   - Run linting: `ruff check <file>`

3. **Fix enum conflicts** using nested classes (already implemented)

### Step 3: DSL Generation

1. **Run package scan**:
   ```bash
   nodetool package scan
   ```

2. **Generate DSL wrappers**:
   ```bash
   nodetool codegen
   ```

### Step 4: Testing

1. **Import test**:
   ```bash
   python -c "from nodetool.nodes.fal.<module> import *"
   ```

2. **Sample node execution** (if API access available)

---

## Module-by-Module Roadmap

### Module 1: image-to-image (173 remaining)

**Strategy:** Add in batches of 25-30

- Batch 1: FLUX variants and advanced models (30)
- Batch 2: Editing and inpainting models (30)
- Batch 3: Style transfer and transformation (30)
- Batch 4: Professional editing suites (30)
- Batch 5: Legacy and specialized models (30)
- Batch 6: Remaining models (23)

**Completion:** 100% (345/345)

### Module 2: text-to-image (69 remaining)

**Strategy:** Focus on popular and diverse providers

- Batch 1: Midjourney-like models (25)
- Batch 2: Stable diffusion variants (25)
- Batch 3: Specialized style models (19)

**Completion:** 100% (138/138)

### Module 3: image-to-video (69 remaining)

**Strategy:** Prioritize quality and speed variants

- Batch 1: Major provider models (25)
- Batch 2: Avatar and animation (25)
- Batch 3: Specialized video effects (19)

**Completion:** 100% (137/137)

### Module 4: video-to-video (64 remaining)

**Strategy:** Cover editing, enhancement, and transformation

- Batch 1: Video editing suites (25)
- Batch 2: Enhancement and upscaling (25)
- Batch 3: Specialized effects (14)

**Completion:** 100% (127/127)

### Module 5: text-to-video (49 remaining)

**Strategy:** Add major providers and quality tiers

- Batch 1: Premium video generation (25)
- Batch 2: Fast and specialized variants (24)

**Completion:** 100% (98/98)

### Modules 6-15: Medium-Priority (97 remaining)

Process in parallel batches of 15-20 configs each.

### Modules 16-26: New Categories (28 total)

Create new config files and node modules for each category.

---

## Tools and Scripts

### Existing Tools

| Script | Purpose |
|--------|---------|
| `codegen/generate.py` | Generate nodes from configs |
| `codegen/generate_all.py` | Bulk generate all modules |
| `codegen/auto_config_generator.py` | Auto-generate config templates |
| `codegen/merge_configs.py` | Merge configs into existing files |
| `codegen/bulk_config_generator.py` | Bulk config generation |

### New Tools Needed

1. **Coverage tracker** - Real-time coverage calculation
2. **Batch selector** - Select endpoints by category/provider
3. **Conflict detector** - Find enum naming conflicts before generation
4. **Validation script** - Comprehensive post-generation validation

---

## Risk Mitigation

### Risk 1: Enum Name Conflicts
**Status:** ✅ RESOLVED - Using nested classes (see commit b6c2a9c)

### Risk 2: API Schema Changes
**Mitigation:** Schema caching with `--no-cache` flag for refresh

### Risk 3: Large-Scale Generation Failures
**Mitigation:** Generate in batches with timeout protection

### Risk 4: DSL Generation Errors
**Mitigation:** Run `nodetool package scan` first to validate

---

## Success Metrics

### Coverage Targets

| Milestone | Coverage | Configs | Date |
|-----------|----------|---------|------|
| Current | 54.43% | 584 | Feb 7, 2026 |
| M1 | 75% | 805 | TBD |
| M2 | 90% | 966 | TBD |
| M3 | 100% | 1,073 | TBD |

### Quality Gates

- ✅ All generated code compiles without syntax errors
- ✅ All nodes pass ruff linting
- ✅ All nodes have complete docstrings and use cases
- ✅ DSL generation completes successfully
- ✅ Package scan finds all nodes

---

## Estimated Effort

### By Phase

| Phase | Configs | Est. Time |
|-------|---------|-----------|
| Phase 1.1 (High-value) | 424 | 2-3 weeks |
| Phase 1.2 (Medium) | 97 | 1 week |
| Phase 2.1 (New mod) | 9 | 3-5 days |
| Phase 2.2 (Specialized) | 19 | 1 week |
| Testing & Validation | - | 3-5 days |
| **Total** | **549** | **4-6 weeks** |

---

## Next Immediate Actions

### Week 1: Infrastructure + First Batch

1. ✅ Verify all tools are working
2. ✅ Create coverage tracking script
3. Generate Batch 1 of image-to-image (30 configs)
4. Generate and validate nodes
5. Complete DSL generation

### Week 2: High-Value Modules

1. Continue image-to-image batches (60 more)
2. Start text-to-image expansion (30 configs)
3. Begin image-to-video expansion (30 configs)

### Week 3-4: Core Modules

1. Complete Phase 1.1 (all high-value modules)
2. Begin Phase 1.2 (medium-priority modules)

### Week 5-6: Final Push

1. Create new module categories
2. Complete all remaining configs
3. Final validation and testing
4. Documentation

---

## Code Quality Standards

### Configuration Standards

- **Docstrings:** Clear, descriptive, includes capabilities
- **Tags:** 5-7 relevant tags for searchability
- **Use Cases:** 5 concrete, actionable use cases
- **Field Overrides:** Consistent naming across modules
- **Enum Overrides:** Prevent naming conflicts

### Generated Code Standards

- **Python 3.11+ syntax**
- **Type hints:** Complete pydantic Field definitions
- **Nesting:** Enums nested inside node classes
- **Imports:** Minimal, only what's used
- **Formatting:** Consistent with codebase style

---

## Appendix: Category Details

### New Category Templates

#### audio-to-text.py
```python
"""
Configuration for audio-to-text module.
Advanced speech recognition and diarization beyond speech-to-text.
"""
CONFIGS = {
    # Endpoints for audio-to-text
}
```

#### video-to-audio.py
```python
"""
Configuration for video-to-audio module.
Audio extraction and processing from video sources.
"""
CONFIGS = {
    # Endpoints for video-to-audio
}
```

### Reference: Shared Enums

Use `SHARED_ENUMS` for common enum values across modules:

```python
SHARED_ENUMS = {
    "ImageSizePreset": {
        "values": [...],
        "description": "..."
    }
}
```

---

**Document Version:** 1.0
**Last Updated:** February 7, 2026
**Status:** Ready for Execution
