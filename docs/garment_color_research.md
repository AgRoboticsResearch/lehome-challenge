# Research: Garment Color Randomization for Training Diversity

**Date:** 2026-03-31
**Status:** Research complete, implementation pending

## Context

The SmolVLA MoE expert model for `pant_long` achieves 60% overall success rate at best checkpoint (015000), but only 20% on unseen garments. The competition evaluates on 20 garments per category (10 seen + 2 unseen + 8 hidden generalization). Poor generalization to unseen appearances is a key bottleneck. Adding color/texture diversity to training data is one strategy to improve robustness.

---

## Current Model Performance (pant_long_0329)

| Metric | Value |
|--------|-------|
| Best checkpoint | 015000 (60% overall) |
| Seen garments avg | ~62% (range 20%-100%) |
| Unseen garments avg | 20% (both Unseen_0 & Unseen_1) |
| Weak spots | Seen_3 (20%), Seen_9 (60%) |
| Training | 20k steps, batch 32, expert-only MoE, ~15 hours |

### Per-Garment Breakdown (Best Checkpoint: 015000)

| Garment | Success Rate |
|---------|-------------|
| Pant_Long_Seen_0 | 40% |
| Pant_Long_Seen_1 | 60% |
| Pant_Long_Seen_2 | 100% |
| Pant_Long_Seen_3 | 20% |
| Pant_Long_Seen_4 | 100% |
| Pant_Long_Seen_5 | 100% |
| Pant_Long_Seen_6 | 60% |
| Pant_Long_Seen_7 | 60% |
| Pant_Long_Seen_8 | 80% |
| Pant_Long_Seen_9 | 60% |
| Pant_Long_Unseen_0 | 20% |
| Pant_Long_Unseen_1 | 20% |

---

## Garment Asset Structure

### Directory Layout
```
Assets/objects/Challenge_Garment/Release/
├── Color_Texture/           # 51 texture categories with USD materials
│   ├── Fabric004/, Fabric014/, ... Fabric082B/   (18 fabric textures)
│   ├── Tiles051/, Tiles054/, ... Tiles131/       (7 tile textures)
│   ├── Candy001/, Candy002/, Candy003/           (3 candy)
│   ├── Carpet006/, Carpet009/, Carpet010/, Carpet015/  (4 carpet)
│   ├── ChristmasTreeOrnament001/, ...006/        (4 ornament)
│   ├── Leather020/, Leather035A/                 (2 leather)
│   ├── PaintedPlaster002/,003/,005/              (3 plaster)
│   ├── GlazedTerracotta001/, PavingStones058/, SolarPanel004/
├── Pant_Long/               # 12 instances (10 seen + 2 unseen)
├── Pant_Short/              # 12 instances
├── Top_Long/                # 12 instances
├── Top_Short/               # 12 instances
```

### Per-Garment Files
Each garment has:
- `{name}_obj_exp.usd` — binary USD mesh (the garment geometry)
- `{name}_obj_exp.json` — config with `asset_path`, `visual_usd_paths`, `scale`, `initial_pos_range`, etc.
- Some have `texture/` subdirectory with Albedo, NRM, ORM PNGs

### Texture Categories (51 total)
Each texture has: BaseColor.jpg, Normal.jpg, Roughness.jpg, Displacement.jpg, and a .usd material file.

### Current Texture Assignments (sampled)

| Garment | visual_usd_paths |
|---------|-----------------|
| Pant_Long_Seen_0 | Tiles054, Fabric079 |
| Top_Short_Seen_0 | Fabric050 |
| Top_Short_Seen_3 | Fabric022, Fabric082B |
| Top_Short_Unseen_0 | Tiles081, Fabric079 |
| Top_Long_Seen_0 | [] (empty - uses embedded materials) |

**Finding:** Garments do NOT all have the same color. Some have multiple materials (for sub-meshes). Some have no external textures (embedded in USD binary). There is already variety, but it's fixed per garment instance.

---

## Material Loading Code Path

### Key Files
1. **JSON config** — `Assets/objects/Challenge_Garment/Release/{Type}/{GarmentName}/{name}.json`
2. **GarmentObject class** — `source/lehome/lehome/assets/object/Garment.py`
3. **Material binding** — `GarmentObject._apply_visual_material()` (lines 517-595)
4. **Task environment** — `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py`

### How It Works
1. `ChallengeGarmentLoader` reads garment JSON config
2. `GarmentObject.__init__` receives `visual_usd_paths` from config
3. `_apply_visual_material()` loads each material USD file via `add_reference_to_stage()`
4. Uses `omni.kit.commands.execute("BindMaterialCommand")` to bind materials to mesh sub-prims (mesh, mesh1, mesh2...)
5. If `visual_usd_paths` is empty, the garment uses materials embedded in its USD mesh file

### Existing Randomization Infrastructure
- `particle_garment_cfg.yaml` has `texture_randomization` and `light_randomization` fields (currently disabled)
- `garment_bi_v2.py` already does table surface texture randomization
- `PreviewSurface` API (from Isaac Sim) supports `set_color()`, `set_roughness()`, `set_metallic()`

---

## Approaches for Color Diversity

### Approach 1: Swap JSON Texture Paths (Simplest)
- Edit garment JSON files to point `visual_usd_paths` to different Color_Texture entries
- Record teleoperation data with each variant, merge datasets
- **Pros:** No code changes, uses existing material library
- **Cons:** Manual per-variant recording, limited to 51 existing textures

### Approach 2: Runtime Texture Randomization in GarmentObject
- Modify `_apply_visual_material()` to randomly select from Color_Texture library on each episode reset
- Works during both data recording and evaluation
- **Pros:** Automatic diversity, every episode has different appearance
- **Cons:** Need to handle garments without `visual_usd_paths` (embedded materials)

### Approach 3: Programmatic Color Tinting via USD API
- After material binding, use PreviewSurface `set_color()` to apply random RGB tint
- Can work on any garment regardless of texture setup
- **Pros:** Maximum flexibility, works on all garments including embedded materials
- **Cons:** Tinting may look unrealistic if base texture has strong patterns

### Approach 4: Enable Existing Randomization Config
- Toggle `texture_randomization` in `particle_garment_cfg.yaml`
- **Pros:** May already be implemented
- **Cons:** Need to verify what this actually does — it may be for table surface only

---

## Recommendations for Implementation (Future)

1. **Best combined approach:** Runtime texture randomization (Approach 2) + programmatic tinting (Approach 3) as fallback for garments with embedded materials
2. **Key code changes needed:**
   - `source/lehome/lehome/assets/object/Garment.py` — add random material selection in `_apply_visual_material()`
   - Possibly `garment_bi_v2.py` — add reset hook to re-randomize materials
3. **Important consideration:** Since this is imitation learning, the model only sees what's in recorded demonstrations. Runtime randomization during recording = diverse training data. Runtime randomization during eval = tests generalization.
