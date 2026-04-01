# Mesh vs Visual Type: Success Rate Analysis

**Date:** April 2026  
**Policy:** SmolVLA MoE Expert (no state projection variant)  
**Evaluation:** 5 episodes per garment variant, best checkpoint selected per category

---

## 1. Definitions

| Term | Source Field | Role |
|---|---|---|
| **Mesh** | `asset_path` in `*_obj_exp.json` | Physics collision geometry (cloth particle topology). Determines simulation behavior. |
| **Visual** | `visual_usd_paths` in `*_obj_exp.json` | Material/texture overlays (PBR: color, normal, roughness). Purely cosmetic, no physics effect. |

Multiple garment variants can share the **same mesh** but have **different visuals**, enabling isolation of visual impact on success rates.

---

## 2. Overall Results (Best Checkpoint)

| Experiment | Garment Type | Best Ckpt | Success Rate | Garments Tested |
|---|---|---|---|---|
| pant_short_no_st_proj | Pant_Short | 011000 | **90.00%** | 12 (10 seen + 2 unseen) |
| top_long_no_st_proj | Top_Long | 015000 | **78.33%** | 12 |
| pant_long_no_st_proj | Pant_Long | 019000 | **58.33%** | 12 |
| pant_long_0329 | Pant_Long | 013000 | **58.33%** | 12 |
| top_short_no_st_proj | Top_Short | 020000 | **51.67%** | 12 |

---

## 3. Shared-Mesh Analysis (Same Geometry, Different Visuals)

### 3.1 Pant_Short — 4 meshes across 10 garments — Best Ckpt 011000

```
┌─────────────┬────────────┬───────────────┬───────────────────────────┬──────────┐
│    Mesh     │  Garments  │ Visual Count  │ Avg Success (ckpt 011000) │  Range   │
├─────────────┼────────────┼───────────────┼───────────────────────────┼──────────┤
│ PS_049      │ Seen_0,1,2 │ 2 each        │ 93%                       │ 80-100%  │
├─────────────┼────────────┼───────────────┼───────────────────────────┼──────────┤
│ PS_050      │ Seen_3,4,5 │ 2 each        │ 100%                      │ 100-100% │
├─────────────┼────────────┼───────────────┼───────────────────────────┼──────────┤
│ PS_M1_089   │ Seen_6,7   │ 1 each        │ 90%                       │ 80-100%  │
├─────────────┼────────────┼───────────────┼───────────────────────────┼──────────┤
│ PS_Short047 │ Seen_8,9   │ 1 each        │ 100%                      │ 100-100% │
├─────────────┼────────────┼───────────────┼───────────────────────────┼──────────┤
│ PS_Short130 │ Unseen_0   │ 1             │ 0%                        │ —        │
└─────────────┴────────────┴───────────────┴───────────────────────────┴──────────┘
```

### 3.2 Top_Short — 4 meshes across 10 garments — Best Ckpt 020000

```
┌──────────────────┬────────────┬───────────────┬───────────────────────────┬─────────┐
│       Mesh       │  Garments  │ Visual Count  │ Avg Success (ckpt 020000) │  Range  │
├──────────────────┼────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCSC_067         │ Seen_0,1,2 │ 1 each        │ 73%                       │ 60-80%  │
├──────────────────┼────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCSC_Top004_1    │ Seen_3,4   │ 4 each        │ 60%                       │ 20-100% │
├──────────────────┼────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCSO_Baggy_Shirt │ Seen_5,6,7 │ 2 each        │ 73%                       │ 60-80%  │
├──────────────────┼────────────┼───────────────┼───────────────────────────┼─────────┤
│ TNSC_Tshirt3     │ Seen_8,9   │ 1 each        │ 10%                       │ 0-20%   │
└──────────────────┴────────────┴───────────────┴───────────────────────────┴─────────┘
```

### 3.3 Top_Long — 11 unique meshes across 12 garments — Best Ckpt 015000

Each garment has a **unique mesh** except TCLC_015 (shared between Seen_1 and Unseen_1). Visual count ranges from 0-2 with no clear correlation.

```
┌───────────┬─────────────┬───────────────┬───────────────────────────┬─────────┐
│   Mesh    │  Garments   │ Visual Count  │ Avg Success (ckpt 015000) │  Range  │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCLC_015  │ Seen_1,     │ 1 each        │ 60%                       │ 20-100% │
│           │ Unseen_1    │               │ (100%, 20%)               │         │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCLC_002  │ Seen_0      │ 0 (embedded)  │ 80%                       │ —       │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCLC_028  │ Seen_2      │ 2             │ 60%                       │ —       │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCLO_001  │ Seen_3      │ 2             │ 80%                       │ —       │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCLO_073  │ Seen_4      │ 2             │ 80%                       │ —       │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCLO_027  │ Seen_5      │ 2             │ 80%                       │ —       │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TNLC_010  │ Seen_6      │ 1             │ 100%                      │ —       │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TNLC_033  │ Seen_7      │ 2             │ 40%                       │ —       │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TNLO_008  │ Seen_8      │ 1             │ 100%                      │ —       │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TNLO_034  │ Seen_9      │ 2             │ 40%                       │ —       │
├───────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ TCLC_018  │ Unseen_0    │ 2             │ 100%                      │ —       │
└───────────┴─────────────┴───────────────┴───────────────────────────┴─────────┘
```

The only shared mesh is **TCLC_015** (Seen_1=100% vs Unseen_1=20%) — same geometry but the unseen variant is dramatically harder, likely due to different success thresholds or initial pose.

### 3.4 Pant_Long — 12 unique meshes across 12 garments — Best Ckpt 019000

**Zero mesh sharing** — every garment is geometrically unique. The spread is 0%-100% across meshes.

```
┌──────────────┬─────────────┬───────────────┬───────────────────────────┬─────────┐
│     Mesh     │  Garments   │ Visual Count  │ Avg Success (ckpt 019000) │  Range  │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_053       │ Seen_4      │ 2             │ 100%                      │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_079       │ Seen_5      │ 1             │ 80%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_019       │ Seen_0      │ 2             │ 60%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_022       │ Seen_1      │ 2             │ 60%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_Pants100  │ Seen_7      │ 1             │ 60%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_Pants109  │ Unseen_0    │ 1             │ 60%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_030       │ Seen_2      │ 1             │ 40%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_Pants095  │ Seen_6      │ 1             │ 40%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_Pants106  │ Seen_8      │ 1             │ 40%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_Pants121  │ Unseen_1    │ 2             │ 40%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_046       │ Seen_3      │ 2             │ 20%                       │ —       │
├──────────────┼─────────────┼───────────────┼───────────────────────────┼─────────┤
│ PL_Pants107  │ Seen_9      │ 1             │ 0%                        │ —       │
└──────────────┴─────────────┴───────────────┴───────────────────────────┴─────────┘
```

---

## 4. Visual Count vs Success Rate

Aggregating all garments across all categories at their best checkpoint:

| Visual Count | # Garments | Avg Success Rate | Std Dev |
|---|---|---|---|
| 0 | 1 | 80.0% | — |
| 1 | 16 | 67.5% | 30.8pp |
| 2 | 26 | 68.1% | 26.2pp |
| 4 | 3 | 60.0% | 40.0pp |

**Conclusion:** No meaningful correlation between visual count and success rate.

---

## 5. Within-Mesh Variance (Visual Impact Isolation)

For garment groups sharing identical mesh geometry, the spread in success rates attributable to visual differences:

| Mesh | Category | Min | Max | Spread |
|---|---|---|---|---|
| PS_049 | Pant_Short | 80% | 100% | 20pp |
| PS_050 | Pant_Short | 100% | 100% | 0pp |
| PS_M1_089 | Pant_Short | 80% | 100% | 20pp |
| PS_Short047 | Pant_Short | 100% | 100% | 0pp |
| TCSC_067 | Top_Short | 60% | 80% | 20pp |
| TCSC_Top004 | Top_Short | 20% | 100% | 80pp |
| TCSO_Baggy | Top_Short | 60% | 80% | 20pp |
| TNSC_Tshirt3 | Top_Short | 0% | 20% | 20pp |

**Typical visual-only spread: ~20pp** (except TCSC_Top004 at 80pp, which may be confounded by its larger scale).

---

## 6. Scale Impact

| Scale | Garments | Avg Success |
|---|---|---|
| 0.37 | Pant_Long (all 12) | 50.0% |
| 0.40 | Top_Long_Seen_3 | 80.0% |
| 0.45 | Most Top & Pant_Short | ~73% |
| 0.65 | Top_Short_Seen_3,4 + Unseen_1 | 60.0% |

Scale differences across categories confound direct comparison. Within Top_Short, the 0.65-scaled `TCSC_Top004` achieves 60% vs 73% for the 0.45-scaled majority.

---

## 7. Summary Diagram

```
Impact on Success Rate
========================

Mesh Geometry (cloth topology/shape)
████████████████████████████████  HIGH — dominant factor, determines difficulty
                                   Cross-mesh spread: 10%-100%

Scale
████████████████████              MEDIUM — larger garments harder to manipulate
                                   0.37→50%, 0.45→73%, 0.65→60%

Visual Material (texture/color)
████████                          LOW — ~20pp variance within same mesh
                                   Policy generalizes across appearance

Visual Count (# sub-meshes)
██                                 NEGLIGIBLE — no correlation found
                                   0→80%, 1→68%, 2→68%, 4→60%
```

---

## 8. Key Takeaways

1. **Mesh geometry is the dominant factor** in success rate. In Top_Short, cross-mesh spread is 10%-73%, while within-mesh (visual-only) spread is ~20pp.

2. **Visual materials have minimal impact.** The policy generalizes well across different textures/colors applied to the same mesh geometry.

3. **Training on diverse mesh shapes is more important** than visual diversity for robust garment manipulation.

4. **Unseen mesh geometries remain challenging.** `Pant_Short_Unseen_0` (PS_Short130) achieves 0% across all checkpoints, and `Top_Short_Unseen_0` (TNSC_Top231) also scores 0%.

5. **TCSC_Top004 at scale 0.65** shows unusually high within-mesh variance (20%-100%), possibly due to the larger scale making the garment harder to manipulate consistently.

---

## Data Sources

- Eval reports: `outputs/eval_reports/*/heatmap_data.csv`
- Asset configs: `Assets/objects/Challenge_Garment/Release/{Top_Long,Top_Short,Pant_Long,Pant_Short}/*/*.json`
- Asset loader: `source/lehome/lehome/assets/object/Garment.py`
