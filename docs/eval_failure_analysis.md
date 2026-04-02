# Evaluation Failure Analysis — All 4 Garment Categories

**Generated:** 2026-04-01
**Source reports:**
- `outputs/eval_reports/smolvla_moe_expert_pant_long_no_st_proj_20260318_095051/`
- `outputs/eval_reports/smolvla_moe_expert_pant_short_no_st_proj_20260318_095607/`
- `outputs/eval_reports/smolvla_moe_expert_top_short_no_st_proj_20260319_093958/`
- `outputs/eval_reports/smolvla_moe_expert_top_long_no_st_proj_20260319_094012/`
- `outputs/smolvla_moe_expert_pant_long_0329_20260329_092020/`
- `outputs/eval_reports/smolvla_moe_expert_pant_long_0329_20260331_182039/`

---

## Complete Garment Table

### PANT_LONG (12 unique meshes, 0 shared)

| Garment | Mesh | Visual | Scale | Best% | Verdict |
|---------|------|--------|-------|-------|---------|
| Seen_0 | PL_019 | Tiles054+Fabric079 | 0.37 | 60% | hard mesh |
| Seen_1 | PL_022 | Tiles051+Fabric079 | 0.37 | 60% | ok |
| Seen_2 | PL_030 | Tiles118 | 0.37 | 80% | ok |
| **Seen_3** | **PL_046** | **PavingStones058+Fabric079** | **0.37** | **40%** | **hard mesh** |
| Seen_4 | PL_053 | Tiles064+Fabric045 | 0.37 | 100% | easy |
| Seen_5 | PL_079 | ChristmasTreeOrnament006 | 0.37 | 100% | easy |
| Seen_6 | PL_Pants095 | ChristmasTreeOrnament004 | 0.37 | 80% | ok |
| Seen_7 | PL_Pants100 | ChristmasTreeOrnament001 | 0.37 | 60% | hard mesh |
| Seen_8 | PL_Pants106 | ChristmasTreeOrnament003 | 0.37 | 80% | ok |
| Seen_9 | PL_Pants107 | PaintedPlaster005 | 0.37 | 60% | ok |
| **Unseen_0** | **PL_Pants109** | **PaintedPlaster002** | **0.37** | **20%** | **mesh + visual** |
| **Unseen_1** | **PL_Pants121** | **PaintedPlaster003+PaintedPlaster003** | **0.37** | **20%** | **mesh + visual** |

```
Seen avg:  62%    Unseen avg:  20%    Gap: 42%

All 12 garments have UNIQUE meshes (no sharing).
Unseen_0/1 use novel PaintedPlaster materials never seen in training.
```

---

### PANT_SHORT (5 unique meshes, heavy sharing)

| Garment | Mesh | Visual | Scale | Best% | Verdict |
|---------|------|--------|-------|-------|---------|
| Seen_0 | PS_049 | Fabric071+Fabric022 | 0.45 | 100% | ok |
| Seen_1 | PS_049 | SolarPanel004+Fabric022 | 0.45 | 100% | ok |
| Seen_2 | PS_049 | Tiles131+Fabric022 | 0.45 | 100% | ok |
| Seen_3 | PS_050 | Carpet010+Fabric022 | 0.45 | 100% | ok |
| Seen_4 | PS_050 | Candy001+PaintedPlaster003 | 0.45 | 100% | ok |
| Seen_5 | PS_050 | ChristmasTreeOrnament004+PaintedPlaster005 | 0.45 | 100% | ok |
| Seen_6 | PS_M1_089 | Carpet009 | 0.45 | 80% | ok |
| Seen_7 | PS_M1_089 | Candy002 | 0.45 | 100% | ok |
| Seen_8 | PS_Short047 | Leather035A | 0.45 | 100% | ok |
| Seen_9 | PS_Short047 | GlazedTerracotta001 | 0.45 | 100% | ok |
| **Unseen_0** | **PS_Short130** | **Leather020** | **0.45** | **0%** | **MESH (unique, 0% all ckpts)** |
| Unseen_1 | PS_M1_089 | PaintedPlaster002 | 0.45 | 100% | ok (shared mesh) |

```
Seen avg:  98%    Unseen avg:  50%    Gap: 48%

Only 5 unique mesh shapes:
  PS_049      → Seen_0, Seen_1, Seen_2 (all 100%)
  PS_050      → Seen_3, Seen_4, Seen_5 (all 100%)
  PS_M1_089   → Seen_6, Seen_7, Unseen_1 (80-100%)
  PS_Short047 → Seen_8, Seen_9 (all 100%)
  PS_Short130 → Unseen_0 ONLY (0% across ALL 20 checkpoints → pure mesh failure)
```

---

### TOP_LONG (11 unique meshes, 1 shared pair)

| Garment | Mesh | Visual | Scale | Best% | Verdict |
|---------|------|--------|-------|-------|---------|
| Seen_0 | TCLC_002 | (baked-in) | 0.45 | 80% | ok |
| Seen_1 | TCLC_015 | Fabric055 | 0.45 | 100% | easy |
| Seen_2 | TCLC_028 | Carpet015+Fabric022 | 0.45 | 60% | hard mesh |
| Seen_3 | TCLO_001 | Fabric004+Fabric079 | 0.40 | 80% | ok |
| Seen_4 | TCLO_073 | Fabric082B+Fabric045 | 0.45 | 80% | ok |
| Seen_5 | TCLO_027 | Fabric040+Fabric022 | 0.45 | 80% | ok |
| Seen_6 | TNLC_010 | Fabric076 | 0.45 | 100% | easy |
| **Seen_7** | **TNLC_033** | **Fabric054+Fabric034** | **0.45** | **40%** | **hard mesh** |
| Seen_8 | TNLO_008 | Fabric004 | 0.45 | 100% | easy |
| Seen_9 | TNLO_034 | Fabric027+Fabric034 | 0.45 | 100% | easy |
| Unseen_0 | TCLC_018 | Fabric018+Fabric045 | 0.45 | 100% | ok (similar mesh) |
| **Unseen_1** | **TCLC_015** | **Fabric082B** | **0.45** | **20%** | **VISUAL (same mesh as Seen_1!)** |

```
Seen avg:  82%    Unseen avg:  60%    Gap: 22%

11 unique meshes, 1 shared pair:
  TCLC_015 → Seen_1 (100%) + Unseen_1 (20%)
              SAME MESH, different visual → Fabric055 vs Fabric082B
              → PURE VISUAL FAILURE

  Unseen_0 (100%) uses TCLC_018, similar to other training meshes → works fine
```

---

### TOP_SHORT (5 unique meshes, heavy sharing)

| Garment | Mesh | Visual | Scale | Best% | Verdict |
|---------|------|--------|-------|-------|---------|
| Seen_0 | TCSC_067 | Fabric050 | 0.45 | 80% | ok |
| Seen_1 | TCSC_067 | Fabric057 | 0.45 | 80% | ok |
| Seen_2 | TCSC_067 | Tiles131 | 0.45 | 60% | ok |
| **Seen_3** | **TCSC_Top004_1** | **Fabric022 x3+Fabric082B** | **0.65** | **20%** | **hard mesh** |
| Seen_4 | TCSC_Top004_1 | Fabric045 x3+Fabric014 | 0.65 | 100% | easy |
| Seen_5 | TCSO_Baggy_Shirt | Fabric053+Fabric045 | 0.45 | 60% | ok |
| Seen_6 | TCSO_Baggy_Shirt | Tiles131+Fabric022 | 0.45 | 60% | ok |
| Seen_7 | TCSO_Baggy_Shirt | Fabric076+Fabric022 | 0.45 | 80% | ok |
| Seen_8 | TNSC_Tshirt3 | Fabric072 | 0.45 | 20% | hard mesh |
| **Seen_9** | **TNSC_Tshirt3** | **Tiles079** | **0.45** | **0%** | **VISUAL (same mesh as Seen_8!)** |
| **Unseen_0** | **TNSC_Top231** | **Tiles081+Fabric079** | **0.45** | **0%** | **MESH (unique, 0% all ckpts)** |
| Unseen_1 | TCSC_Top004_1 | Carpet006 x3+Candy003 | 0.65 | 60% | ok (shared mesh) |

```
Seen avg:  56%    Unseen avg:  30%    Gap: 26%

Only 5 unique mesh shapes:
  TCSC_067        → Seen_0, Seen_1, Seen_2 (60-80%)
  TCSC_Top004_1   → Seen_3 (20%), Seen_4 (100%), Unseen_1 (60%)
                     Same mesh: Seen_4=100% vs Seen_3=20% → mesh+visual interaction
  TCSO_Baggy_Shirt → Seen_5, Seen_6, Seen_7 (60-80%)
  TNSC_Tshirt3    → Seen_8 (20%), Seen_9 (0%)
                      SAME MESH, same checkpoints → Fabric072 vs Tiles079
                      → PURE VISUAL FAILURE (Seen_8=20% vs Seen_9=0%)
  TNSC_Top231     → Unseen_0 ONLY (0% across ALL checkpoints → pure mesh failure)
```

---

## Shared Mesh Comparison (Visual vs Mesh Evidence)

These garments share the EXACT SAME mesh geometry. Any SR difference is purely visual:

| Mesh | Garment A | A Visual | A SR | Garment B | B Visual | B SR | Δ | Cause |
|------|-----------|----------|------|-----------|----------|------|---|-------|
| TCLC_015 | Top_Long_Seen_1 | Fabric055 | 100% | Top_Long_Unseen_1 | Fabric082B | 20% | **80%** | **VISUAL** |
| TNSC_Tshirt3 | Top_Short_Seen_8 | Fabric072 | 20% | Top_Short_Seen_9 | Tiles079 | 0% | **20%** | **VISUAL** |
| PS_M1_089 | Pant_Short_Seen_7 | Candy002 | 100% | Pant_Short_Unseen_1 | PaintedPlaster002 | 100% | 0% | ok |
| TCSC_067 | Top_Short_Seen_0 | Fabric050 | 80% | Top_Short_Seen_2 | Tiles131 | 60% | 20% | marginal |
| TCSC_Top004_1 | Top_Short_Seen_4 | Fabric045 | 100% | Top_Short_Seen_3 | Fabric022 | 20% | **80%** | **mesh+visual** |

> **Key finding**: `Top_Long_Seen_1` vs `Top_Long_Unseen_1` share TCLC_015 mesh. Fabric055→100%, Fabric082B→20%. **80% gap from visual alone.**

> **Key finding**: `Top_Short_Seen_8` vs `Top_Short_Seen_9` share TNSC_Tshirt3 mesh, same checkpoints/scale. Fabric072→20%, Tiles079→0%. **Visual alone turns 20% into 0%.**

---

## Failure Verdict Summary

### Unseen Garments (8 total, the ones that matter for generalization)

| Garment | Best% | Verdict | Evidence |
|---------|-------|---------|----------|
| Pant_Long_Unseen_0 | 20% | mesh + visual | unique mesh + novel PaintedPlaster |
| Pant_Long_Unseen_1 | 20% | mesh + visual | unique mesh + novel PaintedPlaster |
| **Pant_Short_Unseen_0** | **0%** | **MESH** | unique mesh (PS_Short130), 0% across all 20 ckpts |
| Pant_Short_Unseen_1 | 100% | ok | shared mesh (PS_M1_089), known mesh |
| Top_Long_Unseen_0 | 100% | ok | unique but similar mesh |
| **Top_Long_Unseen_1** | **20%** | **VISUAL** | same mesh as Seen_1 (TCLC_015), only visual differs |
| **Top_Short_Unseen_0** | **0%** | **MESH** | unique mesh (TNSC_Top231), 0% across all 12 ckpts |
| Top_Short_Unseen_1 | 60% | ok | shared mesh (TCSC_Top004_1), known mesh |

```
  MESH failure:   ██  2/8  (25%)  ← Pant_Short_Unseen_0, Top_Short_Unseen_0
  VISUAL failure:  █  1/8  (12%)  ← Top_Long_Unseen_1
  MESH+VISUAL:     ██  2/8  (25%)  ← Pant_Long_Unseen_0/1
  No failure:      ███ 3/8  (38%)  ← working fine
```

### Hard Seen Garments (fail despite being in training)

| Garment | Best% | Verdict | Evidence |
|---------|-------|---------|----------|
| Top_Short_Seen_9 | 0% | **VISUAL** | same mesh as Seen_8 (TNSC_Tshirt3), only Tiles079 differs |
| Pant_Long_Seen_3 | 40% | hard mesh | unique mesh PL_046, complex checkpoints |
| Top_Short_Seen_3 | 20% | mesh+visual | shared mesh TCSC_Top004_1, but Fabric022 pattern hard |
| Top_Long_Seen_7 | 40% | hard mesh | unique mesh TNLC_033 |

---

## Competition Implications

Rules: 10 seen + 2 unseen + 8 hidden = 20 garments per category.

### Strategy Priority

1. **Visual augmentation** (material swapping) — Proven to cause up to 80% SR difference on same mesh. Can fix Top_Long_Unseen_1 type failures and Seen_9=0% type failures.

2. **More diverse mesh training** — Critical for the 0% mesh failures (Pant_Short_Unseen_0, Top_Short_Unseen_0). No visual fix possible.

3. **Physics randomization** — Could help bridge mesh difficulty gap by forcing policy to learn robust manipulation.

### Garment Asset Structure

```
Each garment = mesh geometry (.usd) + visual material (Color_Texture/) + shared physics (YAML)

  44 materials in shared Color_Texture/ library
  4 garment categories x 12 garments = 48 total garments
  All share identical physics via particle_garment_cfg.yaml

  Mesh counts per category:
    Pant_Long:  12 unique (0 shared) — most mesh-diverse
    Top_Long:   11 unique (1 pair shared)
    Pant_Short:  5 unique (7 share) — least mesh-diverse
    Top_Short:   5 unique (7 share) — least mesh-diverse
```
