# Evo-RL Task Field Investigation - Key Findings

## Problem Statement

Training Pistar06 value model with `lerobot-value-train` failed with:
```
ValueError: Corresponding feature is not valid: {'dtype': 'float32', 'shape': ()}
```

Initial hypothesis: The validation code in `get_hf_features_from_features` doesn't handle `shape: ()`.

**Correction**: The validation code IS correct - the problem was with the dataset format.

---

## Root Cause Analysis

### The Issue: `task` field in wrong locations

The dataset incorrectly had `task` in:
1. **Data frames** (parquet files) - ❌ WRONG
2. **Features dictionary** (info.json) - ❌ WRONG

### Why This Matters

The `task` field is **NOT stored in data frames** - it's added **dynamically** during dataset loading.

**Reference**: [lerobot_dataset.py:1089-1091](../third_party/Evo-RL/src/lerobot/datasets/lerobot_dataset.py)

```python
def __getitem__(self, idx) -> dict:
    # ... load data from parquet files ...

    # Add task as a string FROM meta/tasks.parquet
    task_idx = item["task_index"].item()          # Get task_index from data frame
    item["task"] = self.meta.tasks.iloc[task_idx].name  # Look up task name
    return item
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Files (parquet)                         │
│  ┌─────────────┬─────────────┬─────────────┬──────────────┐    │
│  │ observation │   action    │   ...       │  task_index  │    │
│  │    .state   │             │             │      ────────│────┼──┐
│  │  [0.1, ...] │  [0.2, ...] │   ...       │      0       │    │  │
│  └─────────────┴─────────────┴─────────────┴──────────────┘    │  │
└─────────────────────────────────────────────────────────────────┘  │
                                                                     │
                              ┌──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              meta/tasks.parquet (stored separately)              │
│  ┌─────────────────────────┬──────────────┐                     │
│  │        Task Name        │  task_index  │                     │
│  ├─────────────────────────┼──────────────┤                     │
│  │  "Fold the garment"     │      0       │ ◄─────┐             │
│  └─────────────────────────┴──────────────┘       │             │
└───────────────────────────────────────────────────┘       │
                                                              │
                                                              │
┌─────────────────────────────────────────────────────────────────┐
│                    __getitem__(idx)                             │
│                                                                  │
│   1. Load data frame → gets task_index = 0                       │
│   2. Look up: self.meta.tasks.iloc[0].name                      │
│   3. Add to item: item["task"] = "Fold the garment"              │
│                                                                  │
│   Result: item = {observation, action, ..., task_index, task}   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Correct Dataset Format

### What SHOULD be in the dataset:

| Component          | task_index | task | episode_success (in data) | episode_success (in episodes table) |
|-------------------|:----------:|:----:|:-------------------------:|:-----------------------------------:|
| Data frames       | ✅         | ❌   | ❌                        | N/A                                |
| Features dict     | ✅         | ❌   | ❌                        | N/A                                |
| episodes table    | N/A        | N/A  | N/A                       | ✅ (as "success"/"failure")         |
| meta/tasks.parquet| ✅         | ✅   | N/A                       | N/A                                |

### Comparison: Official vs Your Dataset

**Official Example Dataset** (`Datasets/example/top_long_merged/`):
```bash
# Data file columns
['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
# Note: NO 'task' column!

# info.json features
["observation.state", "action", "observation.images.*", "timestamp", "frame_index",
 "episode_index", "index", "task_index"]
# Note: NO 'task' in features!
```

**Your Original Dataset** (WRONG):
```bash
# Data file columns
[..., 'task_index', 'episode_success', 'task']  # ❌ Has 'task' and 'episode_success'

# info.json features
{
  "task": {"dtype": "string", "shape": []},          # ❌ Should not exist
  "episode_success": {"dtype": "float32", "shape": []}  # ❌ Should not exist
}
```

**Your Fixed Dataset** (CORRECT):
```bash
# Data file columns
[..., 'task_index']  # ✅ Only task_index, no 'task' or 'episode_success'

# info.json features
{
  "task_index": {"dtype": "int64", "shape": [1]}  # ✅ Only task_index
}

# episodes table
{
  "episode_index": 0,
  "episode_success": "success"  # ✅ String value in episodes table
}
```

---

## The Fix

### Changes Made

1. **Removed `task` from data frames**
   - `task` will be added dynamically based on `task_index`

2. **Removed `task` from features dictionary**
   - `task` is not a data frame feature

3. **Removed `episode_success` from data frames**
   - Moved to episodes table as per-episode metadata

4. **Removed `episode_success` from features dictionary**
   - Only exists in episodes table, not as a data frame feature

5. **Converted `episode_success` values**
   - From: `float` (0.0/1.0)
   - To: `string` ("success"/"failure")

### Fixed Dataset Location

```
/Users/moky/codes/lehome-challenge/Datasets/pant_long_evo_rl_merged_fixed_v3/
```

---

## Commands

### Sync to Remote Server

```bash
# Sync the fixed dataset to remote server
rsync -avz --progress \
  /Users/moky/codes/lehome-challenge/Datasets/pant_long_evo_rl_merged_fixed_v3/ \
  hls@192.168.3.102:/home/hls/Datasets/pant_long_evo_rl_merged_fixed_v3/

# Or replace the existing dataset
rsync -avz --progress --delete \
  /Users/moky/codes/lehome-challenge/Datasets/pant_long_evo_rl_merged_fixed_v3/ \
  hls@192.168.3.102:/home/hls/Datasets/pant_long_evo_rl_merged/
```

### Training Command

```bash
lerobot-value-train \
  --dataset.repo_id=/home/hls/Datasets/pant_long_evo_rl_merged_fixed_v3 \
  --value.type=pistar06 \
  --value.vision_repo_id=google/siglip-so400m-patch14-384 \
  --value.language_repo_id=google/gemma-3-270m \
  --value.camera_features='["observation.images.top_rgb","observation.images.left_rgb","observation.images.right_rgb"]' \
  --value.state_feature=observation.state \
  --batch_size=32 \
  --steps=8000 \
  --num_workers=4
```

### Using Config File

Create `configs/train_value_pistar06.yaml`:
```yaml
dataset:
  repo_id: /home/hls/Datasets/pant_long_evo_rl_merged_fixed_v3

value:
  type: pistar06
  vision_repo_id: google/siglip-so400m-patch14-384
  language_repo_id: google/gemma-3-270m
  camera_features:
    - observation.images.top_rgb
    - observation.images.left_rgb
    - observation.images.right_rgb
  state_feature: observation.state

batch_size: 32
steps: 8000
num_workers: 4
```

Then run:
```bash
lerobot-value-train --config_path=configs/train_value_pistar06.yaml
```

---

## Key Takeaways

1. **Dynamic fields exist**: Some fields like `task` are added dynamically during `__getitem__`, not stored in data frames
2. **Episodes vs frames**: Per-episode metadata belongs in the episodes table, not data frames
3. **Code is reference**: Always check the source code (e.g., `__getitem__`) to understand the expected format
4. **Compare with examples**: Official datasets are the best reference for correct format
5. **Validation code is correct**: The issue was almost always the dataset format, not the validation logic

---

## Common Issue: `chunks_size` Mismatch

### Problem

When merging datasets, you may encounter this error:
```
FileNotFoundError: [Errno 2] No such file or directory:
'data/chunk-000/file-001.parquet'
```

### Root Cause

The `chunks_size` in `info.json` doesn't match the actual data file structure:

| Metadata (`chunks_size`) | Actual Files | Result |
|-------------------------|--------------|--------|
| 1000 | 1 file (all frames) | ❌ Mismatch - expects 66 files, only 1 exists |
| 1000 | 66 files (1000 frames each) | ✅ Correct |
| total_frames | 1 file (all frames) | ✅ Works (single-file dataset) |

The merge function calculates expected files as:
```python
expected_files = ceil(total_frames / chunks_size)
```

If `chunks_size=1000` and `total_frames=65909`, it expects 66 files, but only `file-000.parquet` exists.

### Fix Options

**Option 1: Quick Fix (for single-file datasets)**
```python
import json

# Set chunks_size to match actual file structure
with open('dataset/meta/info.json', 'r') as f:
    info = json.load(f)
info['chunks_size'] = info['total_frames']  # Single chunk

with open('dataset/meta/info.json', 'w') as f:
    json.dump(info, f, indent=2)
```

**Option 2: Proper Re-chunking (recommended for production)**
```python
from lerobot.datasets.dataset_tools import reshape_dataset

reshape_dataset(
    repo_id="your_dataset",
    chunks_size=1000  # Standard chunk size
)
```

### When to Use Each Option

| Scenario | Recommended Approach |
|----------|---------------------|
| Quick testing/debugging | Option 1 (quick fix) |
| Production training | Option 2 (proper re-chunking) |
| Before merging datasets | Either (but be consistent) |

### Note

The standard `chunks_size: 1000` is optimal for:
- Faster data loading (load only needed chunks)
- Better caching behavior
- Parallel data loading across workers

Single-file datasets work but may have slower I/O performance during training.

---

## Files Created/Modified

- [fix_dataset_format.py](../scripts/fix_dataset_format.py) - Script to fix dataset format
- [dataset-format-fix-summary.md](dataset-format-fix-summary.md) - Detailed fix documentation

---

## Verification

To verify the dataset format is correct:

```python
import sys
sys.path.insert(0, 'third_party/Evo-RL/src')
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset('/path/to/dataset')
print(f"Features: {list(dataset.features.keys())}")
# Should NOT include 'task' or 'episode_success'

item = dataset[0]
print(f"Item keys: {list(item.keys())}")
# Should include 'task' (added dynamically) but NOT 'episode_success'
```

Expected output:
```
Features: ['observation.state', 'action', 'observation.images.*', ..., 'task_index']
Item keys: [..., 'task_index', 'task']  # 'task' added dynamically
```
