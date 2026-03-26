# Evaluation.py Dataset Format Fix

## Problem Statement

The `scripts/utils/evaluation.py` file was creating datasets with an incorrect format that doesn't match LeRobot standards. This could cause issues when training value models or using datasets generated from evaluation runs.

## Root Cause Analysis

### Issues Found

ied in `evaluation.py`:

| Issue | Location | Description |
|-------|----------|-------------|
| ❌ **episode_success in features** | Lines 58-63, 89-93 | `episode_success` was added to features dictionary (should NOT be there) |
| ❌ **episode_success in data frames** | Lines 195, 213-218 | `episode_success` was added to each frame (should be in episodes table only) |
| ✅ **task in frames** | Line 193 | `task` IS required in frames during recording (for validation) |

### Understanding LeRobot Dataset Format

**Key insight**: There's a distinction between what's in memory during recording vs what's stored on disk:

| Component | In Memory (Recording) | On Disk (Stored) | In Features Dict |
|-----------|----------------------|------------------|------------------|
| `task` | ✅ Required (for validate_frame) | ❌ Not stored (task_index is used) | ❌ Not in features |
| `task_index` | ✅ Auto-added (DEFAULT_FEATURES) | ✅ Stored in data files | ✅ In features |
| `episode_success` | ❌ Not in frames | ❌ Not in data files | ❌ Not in features |
| `episode_success` | - | ✅ In episodes table only | ❌ Not in features |

Reference: [`lerobot/datasets/lerobot_dataset.py:1150`](../.venv/lib/python3.11/site-packages/lerobot/datasets/lerobot_dataset.py)
```python
# In add_frame(), LeRobot pops "task" from frame and stores it separately
self.episode_buffer["task"].append(frame.pop("task"))
```

## Fixes Applied

### Approach: Simplified + Post-Processing

The user directed a simplified approach:
1. **evaluation.py**: Generate clean datasets matching `pant_long_merged` format (no episode_success)
2. **add_episode_success.py**: Separate script to add episode_success column to episodes table

This avoids complex wrapper classes and works with standard lerobot 0.4.3.

### Fix 1: Remove `episode_success` from Features Dictionary

**Before** (Lines 57-64):
```python
# Ensure episode_success field is present for Evo-RL
if "episode_success" not in features:
    features["episode_success"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
# Note: task is a special field handled by LeRobot, not in features
```

**After**:
```python
# Note: episode_success is NOT in features dict - it's only in episodes table
# Note: task is added dynamically by LeRobot based on task_index, not in features
```

### Fix 2: Correct Frame Recording

**Before** (Lines 188-196):
```python
frame = {
    k: v
    for k, v in observation_dict.items()
    if k != "observation.top_depth"
}
frame["task"] = args.task_description
# Add episode_success placeholder (will be updated at episode end)
frame["episode_success"] = np.array([0.0], dtype=np.float32)
eval_dataset.add_frame(frame)
```

**After** (Lines 220-239):
```python
frame = {
    k: v
    for k, v in observation_dict.items()
    if k != "observation.top_depth"
}
# Add task field (required by validate_frame, but not stored in data files)
frame["task"] = args.task_description

# Determine which dataset(s) to record to based on save_mode
save_mode = args.save_mode
if save_mode == "both":
    # In "both" mode, we record to both datasets during the episode
    # NOTE: LeRobot's add_frame() pops "task" from the frame, so we need to re-add it
    eval_dataset_success.add_frame(frame)
    frame["task"] = args.task_description  # Re-add after pop
    eval_dataset_failure.add_frame(frame)
else:
    # For other modes, record to the single dataset
    eval_dataset.add_frame(frame)
```

### Fix 3: Task Field Re-Add in "both" Mode (2026-03-21)

**Problem**: When using `--save_mode both`, the second `add_frame()` call failed with:
```
ValueError: Feature mismatch in `frame` dictionary: Missing features: {'task'}
```

**Root Cause**: LeRobot's `add_frame()` method calls `frame.pop("task")` on line 1150 of `lerobot_dataset.py`. After the first dataset's `add_frame()` call, the task field is removed from the frame dict, causing the second call to fail.

**Solution**: Re-add the task field after the first `add_frame()` call:
```python
if save_mode == "both":
    eval_dataset_success.add_frame(frame)
    frame["task"] = args.task_description  # Re-add after pop
    eval_dataset_failure.add_frame(frame)
```

### Fix 4: Save Mode Implementation

Added `--save_mode` argument to `scripts/utils/parser.py`:
```python
parser.add_argument(
    "--save_mode",
    type=str,
    choices=["success", "failure", "both", "all"],
    default="success",
    help="Dataset save mode: 'success' (only successful episodes), 'failure' (only failed episodes), 'both' (separate datasets for success and failure), 'all' (all episodes in one dataset)",
)
```

**Implementation** (Lines 290-320 in evaluation.py):
- `success`: Only save successful episodes, discard failures
- `failure`: Only save failed episodes, discard successes
- `both`: Record to both datasets during episode, save only the appropriate one at end
- `all`: Save all episodes to single dataset (default behavior)

### Fix 5: Separate Script for Adding episode_success

Created [`scripts/add_episode_success.py`](../scripts/add_episode_success.py) to post-process datasets:

```python
# Usage examples:
python scripts/add_episode_success.py --dataset_root /path/to/dataset --all_success
python scripts/add_episode_success.py --dataset_root /path/to/dataset --all_failure
python scripts/add_episode_success.py --dataset_root /path/to/dataset --success_episodes 0-199,201-249
```

This script:
1. Reads existing episodes.parquet files
2. Adds episode_success column with "success" or "failure" values
3. Reorders columns to put episode_success after length
4. Updates in-place or creates copy with "_with_success" suffix

## Verification

### Expected Dataset Format

After the fix, datasets created by evaluation.py should match the format of `Datasets/example/pant_long_merged`:

| Component | Expected Content |
|-----------|------------------|
| **Data frame columns** | `observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` |
| **Features dict** | NO `task`, NO `episode_success` |
| **Episodes table** | No episode_success by default (use add_episode_success.py to add) |
| **tasks.parquet** | HAS `task_index` column |

### Verification Command

```python
import sys
sys.path.insert(0, 'third_party/Evo-RL/src')
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

dataset = LeRobotDataset('lehome_eval', root=Path('path/to/eval/dataset'))
print(f"Features: {list(dataset.features.keys())}")
# Should NOT include 'task' or 'episode_success'

item = dataset[0]
print(f"Item keys: {list(item.keys())}")
# Should include 'task' (added dynamically) but NOT 'episode_success'

# Check episodes table
episodes_df = dataset.episodes
print(f"Episodes has episode_success: {'episode_success' in episodes_df.columns}")
# Should be False (use add_episode_success.py to add)
```

## Summary of Changes

1. **Removed `episode_success` from features dictionary** - It's not a data frame feature
2. **Kept `task` in frames during recording** - Required by `validate_frame()`, but not stored in data files
3. **Removed `episode_success` from data frames** - Use separate script to add to episodes table
4. **Added `--save_mode` argument** - Control which episodes to save (success/failure/both/all)
5. **Fixed task field re-add in "both" mode** - LeRobot pops task, so we re-add between calls
6. **Created separate `add_episode_success.py` script** - Post-process datasets for Pistar06 training

## Files Modified

- [`scripts/utils/evaluation.py`](../scripts/utils/evaluation.py)
  - Removed `episode_success` from features dict
  - Corrected frame recording logic
  - Added `--save_mode` support with separate dataset creation for "both" mode
  - Fixed task field re-add after first `add_frame()` call in "both" mode (Line 234)

- [`scripts/utils/parser.py`](../scripts/utils/parser.py)
  - Added `--save_mode` argument with choices: ["success", "failure", "both", "all"]

- [`scripts/add_episode_success.py`](../scripts/add_episode_success.py) (NEW)
  - Standalone script to add episode_success column to episodes table
  - Supports all_success, all_failure, or specific episode indices

## Related Documents

- [`⚡️evo-rl-task-field-investigation.md`](./⚡️evo-rl-task-field-investigation.md) - Original investigation of dataset format issues

---

**Date**: 2026-03-21
**Status**: ✅ Fixed with simplified approach + post-processing script
