# Dataset Format Fix Summary

## Root Cause Analysis

The original dataset format was incompatible with Evo-RL's `lerobot-value-train` due to **two critical issues**:

### Issue 1: `task` field in wrong location
- **Problem**: The dataset had `task` in both:
  - Data frames (parquet files)
  - Features dictionary (info.json)

- **Expected**: The `task` field should **NOT** be in data frames at all!
  - It's added **dynamically** during `__getitem__` based on `task_index`
  - Official LeRobot datasets do NOT have `task` in features

- **Reference**: [lerobot_dataset.py:1089-1091](third_party/Evo-RL/src/lerobot/datasets/lerobot_dataset.py)
  ```python
  # Add task as a string
  task_idx = item["task_index"].item()
  item["task"] = self.meta.tasks.iloc[task_idx].name
  ```

### Issue 2: `episode_success` in wrong location
- **Problem**: The dataset had `episode_success` in:
  - Data frames (per frame)
  - Features dictionary

- **Expected**: `episode_success` should be:
  - Only in the **episodes table** (per episode metadata)
  - As string values: "success" or "failure"
  - NOT in data frames
  - NOT in features dictionary

## Verification

Compare with official example dataset:

```bash
# Example dataset does NOT have 'task' in data frames
$ Datasets/example/top_long_merged/data/chunk-000/file-000.parquet
Columns: ['observation.state', 'action', 'timestamp', 'frame_index',
          'episode_index', 'index', 'task_index']
# Note: 'task' is NOT in columns!

# Example dataset does NOT have 'task' in features
$ grep -A 2 '"task"' Datasets/example/top_long_merged/meta/info.json
# No results - 'task' is not in features!
```

## Fixed Dataset Location

```
/Users/moky/codes/lehome-challenge/Datasets/pant_long_evo_rl_merged_fixed_v3/
```

## Changes Made

1. **Moved `episode_success` from data frames to episodes table**
   - Converted from float (0.0/1.0) to string ("success"/"failure")
   - Placed in episodes table as per-episode metadata

2. **Removed `task` from data frames**
   - `task` will be added dynamically during loading
   - Based on `task_index` and `meta/tasks.parquet`

3. **Removed `task` from features dictionary**
   - `task` is not a data frame feature
   - Only `task_index` should be in features

4. **Removed `episode_success` from features dictionary**
   - `episode_success` is only in episodes table
   - Not a per-frame feature

## Deployment to Remote Server

To use this dataset for training on `hls@192.168.3.102`:

```bash
# Option 1: Copy fixed dataset
scp -r /Users/moky/codes/lehome-challenge/Datasets/pant_long_evo_rl_merged_fixed_v3 \
    hls@192.168.3.102:/home/hls/Datasets/pant_long_evo_rl_merged

# Option 2: Run fix script on remote server
# First, copy the script and run it on the original dataset
scp scripts/fix_dataset_format.py hls@192.168.3.102:/home/hls/
ssh hls@192.168.3.102
cd /home/hls
python fix_dataset_format.py \
    --dataset_root /home/hls/Datasets/pant_long_evo_rl_merged \
    --output_root /home/hls/Datasets/pant_long_evo_rl_merged_fixed
```

## Training Command

```bash
lerobot-value-train --dataset.repo_id=/home/hls/Datasets/pant_long_evo_rl_merged_fixed_v3
```

## Key Takeaways

1. **Dynamic fields**: Some fields like `task` are added dynamically during dataset loading, not stored in data frames
2. **Episodes vs frames**: Per-episode metadata (like `episode_success`) belongs in the episodes table, not data frames
3. **Validation**: Always compare against official datasets to understand the expected format
4. **Code is reference**: The Evo-RL recording and loading code is the source of truth for format expectations

## Files Modified

- [fix_dataset_format.py](scripts/fix_dataset_format.py) - Updated to remove both `task` and `episode_success` from data frames and features
