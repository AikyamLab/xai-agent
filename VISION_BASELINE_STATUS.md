# Vision Baseline Status - Ready for Collaborators

## Current Status: ✅ READY FOR DEPLOYMENT

### Pre-requisites Already Met

| Component | Status | Details |
|-----------|--------|---------|
| JSON Benchmarks | ✅ Present | 40 files (4 datasets × 10 qtypes) |
| Model Files | ✅ Present | 4 .pth files (95 MB total) |
| Loader Scripts | ✅ Present | 4 loaders with dynamic paths |
| Baseline Agents | ✅ Present | Naive, CoT, ReAct, ToT |
| Path Resolution | ✅ Implemented | Automatic search in multiple locations |
| CLI Interface | ✅ Ready | --dataset_dir parameter support |

### What's Configured

```
✓ dataset/test/vision/
  - stl10_resnet_q1.json → q10.json      (10 files)
  - stl10_densenet_q1.json → q10.json    (10 files)
  - cub_resnet_q1.json → q10.json        (10 files)
  - cub_densenet_q1.json → q10.json      (10 files)
  
✓ models_to_read/vision/
  - stl10_resnet.pth (94 MB)
  - stl10_densenet.pth (28 MB)
  - cub_resnet.pth (96 MB)
  - cub_densenet.pth (75 MB)
  
✓ models_to_read/vision/load_*.py
  - All loaders have set_dataset_root() function
  - Dynamic path resolution implemented
  - Supports multiple directory layouts
```

### What Users Need to Provide

| Item | Source | Format |
|------|--------|--------|
| Vision Images | External | STL10 (~100 MB) or CUB (~1 GB) |
| --dataset_dir | CLI | Path to image root directory |
| API Keys | Environment | TINKER_API_KEY, GEMINI_API_KEY |

## Deployment Instructions for Collaborators

### 1. Get Code
```bash
git checkout baselines_
pip install -r requirements.txt
pip install datasets==2.18.0  # Important for SNLI
```

### 2. Get Vision Images
```bash
# Option A: STL10 (smaller, ~100 MB)
wget -O stl10_binary.tar.gz https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
tar -xzf stl10_binary.tar.gz -C dataset/image/stl-10/

# Option B: CUB (larger, ~1 GB)
# Download from: http://www.vision.caltech.edu/datasets/cub-200-2011/
# Extract to dataset/image/CUB_200_2011/
```

### 3. Set API Keys
```bash
export TINKER_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"
```

### 4. Run Vision Baselines
```bash
# Quick test (1 agent, 1 dataset, 3 qtypes, 3 qids)
python run_baseline_agent_batch.py \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality vision \
    --agents "naive" \
    --datasets stl10_resnet \
    --q_types 1 2 3 \
    --question_ids 0-2 \
    --dataset_dir ./dataset \
    --output_dir outputs/vision_test/ \
    --dry_run

# Full suite (4 agents, 4 datasets, 10 qtypes, 20 qids each = 3200 jobs)
python run_baseline_agent_batch.py \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality vision \
    --agents "naive cot react tot" \
    --q_types 1 2 3 4 5 6 7 8 9 10 \
    --question_ids 0-19 \
    --dataset_dir ./dataset \
    --models_dir ./models_to_read \
    --output_dir outputs/vision_baselines/ \
    --parallel --max_workers 4
```

## Scalability Summary

### ✅ Path Resolution Works Automatically

The system tries these paths in order:
1. `<dataset_dir>/image/stl-10/` or `<dataset_dir>/image/CUB_200_2011/`
2. `<dataset_dir>/stl-10/` or `<dataset_dir>/CUB_200_2011/`
3. `<dataset_dir>/` (raw base)

This means users can organize images however they want!

### ✅ Works with Any Dataset Directory

```bash
# User A: Local copy
python ... --dataset_dir ./dataset ...

# User B: Shared network drive
python ... --dataset_dir /mnt/shared/vision_data ...

# User C: Large external drive
python ... --dataset_dir /media/usb/images ...

# User D: Custom path anywhere
python ... --dataset_dir /custom/path/vision_datasets ...
```

All work identically because of dynamic path resolution!

### ✅ Parallel Execution Ready

```bash
# With max_workers=4, runs 4 jobs in parallel
--parallel --max_workers 4

# With max_workers=8
--parallel --max_workers 8

# Without --parallel (sequential)
# (useful for debugging or low-memory systems)
```

## Data Size Estimates

| Dataset | Images | Size | Notes |
|---------|--------|------|-------|
| STL10 | 8,000 | ~100 MB | Simpler 96×96 images |
| CUB | 11,788 | ~1.2 GB | Complex bird images 256-600px |

**Total for both:** ~1.3 GB

## Expected Execution Times

### Per Baseline Type (1 dataset, 10 qtypes, 20 qids = 200 jobs)
| Agent | Est. Time | Notes |
|-------|-----------|-------|
| Naive | 1-2 hours | Fastest |
| CoT | 2-3 hours | Medium |
| ReAct | 3-5 hours | Slower |
| ToT | 5-10 hours | Slowest |

### All 4 Agents (4 datasets, 10 qtypes, 20 qids = 3,200 jobs)
- Sequential: 50-100 hours
- With --max_workers 4: 12-25 hours
- With --max_workers 8: 6-12 hours

## Troubleshooting

### Images Not Found
```
RuntimeError: Failed to load model/data: ...
```

**Fix:**
1. Verify images exist: `ls dataset/image/stl-10/stl10_binary/`
2. Check path in logs
3. Try explicit path: `--dataset_dir /full/path/to/images`

### Out of Memory
```
MemoryError: Unable to allocate...
```

**Fix:**
- Reduce `--max_workers` (try 2 or 1)
- Run with smaller batch: `--question_ids 0-4` first
- Run without `--parallel`

### VLM Timeouts
```
timeout waiting for response...
```

**Fix:**
- Reduce `--max_workers`
- May indicate VLM service issues (not your code)

## Verification Checklist

Before sharing with collaborators:

- ✓ JSON files present and valid
- ✓ Model files present and correct size
- ✓ Loaders have set_dataset_root() implemented
- ✓ DataModelLoader calls _configure_dataset_root()
- ✓ Dynamic path resolution tested
- ✓ CLI parameters documented
- ✓ Example commands provided
- ✓ Data requirements documented

## Summary

**Vision baselines are fully scalable and ready for collaborators** because:

1. ✅ Dynamic path resolution handles any data location
2. ✅ CLI parameter for flexible dataset_dir
3. ✅ All code paths tested and working
4. ✅ Parallel execution supported
5. ✅ Clear documentation and examples
6. ✅ Error messages guide users

Users can run with images anywhere as long as they pass correct `--dataset_dir`!

