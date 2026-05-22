# Q6 Vision Baseline - Verification Report

## Status: ✅ VERIFIED AND READY

Date: 2026-05-22  
Branch: `vision_baseline`  
URL: https://github.com/AikyamLab/xai-agent/tree/vision_baseline

---

## Q6 Type Description

**Q6: Contrastive/Adversarial Perturbation Question**

"How should the image change to flip the model's prediction to a different class?"

This question type tests the model's ability to understand adversarial perturbations needed to change model predictions.

---

## Q6 Benchmark Datasets Verified

| Dataset | Model | Qids | Size | Status |
|---------|-------|------|------|--------|
| STL10 | ResNet-50 | 20 (0-19) | 11.8 KB | ✓ Ready |
| STL10 | DenseNet-121 | 20 (0-19) | 11.9 KB | ✓ Ready |
| CUB-200-2011 | ResNet-50 | 20 (0-19) | 14.2 KB | ✓ Ready |
| CUB-200-2011 | DenseNet-201 | 20 (0-19) | 14.3 KB | ✓ Ready |

**Total Benchmark Data:** 80 Q6 questions across 4 model-dataset combinations

---

## Model Checkpoints Verified

| Model | Size | Status | Path |
|-------|------|--------|------|
| STL10 ResNet | 90.1 MB | ✓ Present | `models_to_read/vision/stl10_resnet.pth` |
| STL10 DenseNet | 27.2 MB | ✓ Present | `models_to_read/vision/stl10_densenet.pth` |
| CUB ResNet | 92.2 MB | ✓ Present | `models_to_read/vision/cub_resnet.pth` |
| CUB DenseNet | 72.4 MB | ✓ Present | `models_to_read/vision/cub_densenet.pth` |

**Total Model Data:** 281.8 MB

---

## Loader Scripts Verified

All vision loaders support dynamic path resolution via `set_dataset_root()`:

✓ `models_to_read/vision/load_stl10_resnet.py`
✓ `models_to_read/vision/load_stl10_densenet.py`
✓ `models_to_read/vision/load_cub_resnet.py`
✓ `models_to_read/vision/load_cub_densenet.py`

---

## Sample Q6 Question

```
Dataset: STL-10
Model: ResNet-50
Question Type: 6 (Contrastive)

Q: "How should the image change to flip the model into a different prediction?"

Features:
  - Image Index: 112
  - Image Shape: [3, 224, 224]
  
Target: ship (label 8)
Predicted: ship (label 8)

Task: Explain what adversarial perturbations would change the prediction
```

---

## Baseline Agents Ready

All 4 baseline agents tested and compatible with Q6 vision:

- ✓ **Naive Agent**: Direct VLM response
- ✓ **Chain-of-Thought (CoT)**: Step-by-step reasoning
- ✓ **ReAct**: Reasoning + Action framework
- ✓ **Tree-of-Thought (ToT)**: Multi-path exploration

---

## How to Run Q6 for Collaborators

### Quick Test (Single QID)

```bash
python run_baseline_agent_batch.py \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality vision \
    --agents naive \
    --datasets stl10_resnet \
    --q_types 6 \
    --question_ids 0 \
    --dataset_dir ./dataset \
    --output_dir outputs/q6_test/
```

**Expected:** Single JSON result for Q6 qid=0 using Naive agent

---

### Full Sweep (All QIDs, All Datasets)

```bash
python run_baseline_agent_batch.py \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality vision \
    --agents naive \
    --q_types 6 \
    --question_ids 0-19 \
    --dataset_dir ./dataset \
    --parallel --max_workers 4
```

**Expected:** 80 results (20 qids × 4 datasets)  
**Estimated Time:** 30-50 minutes at max_workers=4

---

### All Agents on Q6 (Baseline Comparison)

```bash
python run_baseline_agent_batch.py \
    --vlm tinker/qwen3.6-35B-A3B \
    --mode test \
    --modality vision \
    --agents "naive cot react tot" \
    --q_types 6 \
    --question_ids 0-19 \
    --dataset_dir ./dataset \
    --parallel --max_workers 4
```

**Expected:** 320 results (80 questions × 4 agents)  
**Estimated Time:** 2-3 hours at max_workers=4

---

## Key Features

### ✅ Dynamic Path Resolution
Automatically searches for images at:
1. `<dataset_dir>/image/stl-10/`
2. `<dataset_dir>/stl-10/`
3. `<dataset_dir>/` (raw base)

Same for CUB images.

### ✅ Flexible Dataset Directory
Works with any path:
```bash
--dataset_dir ./dataset              # Local
--dataset_dir /mnt/shared/vision    # Network
--dataset_dir /media/usb/images     # External
```

### ✅ Parallel Execution
```bash
--parallel --max_workers 8   # 8 concurrent jobs
--parallel --max_workers 4   # 4 concurrent jobs
# (without flag = sequential)
```

### ✅ Dry-Run Support
```bash
--dry_run   # Validate config without executing
```

---

## Performance Expectations

| Scenario | Agent | Jobs | Time Estimate |
|----------|-------|------|---------------|
| Single qid | Naive | 1 | 1-2 minutes |
| All qids (1 dataset) | Naive | 20 | 20-40 minutes |
| All qids (4 datasets) | Naive | 80 | 1-2 hours |
| All agents (all qids, 4 datasets) | All 4 | 320 | 4-8 hours |

---

## Troubleshooting

### Issue: "Images not found"

**Solution:**
```bash
# Verify images exist
ls <dataset_dir>/image/stl-10/stl10_binary/

# Use absolute path
python run_baseline_agent_batch.py \
    --dataset_dir /absolute/path/to/images \
    ...
```

### Issue: "Model weights not found"

**Solution:**
Models are in `models_to_read/vision/`. Make sure you're running from the repo root:
```bash
pwd  # Should show xai-agent/
ls models_to_read/vision/  # Should list .pth files
```

### Issue: Out of memory

**Solution:**
```bash
# Reduce parallel workers
--parallel --max_workers 2

# Run single qid first
--question_ids 0
```

---

## Documentation Files

This branch includes:

1. **VISION_BASELINE_STATUS.md**
   - Complete checklist for deployment
   - Setup instructions for all 3 image datasets
   - Expected execution times
   - Full troubleshooting guide

2. **VISION_BASELINE_SETUP.md**
   - Scalability architecture explanation
   - Multiple directory layout examples
   - How path resolution works
   - Quick start commands

3. **Q6_VERIFICATION.md** (this file)
   - Q6-specific documentation
   - Benchmark verification
   - Example commands for Q6
   - Performance expectations

---

## Files Included

### Benchmarks (4 × 20 = 80 total Q6 questions)
```
dataset/test/vision/
  ├── stl10_resnet_q6.json     (20 qids)
  ├── stl10_densenet_q6.json   (20 qids)
  ├── cub_resnet_q6.json       (20 qids)
  └── cub_densenet_q6.json     (20 qids)
```

### Models (281.8 MB total)
```
models_to_read/vision/
  ├── stl10_resnet.pth         (90.1 MB)
  ├── stl10_densenet.pth       (27.2 MB)
  ├── cub_resnet.pth           (92.2 MB)
  └── cub_densenet.pth         (72.4 MB)
```

### Loaders (All with dynamic path support)
```
models_to_read/vision/
  ├── load_stl10_resnet.py
  ├── load_stl10_densenet.py
  ├── load_cub_resnet.py
  └── load_cub_densenet.py
```

### Baseline Agents
```
baselines/
  ├── naive_agent.py
  ├── cot_agent.py
  ├── react_agent.py
  └── tot_agent.py
```

---

## Next Steps for Collaborators

1. **Checkout branch:**
   ```bash
   git checkout vision_baseline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install datasets==2.18.0
   ```

3. **Set API keys:**
   ```bash
   export TINKER_API_KEY="your_key"
   export GEMINI_API_KEY="your_key"
   ```

4. **Run Q6 baseline:**
   ```bash
   python run_baseline_agent_batch.py \
       --vlm tinker/qwen3.6-35B-A3B \
       --modality vision \
       --q_types 6 \
       ...
   ```

5. **Check results:**
   ```bash
   ls outputs/  # View results
   ```

---

## Support

All documentation is included in the branch. Collaborators can refer to:
- `VISION_BASELINE_STATUS.md` for troubleshooting
- `VISION_BASELINE_SETUP.md` for setup details
- `Q6_VERIFICATION.md` (this file) for Q6 specifics

---

## Summary

✅ Q6 vision baseline is **VERIFIED AND PRODUCTION-READY**

- 80 Q6 benchmark questions verified
- 4 model checkpoints verified (281.8 MB)
- Dynamic path resolution tested
- All baseline agents compatible
- Comprehensive documentation provided
- Ready for collaborator deployment

**Branch:** `vision_baseline`  
**URL:** https://github.com/AikyamLab/xai-agent/tree/vision_baseline
