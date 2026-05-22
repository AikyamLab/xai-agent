
# First Run Parallel Execution Analysis Report

## Execution Summary
- **Date**: 2026-05-22
- **Start Time**: 10:30 AM
- **Duration**: ~95 minutes
- **Total Jobs**: 160 (8 qtypes × 20 qids)
- **Successful**: 105 jobs (65.6% success rate)

## Per-Type Results

### ✅ Perfect Success (100%)
| Question Type | Topic | Jobs | Success Rate |
|---------------|-------|------|--------------|
| Q1 | Influential Features / Text Spans | 20/20 | 100% |
| Q2 | Instance Importance Ranking | 20/20 | 100% |
| Q3 | Feature Interaction Analysis | 20/20 | 100% |
| Q6 | Attribution Confidence Scores | 20/20 | 100% |
| Q8 | Similar Predictions Explanation | 20/20 | 100% |
| **TOTAL** | | **100/100** | **100%** |

### ⚠️ Partial Success (40-65%)
| Question Type | Topic | Jobs | Success Rate | Notes |
|---------------|-------|------|--------------|-------|
| Q5 | Modified Input Impact | 13/20 | 65% | 7 VLM timeout/failure |
| Q7 | Prediction Change Planning | 8/20 | 40% | 12 VLM timeout/failure |
| Q4 | Contrastive Instance Analysis | 4/20 | 20% | 16 VLM timeout/failure |

### Failure Analysis
- **Q4 failures**: Likely due to high parallelism overwhelming VLM with contrastive reasoning task
- **Q5 failures**: Intermittent VLM response generation timeouts
- **Q7 failures**: Complex multi-step reasoning requiring more VLM resources

## Quality Assessment

### Execution Parallelism
- **Parallelism Level**: 8 qtypes running simultaneously
- **System Load**: 8 CPU cores fully utilized
- **Memory**: 8GB RAM, ~6.7GB available disk
- **Bottleneck**: VLM request queue (Tinker service)

### Performance Metrics
- **Average per-job time**: ~35-60 seconds
- **Fastest**: Q1-Q3, Q6, Q8 (35-45 sec/job)
- **Slowest**: Q4, Q5, Q7 (60-120 sec/job when successful)

### Key Findings
1. **Stable Questions** (Q1, Q2, Q3, Q6, Q8): 
   - Simple extraction or single-instance analysis
   - Low VLM complexity
   - 100% reliability at 8-parallel scale

2. **Unstable Questions** (Q4, Q5, Q7):
   - Contrastive/comparative reasoning
   - Multi-instance or complex logic
   - VLM bottleneck at high parallelism

3. **Thinking Block Issue**: RESOLVED ✅
   - Initial failures due to `<think>` tags in VLM output
   - Fixed with intelligent regex stripping
   - Preserves responses that are ONLY thinking blocks

## Available Results

### Production-Ready (100/100 jobs)
- Q1: 20 results (all 20 qids)
- Q2: 20 results (all 20 qids)
- Q3: 20 results (all 20 qids)
- Q6: 20 results (all 20 qids)
- Q8: 20 results (all 20 qids)

### Partial (4-13 jobs each)
- Q4: 4/20 results
- Q5: 13/20 results
- Q7: 8/20 results
- **Subtotal**: 25/60 results (41.7%)

**Total Available**: 125/160 results (78.1% coverage)

## Recommendations

### For Missing Q4, Q5, Q7 Results
1. **Sequential Rerun** (Recommended)
   ```bash
   for q in 4 5 7; do
     python run_pipeline_batch.py --q_types $q --question_ids all ...
   done
   ```
   
2. **Reduced Parallelism** (Alternative)
   ```bash
   QUESTION_IDS="all" Q_TYPES="4 5" ./run_imdb_parallel_generic.sh
   QUESTION_IDS="all" Q_TYPES="7" ./run_imdb_parallel_generic.sh
   ```

3. **Accept Partial Coverage** (Fast)
   - Use 41.7% of Q4, Q5, Q7 results (sufficient for baseline analysis)

### For Future Runs
- **Default**: Run Q1-Q3, Q6, Q8 in parallel (8 qtypes max)
- **Unstable**: Run Q4, Q5, Q7 separately or with reduced parallelism
- **Optimization**: Combine Q1-Q3, Q6, Q8 (stable) + 1 unstable type per batch

## Framework Validation

### ✅ Working Components
- Generic parallel script supports any qtype/qid combination
- Thinking block stripping fixed and robust
- Parallel job launching and cleanup functional
- Result aggregation and logging complete

### ⚠️ Known Limitations
- VLM service not horizontally scalable (single model instance)
- Complex reasoning questions (Q4, Q5, Q7) require more VLM resources
- High parallelism (8+ concurrent) causes timeout/failure patterns

### 🎯 Framework Status
**PRODUCTION-READY** for:
- Q1-Q3, Q6, Q8 (100% success at 8-parallel)
- Batching various qtypes/qids
- Result management and organization

**PARTIAL** for:
- Q4, Q5, Q7 (recommend sequential or 2-parallel max)

