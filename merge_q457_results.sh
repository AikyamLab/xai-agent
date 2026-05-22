#!/bin/bash

# Merge Q4, Q5, Q7 results from both runs
for q in 4 5 7; do
  echo "Merging Q$q results..."
  
  # Create merged results directory
  mkdir -p outputs_parallel_imdb_merged/q$q/results
  
  # Copy all successful results from first run
  for qid in {0..19}; do
    if [ -f "outputs_parallel_imdb/q$q/results/$qid/result.json" ]; then
      mkdir -p outputs_parallel_imdb_merged/q$q/results/$qid
      cp outputs_parallel_imdb/q$q/results/$qid/result.json outputs_parallel_imdb_merged/q$q/results/$qid/ 2>/dev/null || true
    fi
  done
  
  # Copy all successful results from rerun (overwrite if different)
  for qid in {0..19}; do
    if [ -f "outputs_parallel_imdb_q457/q$q/results/$qid/result.json" ]; then
      mkdir -p outputs_parallel_imdb_merged/q$q/results/$qid
      cp outputs_parallel_imdb_q457/q$q/results/$qid/result.json outputs_parallel_imdb_merged/q$q/results/$qid/
    fi
  done
done

# Count merged results
echo ""
echo "=== MERGED RESULTS COUNT ==="
for q in 4 5 7; do
  count=$(find outputs_parallel_imdb_merged/q$q/results -name "result.json" | wc -l)
  echo "Q$q: $count/20 results available"
done
