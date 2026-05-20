#!/usr/bin/env python3
"""
Baseline Agent Batch Runner

Runs the full 3-agent pipeline (xai_pipeline_v2) with injected baseline
reasoning preambles (naive / cot / react / tot) across multiple datasets,
question types, and question IDs.

How it works:
    - Reuses job-building logic from run_pipeline_batch.py
    - For each job, spawns `run_baseline_agent.py` (instead of
      `xai_pipeline_v2.py`) with BASELINE_TYPE set in the environment
    - Always passes --no-improvement and --no-sf (baselines skip these)
    - Output goes to outputs_baseline_agents/{baseline_type}/...

Usage:
    python run_baseline_agent_batch.py --baseline cot --datasets adult_tabnn \\
        --q_types 1 2 3 --question_ids 0-4 \\
        --vlm Qwen3.6-35B-A3B

    python run_baseline_agent_batch.py --baseline tot --modality tabular \\
        --q_types 1 2 3 4 5 --question_ids all \\
        --vlm Qwen3.6-35B-A3B --dry_run
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

# Reuse data structures and helpers from run_pipeline_batch
from run_pipeline_batch import (
    BASE_DIR,
    DATASET_MODEL_MAP,
    DATASET_MODALITY_MAP,
    MODALITY_DATASETS,
    Job,
    JobResult,
    AutoRange,
    parse_range,
    build_jobs,
    setup_logging,
)

VALID_BASELINES = ["naive", "cot", "react", "tot"]

# Use CWD-relative paths as defaults (overridable by CLI)
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_baseline_agents")
DEFAULT_DATASET_DIR = os.path.join(os.getcwd(), "dataset")
DEFAULT_MODELS_DIR = os.path.join(os.getcwd(), "models_to_read")


def run_single_baseline_agent_job(
    job: Job,
    baseline_type: str,
    output_dir: str,
    dataset_dir: str,
    models_dir: str,
    vlm_model: str,
    faithfulness_threshold: float,
    log_dir: Path,
    mode: str = "test",
    no_eval: bool = False,
    tinker_checkpoint: Optional[str] = None,
    tinker_lora_rank: int = 16,
) -> JobResult:
    """Execute a single baseline-agent pipeline job."""
    start_time = datetime.now()

    # Build command — calls run_baseline_agent.py instead of xai_pipeline_v2.py
    wrapper_script = Path(__file__).parent / "run_baseline_agent.py"
    cmd = [
        sys.executable,
        str(wrapper_script),
        "--dataset", job.dataset_path,
        "--question_id", str(job.question_id),
        "--model_url", job.model_path,
        "--dataset_dir", dataset_dir,
        "--models_dir", models_dir,
        "--output_dir", output_dir,
        "--vlm", vlm_model,
        "--mode", mode,
        "--faithfulness_threshold", str(faithfulness_threshold),
        # Baselines always skip improvement and strategy faithfulness
        "--no-improvement",
        "--no-sf",
    ]

    if tinker_checkpoint is not None:
        cmd.extend(["--tinker_checkpoint", tinker_checkpoint])
        cmd.extend(["--tinker_lora_rank", str(tinker_lora_rank)])
    if no_eval:
        cmd.append("--no-eval")

    # Set BASELINE_TYPE env var for the subprocess
    env = os.environ.copy()
    env["BASELINE_TYPE"] = baseline_type

    # Create log file
    log_file = log_dir / f"{baseline_type}_{job.job_id}.log"

    try:
        with open(log_file, "w") as f:
            f.write(f"Job: {job.job_id}\n")
            f.write(f"Baseline: {baseline_type}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {start_time}\n")
            f.write("=" * 80 + "\n\n")

            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour timeout
                env=env,
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Finished: {end_time}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Return code: {result.returncode}\n")

        return JobResult(
            job=job,
            success=(result.returncode == 0),
            return_code=result.returncode,
            stdout="",
            stderr="",
            duration=duration,
            log_file=str(log_file),
        )

    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        with open(log_file, "a") as f:
            f.write(f"\n\nTIMEOUT after {duration:.2f}s\n")

        return JobResult(
            job=job,
            success=False,
            return_code=-1,
            stdout="",
            stderr="TIMEOUT",
            duration=duration,
            log_file=str(log_file),
        )

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return JobResult(
            job=job,
            success=False,
            return_code=-1,
            stdout="",
            stderr=str(e),
            duration=duration,
            log_file=str(log_file),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Agent Batch Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run CoT baseline on tabular datasets, Q1-Q3
    python run_baseline_agent_batch.py --baseline cot --modality tabular \\
        --q_types 1 2 3 --question_ids 0-9

    # Run all baselines on a single dataset
    python run_baseline_agent_batch.py --baseline naive cot react tot \\
        --datasets adult_tabnn --q_types 1 --question_ids 0-4

    # Dry run to see commands
    python run_baseline_agent_batch.py --baseline tot --modality tabular \\
        --q_types 1 --question_ids 0-4 --dry_run

Available Datasets:
    Vision:  stl10_resnet, stl10_densenet, cub_resnet, cub_densenet
    Text:    imdb_cnn, imdb_2layernn, snli_cnn, snli_2layernn
    Tabular: adult_2layernn, adult_tabnn, cancer_2layernn, cancer_tabnn
        """,
    )

    # Baseline selection
    parser.add_argument("--baseline", nargs="+", required=True,
                        choices=VALID_BASELINES,
                        help="Baseline type(s) to run")

    # Input options (same as run_pipeline_batch.py)
    parser.add_argument("--modality", nargs="+", choices=["vision", "text", "tabular", "all"],
                        help="Modalities to run")
    parser.add_argument("--datasets", nargs="+",
                        help="Specific datasets to run")
    parser.add_argument("--q_types", nargs="+", type=int, default=[1],
                        help="Question types (default: 1)")
    parser.add_argument("--question_ids", type=str, default="0",
                        help="Question IDs: '0-4', 'all', '1-all' (default: 0)")

    # Configuration
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--use_test_variant", action="store_true",
                        help="Use _test variant dataset files if available")

    # Evaluation
    parser.add_argument("--faithfulness_threshold", type=float, default=0.1,
                        help="Threshold for faithfulness (default: 0.1)")

    # Output
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Dataset directory")
    parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR,
                        help="Models directory")
    parser.add_argument("--vlm", type=str, default="Qwen3.6-35B-A3B",
                        help="VLM model ID")
    parser.add_argument("--tinker_checkpoint", type=str, default=None,
                        help="Tinker LoRA/DPO checkpoint (mode=test only)")
    parser.add_argument("--tinker_lora_rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--no-eval", "--no_eval", dest="no_eval", action="store_true",
                        help="Skip evaluation in xai_pipeline_v2")

    # Execution
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Determine datasets
    datasets = []
    if args.datasets:
        datasets = args.datasets
    elif args.modality:
        for mod in args.modality:
            if mod == "all":
                for mod_datasets in MODALITY_DATASETS.values():
                    datasets.extend(mod_datasets)
            else:
                datasets.extend(MODALITY_DATASETS.get(mod, []))
    else:
        print("Error: Must specify --datasets or --modality")
        sys.exit(1)

    # Parse question IDs
    question_ids = parse_range(args.question_ids)

    # Set up per-run log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.output_dir) / "logs" / timestamp
    logger = setup_logging(log_dir, args.verbose)

    logger.info("=" * 70)
    logger.info("BASELINE AGENT BATCH RUNNER")
    logger.info("=" * 70)
    logger.info(f"  Baselines:    {args.baseline}")
    logger.info(f"  Datasets:     {datasets}")
    logger.info(f"  Q Types:      {args.q_types}")
    logger.info(f"  Question IDs: {args.question_ids}")
    logger.info(f"  Mode:         {args.mode}")
    logger.info(f"  VLM:          {args.vlm}")
    logger.info(f"  Output dir:   {args.output_dir}")
    logger.info(f"  Eval:         {'off' if args.no_eval else 'on'}")
    if args.tinker_checkpoint:
        logger.info(f"  Checkpoint:   {args.tinker_checkpoint} (rank={args.tinker_lora_rank})")

    # Build base jobs (same for all baselines)
    base_jobs = build_jobs(
        datasets=datasets,
        q_types=args.q_types,
        question_ids=question_ids,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        mode=args.mode,
        use_test_variant=args.use_test_variant,
        logger=logger,
    )

    if not base_jobs:
        logger.warning("No valid jobs to run. Check your configuration.")
        sys.exit(1)

    total_jobs = len(base_jobs) * len(args.baseline)
    logger.info(f"\nTotal jobs: {total_jobs} ({len(base_jobs)} questions × {len(args.baseline)} baselines)")

    # Dry run
    if args.dry_run:
        logger.info("\n--- Dry Run ---")
        for baseline_type in args.baseline:
            logger.info(f"\n  Baseline: {baseline_type}")
            for job in base_jobs:
                bl_output = os.path.join(args.output_dir, baseline_type)
                parts = [
                    f"BASELINE_TYPE={baseline_type}",
                    f"python run_baseline_agent.py",
                    f"--dataset {job.dataset_path}",
                    f"--question_id {job.question_id}",
                    f"--model_url {job.model_path}",
                    f"--vlm {args.vlm}",
                    f"--output_dir {bl_output}",
                    f"--no-improvement --no-sf",
                ]
                if args.tinker_checkpoint:
                    parts.append(f"--tinker_checkpoint {args.tinker_checkpoint}")
                if args.no_eval:
                    parts.append("--no-eval")
                logger.info(f"    {job.job_id}: {' '.join(parts)}")
        sys.exit(0)

    # Execute
    all_results = []
    for baseline_type in args.baseline:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running baseline: {baseline_type.upper()}")
        logger.info(f"{'='*60}")

        # Each baseline gets its own output subdirectory
        bl_output = os.path.join(args.output_dir, baseline_type)
        os.makedirs(bl_output, exist_ok=True)

        for i, job in enumerate(base_jobs):
            logger.info(f"\n  [{baseline_type}] Job {i+1}/{len(base_jobs)}: {job.job_id}")

            result = run_single_baseline_agent_job(
                job=job,
                baseline_type=baseline_type,
                output_dir=bl_output,
                dataset_dir=args.dataset_dir,
                models_dir=args.models_dir,
                vlm_model=args.vlm,
                faithfulness_threshold=args.faithfulness_threshold,
                log_dir=log_dir,
                mode=args.mode,
                no_eval=args.no_eval,
                tinker_checkpoint=args.tinker_checkpoint,
                tinker_lora_rank=args.tinker_lora_rank,
            )

            status = "✓" if result.success else "✗"
            logger.info(f"  [{baseline_type}] {status} {job.job_id} ({result.duration:.1f}s)")
            all_results.append((baseline_type, result))

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("EXECUTION SUMMARY")
    logger.info(f"{'='*70}")

    for baseline_type in args.baseline:
        bl_results = [r for bl, r in all_results if bl == baseline_type]
        ok = sum(1 for r in bl_results if r.success)
        fail = sum(1 for r in bl_results if not r.success)
        total_dur = sum(r.duration for r in bl_results)
        logger.info(f"  {baseline_type:>6}: {ok} passed, {fail} failed, {total_dur:.1f}s total")

        if fail > 0:
            for r in bl_results:
                if not r.success:
                    logger.info(f"          FAILED: {r.job.job_id} → {r.log_file}")

    total_ok = sum(1 for _, r in all_results if r.success)
    total_fail = sum(1 for _, r in all_results if not r.success)
    logger.info(f"\n  TOTAL: {total_ok} passed, {total_fail} failed")
    logger.info(f"  Logs:  {log_dir}")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
