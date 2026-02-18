#!/usr/bin/env python3
"""
XAI Pipeline Batch Runner

Runs xai_pipeline_v2.py across multiple modalities, question types, and question IDs.

Usage:
    python run_pipeline_batch.py --help
    python run_pipeline_batch.py --datasets stl10_resnet --q_types 1 2 3 4 --question_ids 0 1 2
    python run_pipeline_batch.py --config batch_config.json
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path("/standard/AikyamLab/yuyang/xai_agent/framework/trial_2")
DEFAULT_CONFIG = {
    "dataset_dir": str(BASE_DIR / "dataset"),
    "models_dir": str(BASE_DIR / "models_to_read"),
    "output_dir": str(BASE_DIR / "outputs"),
    "vlm_model": "Qwen/Qwen3-VL-8B-Instruct",
    "faithfulness_threshold": 0.1,
}

# Dataset to model mappings
DATASET_MODEL_MAP = {
    # Vision (all stl10 resnet variants use stl10_resnet_head.pth)
    "stl10_resnet": "vision/stl10_resnet.pth",
    "stl10_densenet": "vision/stl10_densenet.pth",
    "cub_resnet": "vision/cub_resnet.pth",
    "cub_densenet": "vision/cub_densenet.pth",
    # Text
    "imdb_cnn": "text/imdb_cnn.pth",
    "imdb_2layernn": "text/imdb_2layernn.pth",
    "snli_cnn": "text/snli_cnn.pth",
    "snli_2layernn": "text/snli_2layernn.pth",
    # Tabular
    "adult_census": "tabular/adult_census.pth",
    "adult_tabnn": "tabular/adult_tabnn.pth",
    "adult_2layernn": "tabular/adult_2layernn.pth",
    "cancer_2nn": "tabular/cancer_2nn.pth",
    "cancer_tabnn": "tabular/cancer_tabnn.pth",
    "cancer_2layernn": "tabular/cancer_2layernn.pth",
}

# Dataset to modality mapping
DATASET_MODALITY_MAP = {
    "stl10_resnet": "vision",
    "stl10_densenet": "vision",
    "cub_resnet": "vision",
    "cub_densenet": "vision",
    "imdb_cnn": "text",
    "imdb_2layernn": "text",
    "snli_cnn": "text",
    "snli_2layernn": "text",
    "adult_census": "tabular",
    "adult_tabnn": "tabular",
    "adult_2layernn": "tabular",
    "cancer_2nn": "tabular",
    "cancer_tabnn": "tabular",
    "cancer_2layernn": "tabular",
}

MODALITY_DATASETS = {
    "vision": ["stl10_resnet", "stl10_densenet", "cub_resnet", "cub_densenet"],
    "text": ["imdb_cnn", "imdb_2layernn", "snli_cnn", "snli_2layernn"],
    "tabular": ["adult_census", "adult_tabnn", "adult_2layernn", "cancer_2nn", "cancer_tabnn", "cancer_2layernn"],
}


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class Job:
    """Represents a single pipeline job."""
    dataset: str
    q_type: int
    question_id: int
    modality: str
    dataset_path: str
    model_path: str
    job_id: str = field(default="")

    def __post_init__(self):
        if not self.job_id:
            self.job_id = f"{self.dataset}_q{self.q_type}_{self.question_id}"


@dataclass
class JobResult:
    """Result of a job execution."""
    job: Job
    success: bool
    return_code: int
    stdout: str
    stderr: str
    duration: float
    log_file: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================
def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("batch_runner")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_format = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_file = log_dir / "batch_runner.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def parse_range(value: str) -> Optional[List[int]]:
    """Parse a range string into a list of integers.

    Supports:
      - Single IDs:  "0 1 2"
      - Ranges:      "0-4"   ->  [0, 1, 2, 3, 4]
      - Mixed:       "0 2-4" ->  [0, 2, 3, 4]
      - Auto-detect: "all"   ->  None  (count inferred per JSON at build time)
    """
    if value.strip().lower() == "all":
        return None  # sentinel: auto-detect question count from each dataset JSON
    result = []
    for part in value.split():
        if "-" in part and not part.startswith("-"):
            start, end = map(int, part.split("-"))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def get_question_count(dataset_path: str) -> int:
    """Return the number of questions in a benchmark JSON file."""
    with open(dataset_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and "questions" in data:
        return len(data["questions"])
    return 1  # single-question dict


def get_dataset_path(
    dataset: str,
    q_type: int,
    dataset_dir: str,
    modality: str,
    mode: str = "test",
    use_test_variant: bool = False
) -> Optional[str]:
    """Get the path to a dataset file.

    Benchmark JSONs live under ``dataset_dir/{mode}/{modality}/``.
    """
    base_name = f"{dataset}_q{q_type}"
    mode_dir = Path(dataset_dir) / mode / modality

    # Try _test variant first if requested
    if use_test_variant:
        test_path = mode_dir / f"{base_name}_test.json"
        if test_path.exists():
            return str(test_path)

    # Try standard path
    standard_path = mode_dir / f"{base_name}.json"
    if standard_path.exists():
        return str(standard_path)

    # Try special cases (e.g., adult_census_q4_pairs.json)
    pairs_path = mode_dir / f"{base_name}_pairs.json"
    if pairs_path.exists():
        return str(pairs_path)

    return None


def get_model_path(dataset: str, models_dir: str) -> Optional[str]:
    """Get the path to a model file."""
    if dataset not in DATASET_MODEL_MAP:
        return None

    model_rel_path = DATASET_MODEL_MAP[dataset]
    model_path = Path(models_dir) / model_rel_path

    if model_path.exists():
        return str(model_path)

    return None


def build_jobs(
    datasets: List[str],
    q_types: List[int],
    question_ids: Optional[List[int]],   # None = auto-detect from each JSON
    dataset_dir: str,
    models_dir: str,
    mode: str = "test",
    use_test_variant: bool = False,
    logger: Optional[logging.Logger] = None
) -> List[Job]:
    """Build list of jobs to execute.

    When *question_ids* is ``None`` (``--question_ids all``), the count is
    read from each benchmark JSON individually, so different datasets/q_types
    can have different lengths.
    """
    jobs = []

    for dataset in datasets:
        modality = DATASET_MODALITY_MAP.get(dataset)
        if not modality:
            if logger:
                logger.warning(f"Unknown dataset: {dataset}, skipping")
            continue

        model_path = get_model_path(dataset, models_dir)
        if not model_path:
            if logger:
                logger.warning(f"Model not found for dataset: {dataset}, skipping")
            continue

        for q_type in q_types:
            dataset_path = get_dataset_path(
                dataset, q_type, dataset_dir, modality, mode, use_test_variant
            )

            if not dataset_path:
                if logger:
                    logger.warning(f"Dataset file not found: {dataset}_q{q_type}, skipping")
                continue

            # Resolve question IDs: auto-detect when "all" was requested
            if question_ids is None:
                count = get_question_count(dataset_path)
                resolved_ids = list(range(count))
                if logger:
                    logger.info(f"  {dataset}_q{q_type}: auto-detected {count} questions")
            else:
                resolved_ids = question_ids

            for question_id in resolved_ids:
                job = Job(
                    dataset=dataset,
                    q_type=q_type,
                    question_id=question_id,
                    modality=modality,
                    dataset_path=dataset_path,
                    model_path=model_path,
                )
                jobs.append(job)

    return jobs


def run_single_job(
    job: Job,
    output_dir: str,
    dataset_dir: str,
    models_dir: str,
    vlm_model: str,
    faithfulness_threshold: float,
    no_eval: bool,
    no_improvement: bool,
    no_sf: bool,
    sf_max_samples: Optional[int],
    log_dir: Path,
    mode: str = "test",
) -> JobResult:
    """Execute a single pipeline job."""
    start_time = datetime.now()

    # Build command
    cmd = [
        sys.executable,
        str(BASE_DIR / "xai_pipeline_v2.py"),
        "--dataset", job.dataset_path,
        "--question_id", str(job.question_id),
        "--model_url", job.model_path,
        "--dataset_dir", dataset_dir,
        "--models_dir", models_dir,
        "--output_dir", output_dir,
        "--vlm", vlm_model,
        "--mode", mode,
        "--faithfulness_threshold", str(faithfulness_threshold),
    ]

    if no_eval:
        cmd.append("--no-eval")
    if no_improvement:
        cmd.append("--no-improvement")
    if no_sf:
        cmd.append("--no-sf")
    if sf_max_samples is not None:
        cmd.extend(["--sf_max_samples", str(sf_max_samples)])

    # Create log file
    log_file = log_dir / f"{job.job_id}.log"

    try:
        with open(log_file, "w") as f:
            f.write(f"Job: {job.job_id}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {start_time}\n")
            f.write("=" * 80 + "\n\n")

            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour timeout
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Finished: {end_time}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Return code: {result.returncode}\n")

        return JobResult(
            job=job,
            success=result.returncode == 0,
            return_code=result.returncode,
            stdout="",
            stderr="",
            duration=duration,
            log_file=str(log_file),
        )

    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        return JobResult(
            job=job,
            success=False,
            return_code=-1,
            stdout="",
            stderr="Timeout expired",
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


def run_jobs_sequential(
    jobs: List[Job],
    config: Dict,
    log_dir: Path,
    logger: logging.Logger,
) -> List[JobResult]:
    """Run jobs sequentially."""
    results = []

    for i, job in enumerate(jobs, 1):
        logger.info(f"Running job {i}/{len(jobs)}: {job.job_id}")

        result = run_single_job(
            job=job,
            output_dir=config["output_dir"],
            dataset_dir=config["dataset_dir"],
            models_dir=config["models_dir"],
            vlm_model=config["vlm_model"],
            faithfulness_threshold=config["faithfulness_threshold"],
            no_eval=config.get("no_eval", False),
            no_improvement=config.get("no_improvement", False),
            no_sf=config.get("no_sf", False),
            sf_max_samples=config.get("sf_max_samples"),
            log_dir=log_dir,
            mode=config.get("mode", "test"),
        )

        results.append(result)

        status = "SUCCESS" if result.success else "FAILED"
        logger.info(f"  {status} ({result.duration:.2f}s)")

    return results


def run_jobs_parallel(
    jobs: List[Job],
    config: Dict,
    log_dir: Path,
    logger: logging.Logger,
    max_workers: int = 4,
) -> List[JobResult]:
    """Run jobs in parallel."""
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                run_single_job,
                job=job,
                output_dir=config["output_dir"],
                dataset_dir=config["dataset_dir"],
                models_dir=config["models_dir"],
                vlm_model=config["vlm_model"],
                faithfulness_threshold=config["faithfulness_threshold"],
                no_eval=config.get("no_eval", False),
                no_improvement=config.get("no_improvement", False),
                no_sf=config.get("no_sf", False),
                sf_max_samples=config.get("sf_max_samples"),
                log_dir=log_dir,
                mode=config.get("mode", "test"),
            ): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                results.append(result)

                status = "SUCCESS" if result.success else "FAILED"
                logger.info(f"Job {job.job_id}: {status} ({result.duration:.2f}s)")
            except Exception as e:
                logger.error(f"Job {job.job_id} raised exception: {e}")
                results.append(JobResult(
                    job=job,
                    success=False,
                    return_code=-1,
                    stdout="",
                    stderr=str(e),
                    duration=0,
                ))

    return results


def save_summary(
    results: List[JobResult],
    log_dir: Path,
    logger: logging.Logger,
):
    """Save execution summary to a JSON file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_jobs": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "total_duration": sum(r.duration for r in results),
        "jobs": [
            {
                "job_id": r.job.job_id,
                "dataset": r.job.dataset,
                "q_type": r.job.q_type,
                "question_id": r.job.question_id,
                "success": r.success,
                "return_code": r.return_code,
                "duration": r.duration,
                "log_file": r.log_file,
            }
            for r in results
        ],
    }

    summary_file = log_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_file}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="XAI Pipeline Batch Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run vision datasets, q1-q4, first 5 questions
    python run_pipeline_batch.py --modality vision --q_types 1 2 3 4 --question_ids 0-4

    # Run specific dataset
    python run_pipeline_batch.py --datasets stl10_resnet --q_types 1 --question_ids 0

    # Run from config file
    python run_pipeline_batch.py --config batch_config.json

    # Dry run
    python run_pipeline_batch.py --datasets stl10_resnet --q_types 1 --question_ids 0 --dry_run

Available Datasets:
    Vision:  stl10_resnet, stl10_densenet, cub_resnet, cub_densenet
    Text:    imdb_cnn, imdb_2layernn, snli_cnn, snli_2layernn
    Tabular: adult_census, adult_tabnn, cancer_2nn, cancer_tabnn
        """,
    )

    # Input options
    parser.add_argument("--modality", nargs="+", choices=["vision", "text", "tabular", "all"],
                        help="Modalities to run (default: auto-detect from datasets)")
    parser.add_argument("--datasets", nargs="+",
                        help="Specific datasets to run")
    parser.add_argument("--q_types", nargs="+", type=int, default=[1],
                        help="Question types to run (default: 1)")
    parser.add_argument("--question_ids", type=str, default="0",
                        help="Question IDs to run, supports ranges like '0-4' (default: 0)")

    # Configuration options
    parser.add_argument("--config", type=str,
                        help="Path to JSON config file")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test",
                        help="Dataset split to use: 'train' loads from dataset/train/, 'test' from dataset/test/ (default: test)")
    parser.add_argument("--use_test_variant", action="store_true",
                        help="Use _test variant dataset files if available")

    # Evaluation options
    parser.add_argument("--no_eval", action="store_true",
                        help="Skip faithfulness evaluation")
    parser.add_argument("--no_improvement", action="store_true",
                        help="Skip improvement phase")
    parser.add_argument("--no_sf", action="store_true",
                        help="Skip strategy faithfulness evaluation")
    parser.add_argument("--sf_max_samples", type=int, default=None,
                        help="Max tool configs to sample for strategy faithfulness (default: None = full 2^N)")
    parser.add_argument("--faithfulness_threshold", type=float, default=0.1,
                        help="Threshold for faithfulness (default: 0.1)")

    # Execution options
    parser.add_argument("--parallel", action="store_true",
                        help="Run jobs in parallel")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Maximum parallel workers (default: 4)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")

    # Output options
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_CONFIG["output_dir"]),
                        help="Output directory")
    parser.add_argument("--vlm", type=str, default=DEFAULT_CONFIG["vlm_model"],
                        help="VLM model ID. Local: 'Qwen/Qwen3-VL-8B-Instruct'. API: 'gemini-2.5-pro', 'gemini-3-pro'")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Load config file if provided
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
            config.update(file_config)

    # Override with command line args
    config["output_dir"] = args.output_dir
    config["vlm_model"] = args.vlm
    config["faithfulness_threshold"] = args.faithfulness_threshold
    config["no_eval"] = args.no_eval
    config["no_improvement"] = args.no_improvement
    config["no_sf"] = args.no_sf
    config["sf_max_samples"] = args.sf_max_samples
    config["mode"] = args.mode

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config["output_dir"]) / "logs" / timestamp
    logger = setup_logging(log_dir, args.verbose)

    logger.info("=" * 70)
    logger.info("XAI Pipeline Batch Runner")
    logger.info("=" * 70)

    # Determine datasets to run
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
        logger.error("Must specify --datasets or --modality")
        sys.exit(1)

    # Parse question IDs
    question_ids = parse_range(args.question_ids)

    logger.info(f"Configuration:")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Mode: {config['mode']}")
    logger.info(f"  Q Types: {args.q_types}")
    logger.info(f"  Question IDs: {'all (auto-detect per JSON)' if question_ids is None else question_ids}")
    logger.info(f"  Use test variant: {args.use_test_variant}")
    logger.info(f"  Parallel: {args.parallel}")
    logger.info(f"  Output dir: {config['output_dir']}")

    # Build jobs
    jobs = build_jobs(
        datasets=datasets,
        q_types=args.q_types,
        question_ids=question_ids,
        dataset_dir=config["dataset_dir"],
        models_dir=config["models_dir"],
        mode=config["mode"],
        use_test_variant=args.use_test_variant,
        logger=logger,
    )

    logger.info(f"\nTotal jobs to run: {len(jobs)}")

    if not jobs:
        logger.warning("No valid jobs to run. Check your configuration.")
        sys.exit(1)

    # Dry run
    if args.dry_run:
        logger.info("\n--- Dry Run - Commands to execute ---")
        for job in jobs:
            cmd = (
                f"python xai_pipeline_v2.py "
                f"--dataset {job.dataset_path} "
                f"--question_id {job.question_id} "
                f"--model_url {job.model_path} "
                f"--dataset_dir {config['dataset_dir']} "
                f"--models_dir {config['models_dir']} "
                f"--mode {config['mode']}"
            )
            logger.info(f"\n{job.job_id}:")
            logger.info(f"  {cmd}")
        sys.exit(0)

    # Execute jobs
    logger.info("\n--- Executing jobs ---")

    if args.parallel:
        results = run_jobs_parallel(jobs, config, log_dir, logger, args.max_workers)
    else:
        results = run_jobs_sequential(jobs, config, log_dir, logger)

    # Save summary
    save_summary(results, log_dir, logger)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Execution Summary")
    logger.info("=" * 70)
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_duration = sum(r.duration for r in results)

    logger.info(f"Total jobs: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total duration: {total_duration:.2f}s")
    logger.info(f"Logs saved to: {log_dir}")

    if failed > 0:
        logger.warning("Some jobs failed. Check logs for details.")
        logger.info("Failed jobs:")
        for r in results:
            if not r.success:
                logger.info(f"  - {r.job.job_id}: {r.log_file}")
        sys.exit(1)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
